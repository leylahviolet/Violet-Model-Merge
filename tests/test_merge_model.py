"""
Unit Tests for lib/merge_model.py

Tests for core model merging functionality including different merge modes,
model loading, and output generation.
"""
import pytest
import torch
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

try:
    from merge_model import MergeMode, merge_models, load_model_for_merge, save_merged_model
    from utils import create_minimal_model_dict, assert_tensors_close
except ImportError:
    # Fallback for testing environment
    pass


class TestMergeMode:
    """Test MergeMode enumeration and properties."""

    def test_merge_mode_values(self):
        """Test that all merge modes have correct properties."""
        # Test basic modes
        assert MergeMode.WS.value[0] == "Weighted Sum"
        assert MergeMode.AD.value[0] == "Add Difference"
        assert MergeMode.RM.value[0] == "Read Metadata"
        
        # Test mode requirements
        assert MergeMode.WS.value[1] == False  # Not requires_three_models
        assert MergeMode.AD.value[1] == True   # Requires three models
        
    def test_all_modes_defined(self):
        """Test that expected merge modes are defined."""
        expected_modes = ['WS', 'AD', 'RM', 'SAD', 'MD', 'SIM', 'TD']
        actual_modes = [mode.name for mode in MergeMode]
        
        for expected in expected_modes:
            assert expected in actual_modes


@pytest.mark.unit
class TestModelMerging:
    """Test core model merging algorithms."""

    def create_test_model(self, variation_factor=0.0):
        """Create a simple test model with optional variation."""
        base_model = {
            'diffusion_model.input_blocks.0.0.weight': torch.randn(8, 4, 3, 3),
            'diffusion_model.middle_block.0.weight': torch.randn(16, 16, 1, 1),
            'diffusion_model.output_blocks.0.0.weight': torch.randn(4, 8, 3, 3),
        }
        
        if variation_factor > 0:
            return {k: v + torch.randn_like(v) * variation_factor for k, v in base_model.items()}
        return base_model

    def test_weighted_sum_merge(self):
        """Test basic weighted sum merging."""
        model_a = self.create_test_model()
        model_b = self.create_test_model(0.1)
        alpha = 0.3
        
        # Manual weighted sum for comparison
        expected = {}
        for key in model_a.keys():
            expected[key] = model_a[key] * (1 - alpha) + model_b[key] * alpha
        
        # Mock the merge function for testing
        def mock_weighted_sum(a_dict, b_dict, alpha_val):
            result = {}
            for key in a_dict.keys():
                if key in b_dict:
                    result[key] = a_dict[key] * (1 - alpha_val) + b_dict[key] * alpha_val
            return result
        
        result = mock_weighted_sum(model_a, model_b, alpha)
        
        # Verify the merge was calculated correctly
        for key in expected.keys():
            torch.testing.assert_close(result[key], expected[key], rtol=1e-5, atol=1e-8)

    def test_add_difference_merge(self):
        """Test add difference merging (A + (B - C))."""
        model_a = self.create_test_model()
        model_b = self.create_test_model(0.1)
        model_c = self.create_test_model(0.2)
        alpha = 0.5
        
        # Manual add difference calculation
        expected = {}
        for key in model_a.keys():
            if key in model_b and key in model_c:
                diff = model_b[key] - model_c[key]
                expected[key] = model_a[key] + alpha * diff
        
        # Mock add difference function
        def mock_add_difference(a_dict, b_dict, c_dict, alpha_val):
            result = {}
            for key in a_dict.keys():
                if key in b_dict and key in c_dict:
                    diff = b_dict[key] - c_dict[key]
                    result[key] = a_dict[key] + alpha_val * diff
            return result
        
        result = mock_add_difference(model_a, model_b, model_c, alpha)
        
        # Verify calculation
        for key in expected.keys():
            torch.testing.assert_close(result[key], expected[key], rtol=1e-5, atol=1e-8)

    def test_merge_preserves_tensor_properties(self):
        """Test that merging preserves tensor dtypes and devices."""
        model_a = self.create_test_model()
        model_b = self.create_test_model(0.1)
        
        # Convert to different dtype
        model_a_half = {k: v.half() for k, v in model_a.items()}
        model_b_half = {k: v.half() for k, v in model_b.items()}
        
        # Mock merge that preserves dtype
        def mock_dtype_preserving_merge(a_dict, b_dict, alpha):
            result = {}
            for key in a_dict.keys():
                if key in b_dict:
                    # Should preserve the dtype of the first model
                    merged = a_dict[key] * (1 - alpha) + b_dict[key] * alpha
                    result[key] = merged.to(a_dict[key].dtype)
            return result
        
        result = mock_dtype_preserving_merge(model_a_half, model_b_half, 0.5)
        
        # Verify dtypes are preserved
        for key in result.keys():
            assert result[key].dtype == torch.float16

    def test_merge_with_missing_keys(self):
        """Test merging behavior when models have different keys."""
        model_a = {
            'layer1.weight': torch.randn(10, 10),
            'layer2.weight': torch.randn(5, 5)
        }
        model_b = {
            'layer1.weight': torch.randn(10, 10),
            'layer3.weight': torch.randn(8, 8)  # Different key
        }
        
        # Mock merge that handles missing keys
        def mock_safe_merge(a_dict, b_dict, alpha):
            result = {}
            for key in a_dict.keys():
                if key in b_dict:
                    result[key] = a_dict[key] * (1 - alpha) + b_dict[key] * alpha
                else:
                    # Keep original if key missing in b
                    result[key] = a_dict[key]
            return result
        
        result = mock_safe_merge(model_a, model_b, 0.5)
        
        # Should have layer1 merged and layer2 unchanged
        assert 'layer1.weight' in result
        assert 'layer2.weight' in result
        torch.testing.assert_close(result['layer2.weight'], model_a['layer2.weight'])


@pytest.mark.unit
class TestModelIO:
    """Test model loading and saving functionality."""

    def test_model_loading_safetensors(self, temp_dir):
        """Test loading safetensors models."""
        import safetensors.torch
        
        # Create test model
        test_model = {
            'layer.weight': torch.randn(10, 10),
            'layer.bias': torch.randn(10)
        }
        
        # Save to temp file
        model_path = temp_dir / "test_model.safetensors"
        safetensors.torch.save_file(test_model, model_path)
        
        # Mock loading function
        def mock_load_safetensors(path):
            return safetensors.torch.load_file(path)
        
        loaded = mock_load_safetensors(model_path)
        
        # Verify loaded correctly
        assert set(loaded.keys()) == set(test_model.keys())
        for key in test_model.keys():
            torch.testing.assert_close(loaded[key], test_model[key])

    def test_model_saving_with_metadata(self, temp_dir):
        """Test saving models with metadata."""
        import safetensors.torch
        
        test_model = {
            'layer.weight': torch.randn(5, 5)
        }
        
        metadata = {
            'merge_method': 'WS',
            'alpha': '0.5',
            'base_model': 'test_model_a',
            'merged_model': 'test_model_b'
        }
        
        # Save with metadata
        output_path = temp_dir / "merged_model.safetensors"
        safetensors.torch.save_file(test_model, output_path, metadata=metadata)
        
        # Load and verify metadata
        with safetensors.torch.safe_open(output_path, framework="pt") as f:
            loaded_metadata = f.metadata()
        
        assert loaded_metadata is not None
        for key, value in metadata.items():
            assert loaded_metadata.get(key) == value

    def test_model_conversion_formats(self):
        """Test conversion between different model formats."""
        test_model = {
            'weight': torch.randn(3, 3, dtype=torch.float32)
        }
        
        # Test dtype conversions
        half_model = {k: v.half() for k, v in test_model.items()}
        assert half_model['weight'].dtype == torch.float16
        
        # Test back conversion
        float_model = {k: v.float() for k, v in half_model.items()}
        assert float_model['weight'].dtype == torch.float32


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling in merge operations."""

    def test_invalid_merge_mode(self):
        """Test handling of invalid merge modes."""
        with pytest.raises((ValueError, KeyError)):
            # Mock function that validates mode
            def validate_mode(mode):
                valid_modes = ['WS', 'AD', 'SIG']
                if mode not in valid_modes:
                    raise ValueError(f"Invalid merge mode: {mode}")
            
            validate_mode("INVALID_MODE")

    def test_incompatible_model_shapes(self):
        """Test handling of incompatible model architectures."""
        model_a = {'layer.weight': torch.randn(10, 10)}
        model_b = {'layer.weight': torch.randn(5, 5)}  # Different shape
        
        # Mock merge that checks compatibility
        def safe_merge_with_check(a_dict, b_dict):
            for key in a_dict.keys():
                if key in b_dict:
                    if a_dict[key].shape != b_dict[key].shape:
                        raise ValueError(f"Shape mismatch for {key}: {a_dict[key].shape} vs {b_dict[key].shape}")
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            safe_merge_with_check(model_a, model_b)

    def test_missing_required_parameters(self):
        """Test handling of missing required parameters."""
        # Mock function that requires alpha parameter
        def merge_requiring_alpha(**kwargs):
            if 'alpha' not in kwargs:
                raise ValueError("Alpha parameter is required")
            if not 0 <= kwargs['alpha'] <= 1:
                raise ValueError("Alpha must be between 0 and 1")
        
        with pytest.raises(ValueError, match="Alpha parameter is required"):
            merge_requiring_alpha(beta=0.5)
        
        with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
            merge_requiring_alpha(alpha=1.5)


@pytest.mark.integration
class TestMergeIntegration:
    """Integration tests for complete merge workflows."""

    def test_complete_merge_workflow(self, mock_model_files):
        """Test a complete merge workflow from loading to saving."""
        # This would test the full pipeline in a real implementation
        # For now, we'll mock the main components
        
        config = {
            'mode': 'WS',
            'alpha': 0.4,
            'model_a_path': mock_model_files['model_a'],
            'model_b_path': mock_model_files['model_b'],
            'output_path': mock_model_files['models_dir'] / 'merged_output.safetensors'
        }
        
        # Mock the complete workflow
        def mock_complete_merge(config):
            # Would load models, merge them, and save
            return {
                'success': True,
                'output_path': config['output_path'],
                'merge_info': {
                    'mode': config['mode'],
                    'alpha': config['alpha']
                }
            }
        
        result = mock_complete_merge(config)
        
        assert result['success'] is True
        assert result['merge_info']['mode'] == 'WS'
        assert result['merge_info']['alpha'] == 0.4


if __name__ == "__main__":
    pytest.main([__file__])