"""
Unit Tests for VAE Operations

Tests for VAE merging, baking, and related functionality.
"""
import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))


@pytest.mark.unit
class TestVAEOperations:
    """Test VAE merging and baking functionality."""

    def create_mock_vae(self, variation=0.0):
        """Create a mock VAE model structure."""
        base_vae = {
            'decoder.conv_out.weight': torch.randn(3, 512, 3, 3),
            'decoder.conv_out.bias': torch.randn(3),
            'encoder.conv_in.weight': torch.randn(512, 3, 3, 3),
            'encoder.conv_in.bias': torch.randn(512),
            'quant_conv.weight': torch.randn(8, 8, 1, 1),
            'post_quant_conv.weight': torch.randn(4, 4, 1, 1)
        }
        
        if variation > 0:
            return {k: v + torch.randn_like(v) * variation for k, v in base_vae.items()}
        return base_vae

    def create_mock_model_with_vae(self):
        """Create a mock model that includes VAE components."""
        return {
            # Diffusion model parts
            'model.diffusion_model.input_blocks.0.0.weight': torch.randn(320, 4, 3, 3),
            'model.diffusion_model.middle_block.0.weight': torch.randn(1280, 1280, 3, 3),
            
            # VAE parts (first_stage_model)
            'first_stage_model.decoder.conv_out.weight': torch.randn(3, 512, 3, 3),
            'first_stage_model.encoder.conv_in.weight': torch.randn(512, 3, 3, 3),
            
            # Text encoder parts
            'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight': torch.randn(49408, 768)
        }

    def test_vae_extraction(self):
        """Test extracting VAE from a full model."""
        full_model = self.create_mock_model_with_vae()
        
        # Mock VAE extraction function
        def extract_vae_keys(model_dict):
            vae_keys = {}
            for key, tensor in model_dict.items():
                if key.startswith('first_stage_model.'):
                    # Remove the 'first_stage_model.' prefix
                    vae_key = key.replace('first_stage_model.', '')
                    vae_keys[vae_key] = tensor
            return vae_keys
        
        extracted_vae = extract_vae_keys(full_model)
        
        # Should have VAE components without the prefix
        assert 'decoder.conv_out.weight' in extracted_vae
        assert 'encoder.conv_in.weight' in extracted_vae
        
        # Should not have diffusion model parts
        assert not any(key.startswith('model.diffusion_model') for key in extracted_vae.keys())

    def test_vae_baking_into_model(self):
        """Test baking a standalone VAE into a model."""
        model = self.create_mock_model_with_vae()
        standalone_vae = self.create_mock_vae()
        
        # Mock VAE baking function
        def bake_vae_into_model(model_dict, vae_dict):
            result = model_dict.copy()
            
            # Replace first_stage_model parts with new VAE
            # Remove existing VAE parts
            result = {k: v for k, v in result.items() if not k.startswith('first_stage_model.')}
            
            # Add new VAE with proper prefix
            for vae_key, vae_tensor in vae_dict.items():
                model_key = f'first_stage_model.{vae_key}'
                result[model_key] = vae_tensor
            
            return result
        
        baked_model = bake_vae_into_model(model, standalone_vae)
        
        # Should have new VAE components
        assert 'first_stage_model.decoder.conv_out.weight' in baked_model
        assert 'first_stage_model.encoder.conv_in.weight' in baked_model
        
        # Should preserve diffusion model parts
        assert 'model.diffusion_model.input_blocks.0.0.weight' in baked_model
        
        # VAE tensors should match the standalone VAE
        torch.testing.assert_close(
            baked_model['first_stage_model.decoder.conv_out.weight'],
            standalone_vae['decoder.conv_out.weight']
        )

    def test_vae_merging(self):
        """Test merging two VAE models."""
        vae_a = self.create_mock_vae()
        vae_b = self.create_mock_vae(0.1)
        alpha = 0.6
        
        # Mock VAE merging function
        def merge_vaes(vae_a_dict, vae_b_dict, alpha_val):
            merged = {}
            for key in vae_a_dict.keys():
                if key in vae_b_dict:
                    merged[key] = vae_a_dict[key] * (1 - alpha_val) + vae_b_dict[key] * alpha_val
                else:
                    merged[key] = vae_a_dict[key]
            return merged
        
        merged_vae = merge_vaes(vae_a, vae_b, alpha)
        
        # Verify merge calculation
        for key in vae_a.keys():
            if key in vae_b:
                expected = vae_a[key] * (1 - alpha) + vae_b[key] * alpha
                torch.testing.assert_close(merged_vae[key], expected, rtol=1e-5, atol=1e-8)

    def test_vae_format_conversion(self):
        """Test VAE format conversion (e.g., .pt to .safetensors)."""
        vae_dict = self.create_mock_vae()
        
        # Test dtype conversion
        half_vae = {k: v.half() for k, v in vae_dict.items()}
        assert all(tensor.dtype == torch.float16 for tensor in half_vae.values())
        
        # Test back to float
        float_vae = {k: v.float() for k, v in half_vae.items()}
        assert all(tensor.dtype == torch.float32 for tensor in float_vae.values())

    def test_vae_compatibility_check(self):
        """Test checking VAE compatibility with models."""
        model = self.create_mock_model_with_vae()
        compatible_vae = self.create_mock_vae()
        
        # Mock compatibility check
        def check_vae_compatibility(model_dict, vae_dict):
            # Check if model has VAE slots
            has_vae_slots = any(key.startswith('first_stage_model.') for key in model_dict.keys())
            
            # Check if VAE has required components
            required_vae_components = ['decoder.conv_out.weight', 'encoder.conv_in.weight']
            has_required_components = all(comp in vae_dict for comp in required_vae_components)
            
            return has_vae_slots and has_required_components
        
        # Should be compatible
        assert check_vae_compatibility(model, compatible_vae) is True
        
        # Test with incomplete VAE
        incomplete_vae = {'decoder.conv_out.weight': torch.randn(3, 512, 3, 3)}
        assert check_vae_compatibility(model, incomplete_vae) is False

    def test_vae_metadata_preservation(self):
        """Test that VAE metadata is preserved during operations."""
        vae_with_metadata = {
            'decoder.conv_out.weight': torch.randn(3, 512, 3, 3),
            'encoder.conv_in.weight': torch.randn(512, 3, 3, 3)
        }
        
        metadata = {
            'vae_type': 'test_vae',
            'architecture': 'standard',
            'resolution': '512x512'
        }
        
        # Mock function that preserves metadata
        def process_vae_with_metadata(vae_dict, meta):
            processed = {k: v.clone() for k, v in vae_dict.items()}
            return processed, meta.copy()
        
        processed_vae, preserved_meta = process_vae_with_metadata(vae_with_metadata, metadata)
        
        # Metadata should be preserved
        assert preserved_meta == metadata
        
        # VAE should be processed correctly
        assert set(processed_vae.keys()) == set(vae_with_metadata.keys())


@pytest.mark.unit
class TestVAEErrorHandling:
    """Test error handling in VAE operations."""

    def test_missing_vae_components(self):
        """Test handling of missing VAE components."""
        incomplete_vae = {
            'decoder.conv_out.weight': torch.randn(3, 512, 3, 3)
            # Missing encoder components
        }
        
        # Mock validation function
        def validate_vae_completeness(vae_dict):
            required = ['decoder.conv_out.weight', 'encoder.conv_in.weight']
            missing = [comp for comp in required if comp not in vae_dict]
            if missing:
                raise ValueError(f"Missing VAE components: {missing}")
        
        with pytest.raises(ValueError, match="Missing VAE components"):
            validate_vae_completeness(incomplete_vae)

    def test_incompatible_vae_architecture(self):
        """Test handling of incompatible VAE architectures."""
        model_vae_shape = torch.randn(3, 512, 3, 3)
        incompatible_vae_shape = torch.randn(3, 256, 3, 3)  # Different channel count
        
        # Mock compatibility check
        def check_vae_shape_compatibility(model_tensor, vae_tensor):
            if model_tensor.shape != vae_tensor.shape:
                raise ValueError(f"VAE shape incompatible: expected {model_tensor.shape}, got {vae_tensor.shape}")
        
        with pytest.raises(ValueError, match="VAE shape incompatible"):
            check_vae_shape_compatibility(model_vae_shape, incompatible_vae_shape)

    def test_vae_loading_errors(self, temp_dir):
        """Test handling of VAE loading errors."""
        nonexistent_path = temp_dir / "nonexistent_vae.safetensors"
        
        # Mock loading function with error handling
        def safe_load_vae(path):
            if not path.exists():
                raise FileNotFoundError(f"VAE file not found: {path}")
        
        with pytest.raises(FileNotFoundError, match="VAE file not found"):
            safe_load_vae(nonexistent_path)


@pytest.mark.integration
class TestVAEIntegration:
    """Integration tests for VAE operations."""

    def test_complete_vae_merge_workflow(self, temp_dir):
        """Test complete VAE merging workflow."""
        import safetensors.torch
        
        # Create test VAEs
        vae_a = {
            'decoder.conv_out.weight': torch.randn(3, 512, 3, 3),
            'encoder.conv_in.weight': torch.randn(512, 3, 3, 3)
        }
        vae_b = {
            'decoder.conv_out.weight': torch.randn(3, 512, 3, 3),
            'encoder.conv_in.weight': torch.randn(512, 3, 3, 3)
        }
        
        # Save test VAEs
        vae_a_path = temp_dir / "vae_a.safetensors"
        vae_b_path = temp_dir / "vae_b.safetensors"
        output_path = temp_dir / "merged_vae.safetensors"
        
        safetensors.torch.save_file(vae_a, vae_a_path)
        safetensors.torch.save_file(vae_b, vae_b_path)
        
        # Mock complete VAE merge workflow
        def complete_vae_merge(vae_a_path, vae_b_path, output_path, alpha=0.5):
            # Load VAEs
            vae_a_loaded = safetensors.torch.load_file(vae_a_path)
            vae_b_loaded = safetensors.torch.load_file(vae_b_path)
            
            # Merge
            merged = {}
            for key in vae_a_loaded.keys():
                if key in vae_b_loaded:
                    merged[key] = vae_a_loaded[key] * (1 - alpha) + vae_b_loaded[key] * alpha
            
            # Save
            safetensors.torch.save_file(merged, output_path)
            return output_path
        
        result_path = complete_vae_merge(vae_a_path, vae_b_path, output_path, alpha=0.3)
        
        # Verify output exists and is correct
        assert result_path.exists()
        
        # Load and verify
        merged_result = safetensors.torch.load_file(result_path)
        assert set(merged_result.keys()) == set(vae_a.keys())

    def test_vae_baking_integration(self, temp_dir):
        """Test VAE baking into model integration."""
        import safetensors.torch
        
        # Create model with existing VAE
        model_with_vae = {
            'model.diffusion_model.input_blocks.0.0.weight': torch.randn(320, 4, 3, 3),
            'first_stage_model.decoder.conv_out.weight': torch.randn(3, 512, 3, 3),
            'first_stage_model.encoder.conv_in.weight': torch.randn(512, 3, 3, 3)
        }
        
        # Create standalone VAE
        new_vae = {
            'decoder.conv_out.weight': torch.randn(3, 512, 3, 3),
            'encoder.conv_in.weight': torch.randn(512, 3, 3, 3)
        }
        
        # Save files
        model_path = temp_dir / "model.safetensors"
        vae_path = temp_dir / "new_vae.safetensors"
        output_path = temp_dir / "model_with_new_vae.safetensors"
        
        safetensors.torch.save_file(model_with_vae, model_path)
        safetensors.torch.save_file(new_vae, vae_path)
        
        # Mock baking workflow
        def bake_vae_workflow(model_path, vae_path, output_path):
            # Load files
            model = safetensors.torch.load_file(model_path)
            vae = safetensors.torch.load_file(vae_path)
            
            # Remove old VAE
            model_no_vae = {k: v for k, v in model.items() if not k.startswith('first_stage_model.')}
            
            # Add new VAE
            for vae_key, vae_tensor in vae.items():
                model_key = f'first_stage_model.{vae_key}'
                model_no_vae[model_key] = vae_tensor
            
            # Save
            safetensors.torch.save_file(model_no_vae, output_path)
            return output_path
        
        result_path = bake_vae_workflow(model_path, vae_path, output_path)
        
        # Verify result
        assert result_path.exists()
        result_model = safetensors.torch.load_file(result_path)
        
        # Should have diffusion model parts
        assert 'model.diffusion_model.input_blocks.0.0.weight' in result_model
        
        # Should have new VAE
        assert 'first_stage_model.decoder.conv_out.weight' in result_model
        torch.testing.assert_close(
            result_model['first_stage_model.decoder.conv_out.weight'],
            new_vae['decoder.conv_out.weight']
        )


if __name__ == "__main__":
    pytest.main([__file__])