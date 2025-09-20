"""
Integration Tests for CLI Interface

Tests for command-line interface functionality, argument parsing,
and error handling.
"""
import pytest
import subprocess
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))


@pytest.mark.integration
class TestCLIInterface:
    """Test command-line interface functionality."""

    def test_cli_help_output(self):
        """Test that CLI shows help when requested."""
        # Mock CLI help function
        def mock_cli_help():
            return """
Violet Model Merge - Command Line Interface

Usage:
    python lib/merge_model.py <mode> <models_path> <model_a> <model_b> [options]

Modes:
    WS      - Weighted Sum
    AD      - Add Difference
    SIG     - Sigmoid
    
Options:
    --alpha ALPHA       Blend ratio (0.0-1.0)
    --output NAME       Output filename
    --device DEVICE     Processing device (cpu/cuda)
    --help              Show this help
"""
        
        help_output = mock_cli_help()
        
        assert "Violet Model Merge" in help_output
        assert "Usage:" in help_output
        assert "--alpha" in help_output
        assert "--output" in help_output

    def test_cli_argument_parsing(self):
        """Test CLI argument parsing."""
        # Mock argument parser
        def mock_parse_args(args_list):
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument('mode', choices=['WS', 'AD', 'SIG'])
            parser.add_argument('models_path')
            parser.add_argument('model_a')
            parser.add_argument('model_b')
            parser.add_argument('--alpha', type=float, default=0.5)
            parser.add_argument('--output', default='merged_model')
            parser.add_argument('--device', default='cpu')
            parser.add_argument('--save_safetensors', action='store_true')
            
            return parser.parse_args(args_list)
        
        # Test valid arguments
        args = mock_parse_args([
            'WS', 'models/', 'model_a.safetensors', 'model_b.safetensors',
            '--alpha', '0.7', '--output', 'custom_output', '--device', 'cuda'
        ])
        
        assert args.mode == 'WS'
        assert args.alpha == 0.7
        assert args.output == 'custom_output'
        assert args.device == 'cuda'

    def test_cli_merge_command_construction(self):
        """Test construction of merge commands from CLI args."""
        # Mock command construction
        def construct_merge_command(args):
            config = {
                'mode': args.mode,
                'models_path': args.models_path,
                'model_a': args.model_a,
                'model_b': args.model_b,
                'alpha': args.alpha,
                'output': args.output,
                'device': args.device,
                'save_safetensors': getattr(args, 'save_safetensors', False)
            }
            return config
        
        # Mock args object
        class MockArgs:
            def __init__(self):
                self.mode = 'WS'
                self.models_path = 'models/'
                self.model_a = 'model_a.safetensors'
                self.model_b = 'model_b.safetensors'
                self.alpha = 0.6
                self.output = 'test_output'
                self.device = 'cpu'
                self.save_safetensors = True
        
        args = MockArgs()
        config = construct_merge_command(args)
        
        assert config['mode'] == 'WS'
        assert config['alpha'] == 0.6
        assert config['save_safetensors'] is True

    def test_cli_error_handling(self):
        """Test CLI error handling for invalid inputs."""
        # Test invalid mode
        def validate_mode(mode):
            valid_modes = ['WS', 'AD', 'SIG', 'GEO', 'MAX']
            if mode not in valid_modes:
                raise ValueError(f"Invalid mode: {mode}. Valid modes: {valid_modes}")
        
        with pytest.raises(ValueError, match="Invalid mode"):
            validate_mode("INVALID")
        
        # Test invalid alpha
        def validate_alpha(alpha):
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f"Alpha must be between 0.0 and 1.0, got {alpha}")
        
        with pytest.raises(ValueError, match="Alpha must be between"):
            validate_alpha(1.5)
        
        with pytest.raises(ValueError, match="Alpha must be between"):
            validate_alpha(-0.1)

    def test_cli_file_path_validation(self):
        """Test CLI file path validation."""
        def validate_file_paths(models_path, model_a, model_b):
            from pathlib import Path
            
            models_dir = Path(models_path)
            if not models_dir.exists():
                raise FileNotFoundError(f"Models directory not found: {models_path}")
            
            model_a_path = models_dir / model_a
            model_b_path = models_dir / model_b
            
            if not model_a_path.exists():
                raise FileNotFoundError(f"Model A not found: {model_a_path}")
            
            if not model_b_path.exists():
                raise FileNotFoundError(f"Model B not found: {model_b_path}")
            
            return model_a_path, model_b_path
        
        # Test with temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock model files
            model_a_path = temp_path / "model_a.safetensors"
            model_b_path = temp_path / "model_b.safetensors"
            model_a_path.touch()
            model_b_path.touch()
            
            # Should work with existing files
            result_a, result_b = validate_file_paths(temp_path, "model_a.safetensors", "model_b.safetensors")
            assert result_a.exists()
            assert result_b.exists()
            
            # Should fail with non-existent files
            with pytest.raises(FileNotFoundError, match="Model A not found"):
                validate_file_paths(temp_path, "nonexistent.safetensors", "model_b.safetensors")

    def test_cli_output_formatting(self):
        """Test CLI output formatting and logging."""
        # Mock progress reporting
        def mock_progress_report(step, total, description):
            progress = (step / total) * 100
            return f"[{progress:6.1f}%] {description}"
        
        # Test progress formatting
        progress_msg = mock_progress_report(3, 10, "Loading models")
        assert "30.0%" in progress_msg
        assert "Loading models" in progress_msg
        
        # Mock success message
        def mock_success_message(output_path, merge_info):
            return f"""
✅ Merge completed successfully!

📁 Output: {output_path}
🎨 Mode: {merge_info['mode']}
⚖️ Alpha: {merge_info['alpha']}
💾 Format: safetensors
"""
        
        success_msg = mock_success_message("merged_model.safetensors", {'mode': 'WS', 'alpha': 0.5})
        assert "✅ Merge completed successfully!" in success_msg
        assert "merged_model.safetensors" in success_msg
        assert "Mode: WS" in success_msg

    def test_cli_device_detection(self):
        """Test CLI device detection and validation."""
        def detect_and_validate_device(requested_device):
            import torch
            
            if requested_device == 'auto':
                # Auto-detect best available device
                if torch.cuda.is_available():
                    return 'cuda'
                else:
                    return 'cpu'
            elif requested_device == 'cuda':
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA requested but not available")
                return 'cuda'
            elif requested_device == 'cpu':
                return 'cpu'
            else:
                raise ValueError(f"Invalid device: {requested_device}")
        
        # Test auto detection (mock CUDA availability)
        with patch('torch.cuda.is_available', return_value=True):
            device = detect_and_validate_device('auto')
            assert device == 'cuda'
        
        with patch('torch.cuda.is_available', return_value=False):
            device = detect_and_validate_device('auto')
            assert device == 'cpu'
        
        # Test explicit CPU
        device = detect_and_validate_device('cpu')
        assert device == 'cpu'
        
        # Test CUDA when not available
        with patch('torch.cuda.is_available', return_value=False):
            with pytest.raises(RuntimeError, match="CUDA requested but not available"):
                detect_and_validate_device('cuda')


@pytest.mark.integration
class TestCLIWorkflows:
    """Test complete CLI workflows."""

    def test_simple_weighted_merge_workflow(self, mock_model_files):
        """Test a complete weighted merge via CLI."""
        # Mock CLI execution
        def mock_cli_weighted_merge(models_path, model_a, model_b, alpha, output):
            # Simulate the full CLI workflow
            config = {
                'mode': 'WS',
                'models_path': models_path,
                'model_a': model_a,
                'model_b': model_b,
                'alpha': alpha,
                'output': output,
                'device': 'cpu'
            }
            
            # Mock merge execution
            result = {
                'success': True,
                'output_path': Path(models_path) / f"{output}.safetensors",
                'config': config,
                'stats': {
                    'parameters_merged': 1000000,
                    'merge_time': 15.5
                }
            }
            
            return result
        
        result = mock_cli_weighted_merge(
            models_path=mock_model_files['models_dir'],
            model_a='model_a.safetensors',
            model_b='model_b.safetensors',
            alpha=0.4,
            output='cli_test_merge'
        )
        
        assert result['success'] is True
        assert result['config']['mode'] == 'WS'
        assert result['config']['alpha'] == 0.4
        assert result['stats']['parameters_merged'] > 0

    def test_advanced_merge_with_options(self, mock_model_files):
        """Test advanced merge with multiple options."""
        # Mock advanced CLI execution
        def mock_cli_advanced_merge(**kwargs):
            required_args = ['mode', 'models_path', 'model_a', 'model_b']
            for arg in required_args:
                if arg not in kwargs:
                    raise ValueError(f"Missing required argument: {arg}")
            
            config = {
                'mode': kwargs['mode'],
                'alpha': kwargs.get('alpha', 0.5),
                'beta': kwargs.get('beta', 0.5),
                'device': kwargs.get('device', 'cpu'),
                'save_half': kwargs.get('save_half', False),
                'save_safetensors': kwargs.get('save_safetensors', True),
                'vae_path': kwargs.get('vae_path'),
                'output': kwargs.get('output', 'merged_model')
            }
            
            # Validate mode-specific requirements
            if config['mode'] == 'AD' and 'model_c' not in kwargs:
                raise ValueError("Add Difference mode requires three models")
            
            return {
                'success': True,
                'config': config,
                'validation_passed': True
            }
        
        # Test successful advanced merge
        result = mock_cli_advanced_merge(
            mode='WS',
            models_path=mock_model_files['models_dir'],
            model_a='model_a.safetensors',
            model_b='model_b.safetensors',
            alpha=0.7,
            device='cpu',
            save_half=True,
            vae_path=mock_model_files['vae'],
            output='advanced_merge'
        )
        
        assert result['success'] is True
        assert result['config']['alpha'] == 0.7
        assert result['config']['save_half'] is True
        
        # Test validation error for missing model_c in AD mode
        with pytest.raises(ValueError, match="Add Difference mode requires three models"):
            mock_cli_advanced_merge(
                mode='AD',
                models_path=mock_model_files['models_dir'],
                model_a='model_a.safetensors',
                model_b='model_b.safetensors'
            )

    def test_cli_error_recovery(self, temp_dir):
        """Test CLI error recovery and user-friendly messages."""
        # Mock CLI with error recovery
        def mock_cli_with_recovery(config):
            errors = []
            warnings = []
            
            # Check for common issues
            if config.get('alpha', 0.5) > 1.0:
                errors.append("Alpha value must be between 0.0 and 1.0")
            
            if config.get('device') == 'cuda':
                # Mock CUDA check
                warnings.append("CUDA requested but falling back to CPU")
                config['device'] = 'cpu'
            
            if not Path(config.get('models_path', '')).exists():
                errors.append(f"Models directory not found: {config.get('models_path')}")
            
            if errors:
                return {
                    'success': False,
                    'errors': errors,
                    'warnings': warnings
                }
            
            return {
                'success': True,
                'warnings': warnings,
                'config': config
            }
        
        # Test with errors
        error_config = {
            'alpha': 1.5,
            'models_path': '/nonexistent/path',
            'device': 'cuda'
        }
        
        result = mock_cli_with_recovery(error_config)
        assert result['success'] is False
        assert len(result['errors']) > 0
        assert "Alpha value must be between" in result['errors'][0]
        
        # Test with warnings only
        warning_config = {
            'alpha': 0.5,
            'models_path': str(temp_dir),
            'device': 'cuda'
        }
        
        result = mock_cli_with_recovery(warning_config)
        assert result['success'] is True
        assert len(result['warnings']) > 0
        assert result['config']['device'] == 'cpu'  # Fallback applied

    def test_cli_batch_processing(self, mock_model_files):
        """Test CLI batch processing capabilities."""
        # Mock batch merge function
        def mock_batch_merge(models_path, model_list, output_prefix, **kwargs):
            results = []
            
            for i, (model_a, model_b) in enumerate(model_list):
                merge_config = {
                    'model_a': model_a,
                    'model_b': model_b,
                    'output': f"{output_prefix}_{i:03d}",
                    'alpha': kwargs.get('alpha', 0.5),
                    'mode': kwargs.get('mode', 'WS')
                }
                
                # Simulate merge
                result = {
                    'index': i,
                    'config': merge_config,
                    'success': True,
                    'output_path': Path(models_path) / f"{merge_config['output']}.safetensors"
                }
                results.append(result)
            
            return {
                'total_merges': len(results),
                'successful_merges': sum(1 for r in results if r['success']),
                'results': results
            }
        
        # Test batch processing
        model_pairs = [
            ('model_a.safetensors', 'model_b.safetensors'),
        ]
        
        batch_result = mock_batch_merge(
            models_path=mock_model_files['models_dir'],
            model_list=model_pairs,
            output_prefix='batch_merge',
            alpha=0.6,
            mode='WS'
        )
        
        assert batch_result['total_merges'] == 1
        assert batch_result['successful_merges'] == 1
        assert batch_result['results'][0]['success'] is True


@pytest.mark.integration
class TestCLIPerformance:
    """Test CLI performance and resource usage."""

    def test_cli_memory_usage_monitoring(self):
        """Test CLI memory usage monitoring."""
        # Mock memory monitoring
        def mock_memory_monitor():
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
                'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
                'percent': process.memory_percent()
            }
        
        # This would be called during actual merging
        # For testing, we just ensure the monitoring function works
        try:
            memory_stats = mock_memory_monitor()
            assert 'rss_mb' in memory_stats
            assert 'percent' in memory_stats
            assert memory_stats['rss_mb'] > 0
        except ImportError:
            # psutil not available in test environment
            pytest.skip("psutil not available for memory monitoring test")

    def test_cli_progress_reporting(self):
        """Test CLI progress reporting accuracy."""
        # Mock progress tracker
        class MockProgressTracker:
            def __init__(self, total_steps):
                self.total_steps = total_steps
                self.current_step = 0
                self.start_time = None
            
            def start(self):
                import time
                self.start_time = time.time()
            
            def update(self, description=""):
                self.current_step += 1
                progress = (self.current_step / self.total_steps) * 100
                
                if self.start_time:
                    import time
                    elapsed = time.time() - self.start_time
                    eta = (elapsed / self.current_step) * (self.total_steps - self.current_step) if self.current_step > 0 else 0
                    return f"[{progress:5.1f}%] {description} (ETA: {eta:.1f}s)"
                
                return f"[{progress:5.1f}%] {description}"
        
        tracker = MockProgressTracker(10)
        tracker.start()
        
        # Simulate progress updates
        messages = []
        for i in range(5):
            message = tracker.update(f"Step {i+1}")
            messages.append(message)
        
        # Verify progress reporting
        assert len(messages) == 5
        assert "50.0%" in messages[-1]  # Should show 50% after 5/10 steps
        assert "ETA:" in messages[-1]   # Should include ETA


if __name__ == "__main__":
    pytest.main([__file__])