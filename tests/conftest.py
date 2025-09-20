"""
Test Configuration and Fixtures

Common test utilities, fixtures, and mock data for the test suite.
"""
import os
import tempfile
import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any
from unittest.mock import MagicMock, patch


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_safetensors_model():
    """Create a mock safetensors model structure."""
    return {
        "model.diffusion_model.input_blocks.0.0.weight": torch.randn(320, 4, 3, 3),
        "model.diffusion_model.input_blocks.0.0.bias": torch.randn(320),
        "model.diffusion_model.middle_block.0.weight": torch.randn(1280, 1280, 3, 3),
        "model.diffusion_model.output_blocks.0.0.weight": torch.randn(320, 640, 3, 3),
        "first_stage_model.encoder.conv_in.weight": torch.randn(128, 3, 3, 3),
        "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight": torch.randn(49408, 768),
    }


@pytest.fixture
def mock_model_files(temp_dir, mock_safetensors_model):
    """Create mock model files for testing."""
    import safetensors.torch
    
    # Create model directories
    models_dir = temp_dir / "models"
    vae_dir = temp_dir / "vae"
    models_dir.mkdir()
    vae_dir.mkdir()
    
    # Save mock models
    model_a_path = models_dir / "model_a.safetensors"
    model_b_path = models_dir / "model_b.safetensors"
    
    safetensors.torch.save_file(mock_safetensors_model, model_a_path)
    
    # Create slight variation for model B
    model_b = {k: v + torch.randn_like(v) * 0.1 for k, v in mock_safetensors_model.items()}
    safetensors.torch.save_file(model_b, model_b_path)
    
    # Create mock VAE
    vae_model = {"decoder.conv_out.weight": torch.randn(3, 512, 3, 3)}
    vae_path = vae_dir / "test_vae.safetensors"
    safetensors.torch.save_file(vae_model, vae_path)
    
    return {
        "models_dir": models_dir,
        "vae_dir": vae_dir,
        "model_a": model_a_path,
        "model_b": model_b_path,
        "vae": vae_path
    }


@pytest.fixture
def mock_device():
    """Mock device detection."""
    return "cpu"  # Always use CPU for tests to avoid CUDA dependencies


@pytest.fixture
def sample_merge_config():
    """Sample merge configuration for testing."""
    return {
        "mode": "WS",
        "alpha": 0.5,
        "device": "cpu",
        "save_safetensors": True,
        "save_half": False
    }


class MockTqdm:
    """Mock tqdm for testing without progress bars."""
    
    def __init__(self, iterable=None, total=None, desc=None, **kwargs):
        self.iterable = iterable if iterable is not None else range(total or 100)
        self.total = total or (len(iterable) if iterable else 100)
        self.desc = desc
        self.n = 0
    
    def __iter__(self):
        for item in self.iterable:
            self.n += 1
            yield item
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def update(self, n=1):
        self.n += n
    
    def set_description(self, desc):
        self.desc = desc
    
    def close(self):
        pass


@pytest.fixture(autouse=True)
def mock_tqdm():
    """Mock tqdm globally for cleaner test output."""
    with patch('tqdm.tqdm', MockTqdm), \
         patch('tqdm.auto.tqdm', MockTqdm):
        yield


@pytest.fixture
def mock_subprocess_run():
    """Mock subprocess.run for CLI testing."""
    with patch('subprocess.run') as mock:
        mock.return_value = MagicMock(returncode=0, stdout="", stderr="")
        yield mock


# Utility functions for tests
def create_minimal_model_dict() -> Dict[str, torch.Tensor]:
    """Create a minimal model dictionary for testing."""
    return {
        "model.diffusion_model.input_blocks.0.0.weight": torch.randn(8, 4, 3, 3),
        "model.diffusion_model.middle_block.0.weight": torch.randn(8, 8, 1, 1),
        "model.diffusion_model.output_blocks.0.0.weight": torch.randn(4, 8, 3, 3),
    }


def assert_tensors_close(tensor_dict_a: Dict[str, torch.Tensor], 
                        tensor_dict_b: Dict[str, torch.Tensor], 
                        rtol: float = 1e-5, 
                        atol: float = 1e-8) -> None:
    """Assert that two tensor dictionaries are approximately equal."""
    assert set(tensor_dict_a.keys()) == set(tensor_dict_b.keys()), "Tensor keys don't match"
    
    for key in tensor_dict_a.keys():
        torch.testing.assert_close(
            tensor_dict_a[key], 
            tensor_dict_b[key], 
            rtol=rtol, 
            atol=atol,
            msg=f"Tensors for key '{key}' are not close"
        )


def count_parameters(model_dict: Dict[str, torch.Tensor]) -> int:
    """Count total parameters in a model dictionary."""
    return sum(tensor.numel() for tensor in model_dict.values())