"""
Simple smoke test to verify test setup is working correctly.
"""
import torch
import numpy as np


def test_basic_imports():
    """Test that basic imports work."""
    assert torch is not None
    assert np is not None


def test_torch_functionality():
    """Test basic PyTorch functionality."""
    # Create a simple tensor
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    
    # Basic operations
    z = x + y
    expected = torch.tensor([5.0, 7.0, 9.0])
    
    torch.testing.assert_close(z, expected)


def test_numpy_functionality():
    """Test basic NumPy functionality."""
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    
    c = a + b
    expected = np.array([5, 7, 9])
    
    np.testing.assert_array_equal(c, expected)


def test_project_structure():
    """Test that project structure is as expected."""
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    
    # Check that important files exist
    assert (project_root / "lib").exists()
    assert (project_root / "tests").exists()
    assert (project_root / "README.md").exists()
    assert (project_root / "pyproject.toml").exists()
    assert (project_root / "violet_merge.ipynb").exists()


def test_constants():
    """Test basic constants and configurations."""
    # These should always be true
    assert 1 + 1 == 2
    assert "test" == "test"
    assert [1, 2, 3] == [1, 2, 3]


if __name__ == "__main__":
    # Run basic tests if script is executed directly
    print("🧪 Running smoke tests...")
    
    test_basic_imports()
    print("✅ Basic imports work")
    
    test_torch_functionality()
    print("✅ PyTorch functionality works")
    
    test_numpy_functionality()
    print("✅ NumPy functionality works")
    
    test_project_structure()
    print("✅ Project structure is correct")
    
    test_constants()
    print("✅ Basic constants work")
    
    print("🎉 All smoke tests passed!")