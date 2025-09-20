"""
Violet Model Merge - Core Utilities

A comprehensive utility module for model merging operations.
Derived from Chattiori Model Merger by Chattiori (https://github.com/faildes).

Includes:
- Model loading and metadata extraction
- Block-wise merging configurations
- Random parameter generation
- Hashing and caching utilities
- Model format conversions

Author: Chattiori Model Merger Contributors
License: MIT
"""

from __future__ import annotations

import os
import shutil
import json
import re
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import safetensors
import filelock
from tqdm.auto import tqdm
import concurrent.futures as cf

# Type aliases
TensorDict = Dict[str, torch.Tensor]
PathLike = Union[str, Path]

# Constants
FP_SET: Set[torch.dtype] = {torch.float32, torch.float16, torch.float64, torch.bfloat16}

# Model architecture block definitions
NUM_INPUT_BLOCKS = 12
NUM_MID_BLOCK = 1
NUM_OUTPUT_BLOCKS = 12
NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS

BLOCKID = ["BASE"] + [f"IN{i:02}" for i in range(12)] + ["M00"] + [f"OUT{i:02}" for i in range(12)]
BLOCKIDXLL = ["BASE"] + [f"IN{i:02}" for i in range(9)] + ["M00"] + [f"OUT{i:02}" for i in range(9)] + ["VAE"]
BLOCKIDXL = ["BASE"] + [f"IN{i}" for i in range(9)] + ["M"] + [f"OUT{i}" for i in range(9)] + ["VAE"]
BLOCKIDFLUX = ["CLIP", "T5", "IN"] + ["D{:002}".format(x) for x in range(19)] + ["S{:002}".format(x) for x in range(38)] + ["OUT"]

# Model structure parsing regexes
_RE_INPUT = re.compile(r'\.input_blocks\.(\d+)\.')
_RE_MIDDLE = re.compile(r'\.middle_block\.(\d+)\.')
_RE_OUTPUT = re.compile(r'\.output_blocks\.(\d+)\.')

# Fine-tuning configurations
FINETUNEX = ["IN", "OUT", "OUT2", "CONT", "BRI", "COL1", "COL2", "COL3"]
COLS = [[-1, 1/3, 2/3], [1, 1, 0], [0, -1, -1], [1, 0, 1]]
COLSXL = [[0, 0, 1], [1, 0, 0], [-1, -1, 0], [-1, 1, 0]]

# Model prefixes and paths
PREFIXFIX = ("double_blocks", "single_blocks", "time_in", "vector_in", "txt_in")
PREFIX_MODEL = "model.diffusion_model."
BNB_SUFFIX = ".quant_state.bitsandbytes__"
QTYPES = ["fp4", "nf4"]

# Critical fine-tuning keys
FINETUNES = [
    "model.diffusion_model.input_blocks.0.0.weight",
    "model.diffusion_model.input_blocks.0.0.bias",
    "model.diffusion_model.out.0.weight",
    "model.diffusion_model.out.0.bias",
    "model.diffusion_model.out.2.weight",
    "model.diffusion_model.out.2.bias",
]

# Layer block identifiers for 26-block systems
LBLOCKS26 = [
    "encoder",
    "diffusion_model_input_blocks_0_", "diffusion_model_input_blocks_1_", "diffusion_model_input_blocks_2_",
    "diffusion_model_input_blocks_3_", "diffusion_model_input_blocks_4_", "diffusion_model_input_blocks_5_",
    "diffusion_model_input_blocks_6_", "diffusion_model_input_blocks_7_", "diffusion_model_input_blocks_8_",
    "diffusion_model_input_blocks_9_", "diffusion_model_input_blocks_10_", "diffusion_model_input_blocks_11_",
    "diffusion_model_middle_block_",
    "diffusion_model_output_blocks_0_", "diffusion_model_output_blocks_1_", "diffusion_model_output_blocks_2_",
    "diffusion_model_output_blocks_3_", "diffusion_model_output_blocks_4_", "diffusion_model_output_blocks_5_",
    "diffusion_model_output_blocks_6_", "diffusion_model_output_blocks_7_", "diffusion_model_output_blocks_8_",
    "diffusion_model_output_blocks_9_", "diffusion_model_output_blocks_10_", "diffusion_model_output_blocks_11_",
    "embedders",
]

# Model compatibility fixes
CHECKPOINT_DICT_REPLACEMENTS = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}

CHECKPOINT_DICT_SKIP_ON_MERGE = ["cond_stage_model.transformer.text_model.embeddings.position_ids"]
VAE_IGNORE_KEYS = {"model_ema.decay", "model_ema.num_updates"}

# Regex patterns
_SPLIT_PATTERN = re.compile(r"[,\n]+")


class ModelArchitecture(Enum):
    """Model architecture types for different diffusion models."""
    SDXL = "sdxl"
    FLUX = "flux"
    SD_V1_V2 = "sd_v1_v2"
    UNKNOWN = "unknown"


@dataclass
class RandomConfig:
    """Configuration for random parameter generation."""
    min_value: float
    max_value: float
    seed: int
    preset_name: Optional[str] = None


class PresetManager:
    """Manages weight presets for model merging."""
    
    def __init__(self, preset_file: Optional[PathLike] = None, master_file: Optional[PathLike] = None):
        """
        Initialize preset manager with file paths.
        
        Args:
            preset_file: Path to the presets file (defaults to lib/presets/mbwpresets.txt)
            master_file: Path to the master presets file (defaults to lib/presets/mbwpresets_master.txt)
        """
        # Default to the new organized structure
        if preset_file is None:
            preset_file = Path(__file__).parent / "presets" / "mbwpresets.txt"
        if master_file is None:
            master_file = Path(__file__).parent / "presets" / "mbwpresets_master.txt"
            
        self.preset_file = Path(preset_file)
        self.master_file = Path(master_file)
        self._ensure_preset_file()
        self.presets = self._load_presets()
    
    def _ensure_preset_file(self) -> None:
        """Ensure preset file exists, copy from master if needed."""
        if not self.preset_file.exists() and self.master_file.exists():
            shutil.copyfile(self.master_file, self.preset_file)
    
    def _load_presets(self) -> Dict[str, str]:
        """Load presets from file."""
        if not self.preset_file.exists():
            return {}
        
        try:
            with open(self.preset_file, 'r', encoding='utf-8') as f:
                return self._parse_presets(f.read())
        except (IOError, UnicodeDecodeError) as e:
            print(f"⚠️ Warning: Could not load presets from {self.preset_file}: {e}")
            return {}
    
    def _parse_presets(self, presets_text: str) -> Dict[str, str]:
        """
        Parse presets text into a dictionary.
        
        Args:
            presets_text: Raw presets text content
            
        Returns:
            Dictionary mapping preset names to weight strings
        """
        presets = {}
        for line in presets_text.splitlines():
            parts = re.split(r'[:\t]', line, maxsplit=1)
            if len(parts) == 2:
                key, weights = parts
                # Only include if weights string has exactly 26 values
                if len(weights.split(",")) == 26:
                    presets[key.strip()] = weights.strip()
        return presets
    
    def get_preset(self, name: str) -> Optional[str]:
        """Get a preset by name."""
        return self.presets.get(name)
    
    def list_presets(self) -> List[str]:
        """List all available preset names."""
        return list(self.presets.keys())


class StringParser:
    """Utility class for parsing various string formats."""
    
    @staticmethod
    def split_values(text: str) -> List[str]:
        """Split comma/newline separated values."""
        return [t.strip() for t in _SPLIT_PATTERN.split(text) if t.strip()]
    
    @staticmethod
    def safe_cast(values: List[str], index: int, cast_type: type, default: Any) -> Any:
        """Safely cast a value from a list with fallback."""
        try:
            return cast_type(values[index])
        except (IndexError, ValueError):
            return default
    
    @staticmethod
    def parse_random_spec(spec: str, seed: int) -> str:
        """
        Parse random specification string into formatted output.
        
        Args:
            spec: Random specification like "0.2,0.8[preset]"
            seed: Random seed to use
            
        Returns:
            Formatted specification string
        """
        core, _, rest = spec.replace(" ", "").partition("[")
        preset_part = rest[:-1] if rest.endswith("]") else None
        
        tokens = StringParser.split_values(core)
        min_val = StringParser.safe_cast(tokens, 0, float, 0.0)
        max_val = StringParser.safe_cast(tokens, 1, float, 1.0)
        seed_val = StringParser.safe_cast(tokens, 2, int, seed)
        
        preset_str = f"[{preset_part}]" if preset_part else ""
        return f"({min_val},{max_val},{seed_val},{preset_str})"


class WeightProcessor:
    """Processes weight specifications and generates random values."""
    
    def __init__(self, preset_manager: PresetManager):
        """Initialize with a preset manager."""
        self.preset_manager = preset_manager
    
    def process_weight_spec(self, spec: Union[int, float, str, List[str]], default_preset: str) -> Tuple[Union[float, List[float]], List[str], bool]:
        """
        Process a weight specification into values and metadata.
        
        Args:
            spec: Weight specification (number, string, or list)
            default_preset: Default preset to use
            
        Returns:
            Tuple of (processed_values, remaining_items, uses_blocks)
        """
        if isinstance(spec, (int, float)):
            return float(spec), [default_preset], False
        
        # Convert to list format for processing
        items = spec if isinstance(spec, list) else [spec]
        values, remaining = self._expand_presets(items)
        
        if len(values) == 1:
            return values[0], remaining, True
        return values, remaining, True
    
    def _expand_presets(self, items: List[str]) -> Tuple[List[float], List[str]]:
        """
        Expand preset references in items list.
        
        Args:
            items: List of items that may contain preset references
            
        Returns:
            Tuple of (numerical_values, remaining_items)
        """
        values = []
        remaining = []
        stack = list(items)
        
        while stack:
            item = stack.pop()
            preset_value = self.preset_manager.get_preset(item)
            
            if preset_value:
                # Expand preset and add to stack
                for token in StringParser.split_values(preset_value):
                    if token in self.preset_manager.presets:
                        stack.append(token)
                    else:
                        try:
                            values.append(float(token))
                        except ValueError:
                            remaining.append(token)
            else:
                # Try to parse as number
                try:
                    values.append(float(item))
                except ValueError:
                    remaining.append(item)
        
        return values, remaining
    
    def generate_random_ratios(self, spec: str) -> Tuple[List[float], List[str]]:
        """
        Generate random ratios based on specification string.
        
        Args:
            spec: Random specification like "0.2,0.8,123[PRESET:name(0.5)]"
            
        Returns:
            Tuple of (ratio_list, deep_processing_results)
        """
        core, _, rest = spec.partition("[")
        deep_items = StringParser.split_values(rest[:-1]) if rest.endswith("]") else []
        
        tokens = StringParser.split_values(core.replace(" ", ""))
        min_val = StringParser.safe_cast(tokens, 0, float, 0.0)
        max_val = StringParser.safe_cast(tokens, 1, float, 1.0)
        seed = StringParser.safe_cast(tokens, 2, int, random.randint(1, 4294967295))
        
        # Generate base random ratios
        np.random.seed(seed)
        ratios = np.random.uniform(min_val, max_val, 26).tolist()
        
        # Process deep items
        deep_results = []
        for item in deep_items:
            if "PRESET" in item:
                try:
                    _, preset_spec = item.split(":", 1)
                    name, ratio_spec = preset_spec.split("(", 1)
                    ratio_val = float(ratio_spec.rstrip(")"))
                    deep_results.append(f"PRESET:{name}:{ratio_val}")
                except (ValueError, IndexError):
                    deep_results.append(item)
            elif ":" in item:
                parts = item.split(":", 2)
                if len(parts) == 3:
                    try:
                        rounded_val = round(float(parts[2]), 3)
                        deep_results.append(f"{parts[0]}:{parts[1]}:{rounded_val}")
                    except ValueError:
                        deep_results.append(item)
                else:
                    deep_results.append(item)
            else:
                deep_results.append(item)
        
        return ratios, deep_results


class HashingUtils:
    """Utilities for model hashing and verification."""
    
    @staticmethod
    def calculate_sha256(file_path: PathLike, chunk_size: int = 4 * 1024 * 1024, max_workers: Optional[int] = None) -> str:
        """
        Calculate SHA256 hash of a file with parallel processing.
        
        Args:
            file_path: Path to the file
            chunk_size: Size of chunks to read
            max_workers: Maximum number of worker threads
            
        Returns:
            SHA256 hash as hexadecimal string
        """
        if max_workers is None:
            max_workers = os.cpu_count() or 4
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        hash_sha256 = hashlib.sha256()
        
        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(chunk_size):
                    hash_sha256.update(chunk)
        except IOError as e:
            raise IOError(f"Error reading file {file_path}: {e}")
        
        return hash_sha256.hexdigest()
    
    @staticmethod
    def calculate_short_hash(file_path: PathLike) -> str:
        """Calculate a short hash for quick identification."""
        full_hash = HashingUtils.calculate_sha256(file_path)
        return full_hash[:10]
    
    @staticmethod
    def model_hash(file_path: PathLike) -> str:
        """Calculate model hash with error handling."""
        try:
            return HashingUtils.calculate_sha256(file_path)
        except (FileNotFoundError, IOError) as e:
            print(f"⚠️ Warning: Could not calculate hash for {file_path}: {e}")
            return ""


class CacheManager:
    """Manages caching for model operations."""
    
    def __init__(self, cache_file: PathLike = "cache.json"):
        """Initialize cache manager."""
        self.cache_file = Path(cache_file)
        self._cache_data = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache data from file."""
        if not self.cache_file.exists():
            return {}
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Warning: Could not load cache from {self.cache_file}: {e}")
            return {}
    
    def save_cache(self) -> None:
        """Save cache data to file."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache_data, f, indent=2)
        except IOError as e:
            print(f"⚠️ Warning: Could not save cache to {self.cache_file}: {e}")
    
    def get_cached_hash(self, file_path: PathLike, title: str) -> Optional[str]:
        """Get cached hash for a file."""
        file_key = str(Path(file_path).resolve())
        file_data = self._cache_data.get(file_key, {})
        
        if isinstance(file_data, dict) and 'hashes' in file_data:
            return file_data['hashes'].get(title)
        return None
    
    def cache_hash(self, file_path: PathLike, title: str, hash_value: str) -> None:
        """Cache a hash value for a file."""
        file_key = str(Path(file_path).resolve())
        if file_key not in self._cache_data:
            self._cache_data[file_key] = {'hashes': {}}
        elif 'hashes' not in self._cache_data[file_key]:
            self._cache_data[file_key]['hashes'] = {}
        
        self._cache_data[file_key]['hashes'][title] = hash_value
        self.save_cache()
    
    def get_subsection(self, subsection: str) -> Dict[str, Any]:
        """Get a subsection of cache data."""
        return self._cache_data.get(subsection, {})
    
    def update_subsection(self, subsection: str, data: Dict[str, Any]) -> None:
        """Update a subsection of cache data."""
        self._cache_data[subsection] = data
        self.save_cache()


class ModelLoader:
    """Handles loading and processing of model files."""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """Initialize model loader."""
        self.cache_manager = cache_manager or CacheManager()
    
    def load_model(self, file_path: PathLike, device: str = "cpu", verify_hash: bool = True) -> TensorDict:
        """
        Load a model from file with caching and verification.
        
        Args:
            file_path: Path to the model file
            device: Device to load the model on
            verify_hash: Whether to verify model hash
            
        Returns:
            Model state dictionary
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        print(f"📂 Loading model: {file_path.name}")
        
        try:
            # Load based on file extension
            if file_path.suffix.lower() == '.safetensors':
                import safetensors
                with safetensors.safe_open(file_path, framework="pt", device=device) as f:
                    state_dict = {key: f.get_tensor(key) for key in f.keys()}
            else:
                # PyTorch checkpoint
                checkpoint = torch.load(file_path, map_location=device)
                state_dict = self._extract_state_dict(checkpoint)
            
            # Hash verification
            if verify_hash:
                model_hash = HashingUtils.model_hash(file_path)
                if model_hash:
                    self.cache_manager.cache_hash(file_path, "model", model_hash)
            
            print(f"✅ Model loaded successfully: {len(state_dict)} keys")
            return state_dict
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {file_path}: {e}")
    
    def _extract_state_dict(self, checkpoint: Dict[str, Any]) -> TensorDict:
        """Extract state dictionary from checkpoint."""
        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        elif "model" in checkpoint:
            return checkpoint["model"]
        else:
            # Assume the checkpoint is already a state dict
            return checkpoint
    
    def read_metadata_from_safetensors(self, file_path: PathLike) -> Dict[str, Any]:
        """Read metadata from safetensors file."""
        file_path = Path(file_path)
        try:
            with safetensors.safe_open(file_path, framework="pt") as f:
                return f.metadata() or {}
        except Exception as e:
            print(f"⚠️ Warning: Could not read metadata from {file_path}: {e}")
            return {}


class ModelUtils:
    """Utility functions for model processing."""
    
    @staticmethod
    def detect_architecture(state_dict: TensorDict) -> ModelArchitecture:
        """
        Detect model architecture from state dictionary.
        
        Args:
            state_dict: Model state dictionary
            
        Returns:
            Detected architecture type
        """
        # Check for FLUX architecture
        flux_keys = ["double_blocks", "single_blocks", "time_in", "vector_in", "txt_in"]
        if any(any(key in tensor_key for tensor_key in state_dict.keys()) for key in flux_keys):
            return ModelArchitecture.FLUX
        
        # Check for SDXL vs SD v1/v2
        # SDXL typically has larger dimensions in certain layers
        for key, tensor in state_dict.items():
            if "cond_stage_model" in key and tensor.dim() >= 2:
                if tensor.shape[-1] == 2048:  # SDXL text encoder dimension
                    return ModelArchitecture.SDXL
                elif tensor.shape[-1] == 768:  # SD v1/v2 text encoder dimension
                    return ModelArchitecture.SD_V1_V2
        
        return ModelArchitecture.UNKNOWN
    
    @staticmethod
    def prune_model(state_dict: TensorDict, output_name: str, args: Any, is_xl: bool = False, is_flux: bool = False) -> TensorDict:
        """
        Prune model by removing VAE and other unnecessary components.
        
        Args:
            state_dict: Model state dictionary
            output_name: Output model name
            args: Configuration arguments
            is_xl: Whether this is an SDXL model
            is_flux: Whether this is a FLUX model
            
        Returns:
            Pruned state dictionary
        """
        pruned = {}
        
        for key, tensor in state_dict.items():
            # Skip VAE components if pruning is enabled
            if hasattr(args, 'prune') and args.prune:
                if any(vae_key in key for vae_key in ["first_stage_model", "vae"]):
                    continue
            
            # Skip ignored keys
            if key in VAE_IGNORE_KEYS:
                continue
            
            # Transform checkpoint keys for compatibility
            transformed_key = ModelUtils.transform_checkpoint_key(key)
            pruned[transformed_key] = tensor
        
        print(f"🧹 Pruned model: {len(state_dict)} → {len(pruned)} keys")
        return pruned
    
    @staticmethod
    def transform_checkpoint_key(key: str) -> str:
        """Transform checkpoint keys for compatibility."""
        for old_prefix, new_prefix in CHECKPOINT_DICT_REPLACEMENTS.items():
            if old_prefix in key:
                key = key.replace(old_prefix, new_prefix)
        return key
    
    @staticmethod
    def to_half_precision(tensor: torch.Tensor, enable: bool = True) -> torch.Tensor:
        """Convert tensor to half precision if enabled."""
        if enable and tensor.dtype in FP_SET:
            return tensor.half()
        return tensor
    
    @staticmethod
    def to_half_precision_dict(state_dict: TensorDict, enable: bool = True) -> TensorDict:
        """Convert entire state dict to half precision."""
        if not enable:
            return state_dict
        
        return {
            key: ModelUtils.to_half_precision(tensor, enable)
            for key, tensor in state_dict.items()
        }


# Global instances for backward compatibility
_preset_manager = PresetManager()
_weight_processor = WeightProcessor(_preset_manager)
_cache_manager = CacheManager()
_model_loader = ModelLoader(_cache_manager)

# Legacy function wrappers for backward compatibility
def wgt(x, dp):
    """Legacy wrapper for weight processing."""
    return _weight_processor.process_weight_spec(x, dp)

def deepblock(items: List[str]) -> Tuple[List[float], List[str]]:
    """Legacy wrapper for preset expansion."""
    return _weight_processor._expand_presets(items)

def rand_ratio(s: str):
    """Legacy wrapper for random ratio generation with extended return values."""
    ratios, deep = _weight_processor.generate_random_ratios(s)
    
    # Extract seed from the specification for backward compatibility
    core, _, _ = s.partition("[")
    tokens = StringParser.split_values(core.replace(" ", ""))
    seed = StringParser.safe_cast(tokens, 2, int, random.randint(1, 4294967295))
    
    # Format for legacy compatibility
    alpha_info = rinfo(s, seed)
    
    return ratios, seed, deep, alpha_info

def sha256(filename: str, title: str, cache_data=None) -> str:
    """Legacy wrapper for SHA256 calculation."""
    if cache_data is not None:
        cached = _cache_manager.get_cached_hash(filename, title)
        if cached:
            return cached
    
    hash_value = HashingUtils.calculate_sha256(filename)
    _cache_manager.cache_hash(filename, title, hash_value)
    return hash_value

def load_model(path: str, device: str = "cpu", cache_data=None, verify_hash: bool = True):
    """Legacy wrapper for model loading - returns (theta, sha256, hash, meta, cache_data)."""
    
    # Load the model state dict
    theta = _model_loader.load_model(path, device, verify_hash)
    
    # Calculate hashes
    sha256_hash = HashingUtils.calculate_sha256(path)
    short_hash = HashingUtils.calculate_short_hash(path)
    
    # Read metadata
    try:
        meta = read_metadata_from_safetensors(path)
    except Exception:
        meta = {}
    
    # Handle cache data
    if cache_data is None:
        cache_data = {}
    
    return theta, sha256_hash, short_hash, meta, cache_data

def read_metadata_from_safetensors(filename):
    """Legacy wrapper for metadata reading."""
    return _model_loader.read_metadata_from_safetensors(filename)

def detect_arch(theta):
    """Legacy wrapper for architecture detection - returns (isxl, isflux) tuple."""
    arch = ModelUtils.detect_architecture(theta)
    
    # Convert to legacy format
    isxl = arch == ModelArchitecture.SDXL
    isflux = arch == ModelArchitecture.FLUX
    
    return isxl, isflux

def prune_model(theta, name, args, isxl=False, isflux=False):
    """Legacy wrapper for model pruning."""
    return ModelUtils.prune_model(theta, name, args, isxl, isflux)

def to_half(tensor, enable):
    """Legacy wrapper for half precision conversion."""
    return ModelUtils.to_half_precision(tensor, enable)

def to_half_k(sd, enable):
    """Legacy wrapper for state dict half precision conversion."""
    return ModelUtils.to_half_precision_dict(sd, enable)

# Export everything that was in the original module
weights_presets_list = _preset_manager.presets

# Additional legacy functions that need to be preserved
def rinfo(s: str, seed: int) -> str:
    """Legacy wrapper for random specification parsing."""
    return StringParser.parse_random_spec(s, seed)

def roundeep(term):
    """Round deep processing terms to 3 decimal places."""
    if not term:
        return None
    
    out = []
    for d in term:
        try:
            a, b, c = d.split(":", 2)
            out.append(f"{a}:{b}:{round(float(c), 3)}")
        except ValueError:
            out.append(d)
    return out

def colorcalc(cols, isxl):
    """Calculate color adjustments based on model type."""
    return COLSXL if isxl else COLS

def fineman(fine, isxl=False, isflux=False):
    """Process fine-tuning specifications."""
    if not fine:
        return []
    
    colors = colorcalc(None, isxl)
    
    if fine in FINETUNEX:
        index = FINETUNEX.index(fine)
        if index < len(colors):
            return colors[index]
    
    return []

def weighttoxl(weight):
    """Convert weight format for XL models."""
    # Legacy function - implementation would depend on specific requirements
    return weight

def parse_ratio(ratios, info, dp):
    """Parse ratio specifications into usable format."""
    if isinstance(ratios, str):
        try:
            return eval(ratios)  # Note: This is legacy behavior - not recommended for new code
        except:
            return ratios
    return ratios

def calculate_shorthash(filename: str):
    """Legacy wrapper for short hash calculation."""
    return HashingUtils.calculate_short_hash(filename)

def model_hash(filename: str) -> str:
    """Legacy wrapper for model hash calculation."""
    return HashingUtils.model_hash(filename)

def sha256_from_cache(filename: str, title: str, cache_data):
    """Get SHA256 from cache with legacy format."""
    cached = _cache_manager.get_cached_hash(filename, title)
    return cached, bool(cached), cache_data

def merge_cache_json(model_path):
    """Merge cache JSON data - legacy function."""
    return _cache_manager.get_subsection("models")

def dump_cache(cache_data):
    """Dump cache data - legacy function."""
    _cache_manager.save_cache()

def cache(subsection, cache_data):
    """Cache subsection data - legacy function."""
    return _cache_manager.get_subsection(subsection)

def transform_checkpoint_dict_key(k: str):
    """Transform checkpoint dictionary key for compatibility."""
    return ModelUtils.transform_checkpoint_key(k)

def get_state_dict_from_checkpoint(pl_sd: dict) -> dict:
    """Extract state dict from PyTorch Lightning checkpoint."""
    if "state_dict" in pl_sd:
        # PyTorch Lightning format
        sd = pl_sd["state_dict"]
        # Remove 'model.' prefix if present
        if any(k.startswith("model.") for k in sd.keys()):
            sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
        return sd
    elif "model" in pl_sd:
        return pl_sd["model"]
    else:
        return pl_sd

def qdtyper(sd):
    """Detect quantization type from state dict."""
    if any("fp4" in k for k in sd):
        return "fp4"
    if any("nf4" in k for k in sd):
        return "nf4"
    
    for v in sd.values():
        dt = getattr(v, "dtype", None)
        if dt is not None:
            return dt
    return None

def to_qdtype(sd1, sd2, qd1, qd2, device):
    """Convert state dicts to quantized types."""
    t1 = t2 = torch.float16 if (qd1 in QTYPES and qd2 in QTYPES) else None
    if qd1 in QTYPES:
        sd1, _ = q_dequantize(sd1, qd1, device, t1)
    if qd2 in QTYPES:
        sd2, _ = q_dequantize(sd2, qd2, device, t2)
    return sd1, sd2

def maybe_to_qdtype(a, b, qa, qb, device, isflux):
    """Maybe convert to quantized type based on architecture."""
    return to_qdtype(a, b, qa, qb, device) if isflux and qa != qb else (a, b)

def q_dequantize(sd, qtype, device, dtype, setbnb=True):
    """Dequantize 4-bit tensors."""
    try:
        from bitsandbytes.functional import dequantize_4bit
    except ImportError:
        print("⚠️ Warning: bitsandbytes not available for quantization")
        return sd, dtype
    
    dels = []
    calc = "cuda:0" if torch.cuda.is_available() else ("mps:0" if torch.backends.mps.is_available() else "cpu")
    
    for k, v in list(sd.items()):
        qk = k + BNB_SUFFIX + qtype
        if ("weight" in k) and ("weight." not in k) and (qk in sd):
            qs = q_tensor_to_dict(sd[qk])
            out = torch.empty(qs["shape"], device=calc)
            deq = dequantize_4bit(
                v.to(calc), 
                out=out,
                absmax=sd[k + ".absmax"].to(calc),
                blocksize=qs["blocksize"], 
                quant_type=qs["quant_type"]
            )
            sd[k] = deq.to(device, dtype) if dtype else deq.to(device)
            dels += [k + ".absmax", k + ".quant_map"] + ([qk] if setbnb else [])
        elif isinstance(v, torch.Tensor) and dtype:
            sd[k] = v.to(dtype)
    
    for k in dels:
        sd.pop(k, None)
    return sd, dtype

def q_tensor_to_dict(t):
    """Convert quantized tensor to dictionary."""
    return json.loads(bytes(t.tolist()).decode("utf-8"))

def blocker(blocks: str, blockids: List[str]) -> str:
    """Expand block ranges into individual block names."""
    out = []
    for w in blocks.split():
        if "-" in w:
            a, b = (t.strip() for t in w.split("-", 1))
            try:
                i, j = blockids.index(a), blockids.index(b)
                lo, hi = (i, j) if i <= j else (j, i)
                out.extend(blockids[lo:hi + 1])
            except ValueError:
                out.append(w)  # If block not found, keep as-is
        else:
            out.append(w)
    return " ".join(out)

def blockfromkey(key: str, isxl: bool = False, isflux: bool = False):
    """Determine block identifier from tensor key."""
    # SD1.5/2.0
    if not isxl and not isflux:
        if "time_embed" in key:
            idx = -2
        elif ".out." in key:
            idx = NUM_TOTAL_BLOCKS - 1
        elif (m := _RE_INPUT.search(key)):
            idx = int(m.group(1))
        elif _RE_MIDDLE.search(key):
            idx = NUM_INPUT_BLOCKS
        elif (m := _RE_OUTPUT.search(key)):
            idx = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + int(m.group(1))
        else:
            return "Not Merge", "Not Merge"
        
        b = BLOCKID[idx + 1]
        return b, b

    # Flux
    if isflux:
        if "vae" in key:
            return "VAE", "Not Merge"
        if "t5xxl" in key:
            return "T5", "T5"
        if "text_encoders.clip" in key:
            return "CLIP", "CLIP"
        
        m = re.search(r'\.(\d+)\.', key)
        if "double_blocks" in key and m:
            return f"D{m.group(1).zfill(2)}", f"D{m.group(1).zfill(2)}"
        if "single_blocks" in key and m:
            return f"S{m.group(1).zfill(2)}", f"S{m.group(1).zfill(2)}"
        if "_in" in key:
            return "IN", "IN"
        if "final_layer" in key:
            return "OUT", "OUT"
        return "Not Merge", "Not Merge"

    # SDXL
    if not ("weight" in key or "bias" in key):
        return "Not Merge", "Not Merge"
    if "label_emb" in key or "time_embed" in key:
        return "Not Merge", "Not Merge"
    if "conditioner.embedders" in key:
        return "BASE", "BASE"
    if "first_stage_model" in key:
        return "VAE", "BASE"

    if "model.diffusion_model" in key:
        if "model.diffusion_model.out." in key:
            return "OUT8", "OUT08"
        
        blk = (re.findall(r'input|mid|output', key) or [""])[0].upper().replace("PUT", "")
        nums = re.sub(r"\D", "", key)
        tag = (nums[:1] + "0") if "MID" in blk else nums[:2]
        add = (re.findall(r"transformer_blocks\.(\d+)\.", key) or [""])[0]
        left = blk + tag + add
        right = ("M00" if "MID" in blk else f"{blk}0{tag[0]}")
        return left, right

    return "Not Merge", "Not Merge"

def elementals(key: str, weight_index: int, deep: List[str], current_alpha: float):
    """Apply elemental weight modifications based on deep processing rules."""
    skey = key + BLOCKID[weight_index + 1]

    def _neg(tokens: List[str]):
        return (True, tokens[1:]) if tokens and tokens[0] == "NOT" else (False, tokens)

    for d in deep:
        if d.count(":") != 2:
            continue
        
        dbs_s, dws_s, dr_s = d.split(":", 2)

        dbs = blocker(dbs_s, BLOCKID).split()
        dws = dws_s.split()
        dbn, dbs = _neg(dbs)
        dwn, dws = _neg(dws)

        ok = (any(db in skey for db in dbs) ^ dbn)
        if ok:
            ok = (any(dw in skey for dw in dws) ^ dwn)
        if ok:
            current_alpha = float(dr_s)

    return current_alpha

def diff_inplace(dst, src, func, desc):
    """Apply function to tensors in-place with progress tracking."""
    for k in tqdm(dst.keys(), desc=desc):
        if 'model' not in k:
            continue
        t2 = src.get(k, torch.zeros_like(dst[k])) if k in src else None
        dst[k] = func(dst[k], t2) if t2 is not None else torch.zeros_like(dst[k])

def clone_dict_tensors(d):
    """Clone all tensors in a dictionary."""
    return {k: v.clone() for k, v in tqdm(d.items(), "Cloning dict...")}

def np_trim_percentiles(arr, lo=1, hi=99):
    """Trim array to specified percentile range."""
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return arr
    
    lo_v = np.percentile(arr, lo, method='midpoint')
    hi_v = np.percentile(arr, hi, method='midpoint')
    return arr[(arr >= lo_v) & (arr <= hi_v)]

# Legacy support for direct access to regex patterns
_re_inp = _RE_INPUT
_re_mid = _RE_MIDDLE
_re_out = _RE_OUTPUT

# Export constants for backward compatibility
checkpoint_dict_replacements = CHECKPOINT_DICT_REPLACEMENTS
checkpoint_dict_skip_on_merge = CHECKPOINT_DICT_SKIP_ON_MERGE
vae_ignore_keys = VAE_IGNORE_KEYS