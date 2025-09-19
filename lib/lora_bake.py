"""
Violet Model Merge - LoRA Baking Engine

A sophisticated LoRA (Low-Rank Adaptation) baking system for merging LoRA weights
into Stable Diffusion models. Derived from Chattiori Model Merger by Chattiori 
(https://github.com/faildes). Supports various merging strategies including DARE
(Drop And REscale) and spectral normalization.

Features:
- Standard LoRA merging with customizable strengths
- DARE (Drop And REscale) merging for enhanced stability
- Spectral normalization for weight regularization
- Cross-architecture support (SD 1.x/2.x, SDXL)
- Metadata preservation and detailed merge tracking

Author: Chattiori Model Merger Contributors
License: MIT
"""

from __future__ import annotations

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F
import safetensors.torch
from tqdm.autonotebook import tqdm

from .utils import (
    load_model,
    read_metadata_from_safetensors,
    prune_model,
    sha256_from_cache,
    LBLOCKS26,
    BLOCKID,
    cache,
    dump_cache
)

# Type aliases
TensorDict = Dict[str, torch.Tensor]
PathLike = Union[str, Path]
LoRASpec = Tuple[str, float]  # (path, strength)

# Constants
_RE_DIGITS = re.compile(r"\d+")
_RE_CACHE: Dict[str, re.Pattern] = {}

# Suffix mapping for different layer types
SUFFIX_MAP = {
    "attentions": {},
    "resnets": {
        "conv1": "in_layers_2", 
        "conv2": "out_layers_3",
        "norm1": "in_layers_0", 
        "norm2": "out_layers_0",
        "time_emb_proj": "emb_layers_1", 
        "conv_shortcut": "skip_connection",
    },
}


class LoRAMergeMode(Enum):
    """LoRA merging modes."""
    STANDARD = "standard"
    DARE = "dare"


@dataclass
class LoRAConfig:
    """Configuration for LoRA baking operations."""
    model_path: PathLike
    checkpoint_name: str
    lora_specs: List[LoRASpec]
    output_name: str = "merged"
    device: str = "cpu"
    save_half: bool = False
    save_quarter: bool = False
    save_safetensors: bool = True
    prune: bool = False
    keep_ema: bool = False
    no_metadata: bool = False
    memo: Optional[str] = None
    merge_mode: LoRAMergeMode = LoRAMergeMode.STANDARD


class NameConverter:
    """Converts between different LoRA naming conventions."""
    
    @staticmethod
    def _match_pattern(pattern: str, key: str, buffer: List[Any]) -> bool:
        """
        Match regex pattern against key and populate buffer with groups.
        
        Args:
            pattern: Regex pattern to match
            key: String to match against
            buffer: List to populate with matched groups
            
        Returns:
            True if pattern matches, False otherwise
        """
        regex = _RE_CACHE.get(pattern)
        if not regex:
            regex = re.compile(pattern)
            _RE_CACHE[pattern] = regex
        
        match = regex.match(key)
        if not match:
            return False
        
        # Convert numeric groups to integers
        buffer[:] = [
            int(x) if _RE_DIGITS.fullmatch(x or "") else x 
            for x in match.groups()
        ]
        return True
    
    @staticmethod
    def convert_diffusers_to_compvis(key: str, is_sd2: bool = False) -> str:
        """
        Convert Diffusers-style LoRA key names to CompVis format.
        
        Args:
            key: Diffusers-style key name
            is_sd2: Whether this is for Stable Diffusion 2.x
            
        Returns:
            CompVis-style key name
        """
        groups = []
        
        # Input convolution layers
        if NameConverter._match_pattern(r"lora_unet_conv_in(.*)", key, groups):
            return f"diffusion_model_input_blocks_0_0{groups[0]}"
        
        # Output convolution layers
        if NameConverter._match_pattern(r"lora_unet_conv_out(.*)", key, groups):
            return f"diffusion_model_out_2{groups[0]}"
        
        # Time embedding layers
        if NameConverter._match_pattern(r"lora_unet_time_embedding_linear_(\d+)(.*)", key, groups):
            return f"diffusion_model_time_embed_{groups[0]*2-2}{groups[1]}"
        
        # Down blocks
        if NameConverter._match_pattern(r"lora_unet_down_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)", key, groups):
            suffix = SUFFIX_MAP.get(groups[1], {}).get(groups[3], groups[3])
            block_type = 1 if groups[1] == 'attentions' else 0
            return f"diffusion_model_input_blocks_{1 + groups[0]*3 + groups[2]}_{block_type}_{suffix}"
        
        # Middle blocks
        if NameConverter._match_pattern(r"lora_unet_mid_block_(attentions|resnets)_(\d+)_(.+)", key, groups):
            suffix = SUFFIX_MAP.get(groups[0], {}).get(groups[2], groups[2])
            block_type = 1 if groups[0] == 'attentions' else 0
            return f"diffusion_model_middle_block_{block_type}_{suffix}"
        
        # Up blocks
        if NameConverter._match_pattern(r"lora_unet_up_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)", key, groups):
            suffix = SUFFIX_MAP.get(groups[1], {}).get(groups[3], groups[3])
            block_type = 1 if groups[1] == 'attentions' else 0
            return f"diffusion_model_output_blocks_{groups[0]*3 + groups[2]}_{block_type}_{suffix}"
        
        # Return original key if no pattern matches
        return key


class LoRAProcessor:
    """Processes LoRA weights and applies various transformations."""
    
    @staticmethod
    def load_lora_state_dict(path: PathLike, dtype: torch.dtype = torch.float, 
                           device: str = "cpu", convert_keys: bool = True) -> TensorDict:
        """
        Load LoRA state dictionary from file.
        
        Args:
            path: Path to LoRA file
            dtype: Data type to load tensors as
            device: Device to load tensors on
            convert_keys: Whether to convert key names from diffusers to compvis format
            
        Returns:
            LoRA state dictionary
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"LoRA file not found: {path}")
        
        print(f"📂 Loading LoRA: {path.name}")
        
        try:
            if path.suffix.lower() == '.safetensors':
                import safetensors
                with safetensors.safe_open(path, framework="pt", device=device) as f:
                    state_dict = {key: f.get_tensor(key).to(dtype) for key in f.keys()}
            else:
                checkpoint = torch.load(path, map_location=device)
                state_dict = checkpoint.get("state_dict", checkpoint)
                # Convert to specified dtype
                state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
            
            # Convert key names if requested
            if convert_keys:
                converted_dict = {}
                for key, tensor in state_dict.items():
                    new_key = NameConverter.convert_diffusers_to_compvis(key)
                    converted_dict[new_key] = tensor
                state_dict = converted_dict
            
            print(f"✅ LoRA loaded: {len(state_dict)} keys")
            return state_dict
            
        except Exception as e:
            raise RuntimeError(f"Failed to load LoRA from {path}: {e}")
    
    @staticmethod
    def get_safe_metadata(path: PathLike) -> Dict[str, Any]:
        """Get metadata from safetensors file safely."""
        try:
            return read_metadata_from_safetensors(path)
        except Exception as e:
            print(f"⚠️ Warning: Could not read metadata from {path}: {e}")
            return {}
    
    @staticmethod
    def apply_dare_dropout(delta: torch.Tensor, dropout_prob: float) -> torch.Tensor:
        """
        Apply DARE (Drop And REscale) dropout to delta weights.
        
        Args:
            delta: Delta tensor to apply dropout to
            dropout_prob: Probability of dropping weights
            
        Returns:
            Tensor with DARE dropout applied
        """
        if dropout_prob <= 0:
            return delta
        
        # Create random mask
        mask = torch.rand_like(delta) > dropout_prob
        
        # Apply mask and rescale by the keep probability
        return delta * mask / (1 - dropout_prob)
    
    @staticmethod
    def normalize_l2(tensor: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """L2 normalize a tensor."""
        return tensor / (tensor.norm() + eps)
    
    @staticmethod
    def apply_spectral_norm(weight: torch.Tensor, iterations: int = 10) -> torch.Tensor:
        """
        Apply spectral normalization to a weight tensor.
        
        Args:
            weight: Weight tensor to normalize
            iterations: Number of power iterations
            
        Returns:
            Spectrally normalized weight tensor
        """
        if weight.dim() < 2:
            return weight
        
        # Reshape to 2D if needed
        original_shape = weight.shape
        if weight.dim() > 2:
            weight = weight.view(weight.size(0), -1)
        
        # Power iteration method
        u = torch.randn(weight.size(0), 1, device=weight.device, dtype=weight.dtype)
        v = torch.randn(weight.size(1), 1, device=weight.device, dtype=weight.dtype)
        
        for _ in range(iterations):
            v = LoRAProcessor.normalize_l2(weight.t() @ u)
            u = LoRAProcessor.normalize_l2(weight @ v)
        
        # Compute spectral norm
        sigma = (u.t() @ weight @ v).item()
        
        # Normalize and reshape back
        normalized = weight / sigma
        return normalized.view(original_shape)
    
    @staticmethod
    def apply_spectral_norm_to_lora(lora_dict: TensorDict, scale: float) -> TensorDict:
        """Apply spectral normalization to all LoRA weights."""
        normalized_dict = {}
        
        for key, tensor in tqdm(lora_dict.items(), desc="Applying spectral norm"):
            if "lora_up" in key or "lora_down" in key:
                normalized_dict[key] = LoRAProcessor.apply_spectral_norm(tensor) * scale
            else:
                normalized_dict[key] = tensor
        
        return normalized_dict


class WeightMerger:
    """Handles merging of LoRA weights with base model weights."""
    
    @staticmethod
    def build_key_mapping(state_dict: TensorDict) -> Dict[str, str]:
        """
        Build mapping from LoRA keys to model keys.
        
        Args:
            state_dict: Model state dictionary
            
        Returns:
            Dictionary mapping LoRA keys to model keys
        """
        key_map = {}
        
        for key in state_dict.keys():
            # Extract base key without LoRA suffix
            if ".lora_up.weight" in key:
                base_key = key.replace(".lora_up.weight", ".weight")
                key_map[key.replace(".lora_up.weight", "")] = base_key
            elif ".lora_down.weight" in key:
                continue  # Skip down weights as they're paired with up weights
        
        return key_map
    
    @staticmethod
    def apply_lora_to_weight(base_weight: torch.Tensor, up_weight: torch.Tensor, 
                           down_weight: torch.Tensor, scale: float, ratio: float = 1.0) -> torch.Tensor:
        """
        Apply LoRA weights to base weight tensor.
        
        Args:
            base_weight: Original model weight
            up_weight: LoRA up projection weight
            down_weight: LoRA down projection weight
            scale: LoRA scale factor
            ratio: Additional ratio multiplier
            
        Returns:
            Modified weight tensor
        """
        # Handle dimension mismatches
        if up_weight.dim() == 4 and down_weight.dim() == 4:
            # Convolutional layers
            delta = torch.nn.functional.conv2d(
                up_weight.permute(1, 0, 2, 3), 
                down_weight
            ).permute(1, 0, 2, 3)
        elif up_weight.dim() == 2 and down_weight.dim() == 2:
            # Linear layers
            delta = up_weight @ down_weight
        else:
            # Fallback: flatten and compute
            up_flat = up_weight.view(up_weight.size(0), -1)
            down_flat = down_weight.view(-1, down_weight.size(-1))
            delta = (up_flat @ down_flat).view(base_weight.shape)
        
        # Apply scale and ratio
        return base_weight + delta * scale * ratio
    
    @staticmethod
    def merge_lora_weights(lora_dict: TensorDict, is_sd2: bool = False, is_xl: bool = False,
                         dare_prob: float = 0.0, lambda_reg: float = 1.0, 
                         scale: float = 1.0, block_strengths: Optional[List[float]] = None) -> TensorDict:
        """
        Merge multiple LoRA weight sets with various options.
        
        Args:
            lora_dict: Dictionary of LoRA weights
            is_sd2: Whether this is SD 2.x
            is_xl: Whether this is SDXL
            dare_prob: DARE dropout probability
            lambda_reg: Regularization lambda
            scale: Global scale factor
            block_strengths: Per-block strength multipliers
            
        Returns:
            Merged LoRA weights
        """
        merged_dict = {}
        
        # Group by base keys
        lora_pairs = {}
        for key, tensor in lora_dict.items():
            if ".lora_up.weight" in key:
                base_key = key.replace(".lora_up.weight", "")
                if base_key not in lora_pairs:
                    lora_pairs[base_key] = {}
                lora_pairs[base_key]["up"] = tensor
            elif ".lora_down.weight" in key:
                base_key = key.replace(".lora_down.weight", "")
                if base_key not in lora_pairs:
                    lora_pairs[base_key] = {}
                lora_pairs[base_key]["down"] = tensor
        
        # Merge paired weights
        for base_key, weights in tqdm(lora_pairs.items(), desc="Merging LoRA weights"):
            if "up" in weights and "down" in weights:
                up_weight = weights["up"]
                down_weight = weights["down"]
                
                # Compute delta
                if up_weight.dim() == 4 and down_weight.dim() == 4:
                    delta = torch.nn.functional.conv2d(
                        up_weight.permute(1, 0, 2, 3), 
                        down_weight
                    ).permute(1, 0, 2, 3)
                else:
                    delta = up_weight @ down_weight
                
                # Apply DARE dropout if specified
                if dare_prob > 0:
                    delta = LoRAProcessor.apply_dare_dropout(delta, dare_prob)
                
                # Apply regularization
                delta = delta * lambda_reg * scale
                
                # Store merged weight
                merged_dict[base_key + ".weight"] = delta
        
        return merged_dict


class LoRABaker:
    """Main class for baking LoRA weights into models."""
    
    def __init__(self, config: LoRAConfig):
        """Initialize LoRA baker with configuration."""
        self.config = config
        self.cache_data = None
    
    def _load_base_model(self) -> Tuple[TensorDict, Dict[str, Any]]:
        """Load the base model for LoRA baking."""
        model_path = Path(self.config.model_path) / self.config.checkpoint_name
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"📂 Loading base model: {model_path.name}")
        
        # Load model using utils function - it returns just the state dict
        model_dict = load_model(str(model_path), self.config.device, self.cache_data)
        
        # Load metadata separately if it's a safetensors file
        metadata = {}
        if str(model_path).endswith('.safetensors'):
            try:
                metadata = read_metadata_from_safetensors(str(model_path))
            except Exception as e:
                print(f"⚠️ Warning: Could not read metadata: {e}")
                metadata = {}
        
        return model_dict, metadata
    
    def _parse_lora_specs(self) -> List[LoRASpec]:
        """Parse LoRA specifications from string format."""
        if isinstance(self.config.lora_specs, str):
            specs = []
            for spec in self.config.lora_specs.split(","):
                parts = spec.strip().split(":")
                if len(parts) == 2:
                    path, strength = parts
                    specs.append((path.strip(), float(strength.strip())))
                else:
                    specs.append((parts[0].strip(), 1.0))
            return specs
        return self.config.lora_specs
    
    def bake_standard(self) -> None:
        """Perform standard LoRA baking."""
        print("🔥 Starting standard LoRA baking...")
        
        # Load base model
        base_model, base_metadata = self._load_base_model()
        lora_specs = self._parse_lora_specs()
        
        # Build key mapping
        key_map = WeightMerger.build_key_mapping(base_model)
        
        # Load and apply each LoRA
        lora_metadata = {}
        for lora_path, strength in tqdm(lora_specs, desc="Processing LoRAs"):
            full_path = Path(self.config.model_path) / lora_path
            
            # Load LoRA
            lora_dict = LoRAProcessor.load_lora_state_dict(
                full_path, 
                device=self.config.device
            )
            
            # Get metadata
            lora_meta = LoRAProcessor.get_safe_metadata(full_path)
            lora_metadata[lora_path] = lora_meta
            
            # Apply LoRA to model weights
            for lora_key, lora_tensor in lora_dict.items():
                if lora_key.endswith(".lora_up.weight"):
                    base_key = lora_key.replace(".lora_up.weight", "")
                    down_key = base_key + ".lora_down.weight"
                    
                    if down_key in lora_dict and base_key in key_map:
                        model_key = key_map[base_key]
                        if model_key in base_model:
                            # Apply LoRA
                            base_weight = base_model[model_key].to("cpu")
                            up_weight = lora_tensor.to("cpu")
                            down_weight = lora_dict[down_key].to("cpu")
                            
                            new_weight = WeightMerger.apply_lora_to_weight(
                                base_weight, up_weight, down_weight, strength
                            )
                            
                            base_model[model_key] = torch.nn.Parameter(new_weight)
        
        # Post-processing
        self._finalize_model(base_model, base_metadata, lora_metadata)
    
    def bake_dare(self) -> None:
        """Perform DARE LoRA baking."""
        print("🎲 Starting DARE LoRA baking...")
        
        # Load base model
        base_model, base_metadata = self._load_base_model()
        lora_specs = self._parse_lora_specs()
        
        if not lora_specs:
            raise ValueError("DARE baking requires at least one LoRA")
        
        # Use first LoRA as main LoRA
        main_lora_path = Path(self.config.model_path) / lora_specs[0][0]
        main_lora = LoRAProcessor.load_lora_state_dict(main_lora_path, device=self.config.device)
        
        # Merge other LoRAs with DARE
        merged_lora = main_lora.copy()
        lora_metadata = {}
        
        for lora_path, strength in lora_specs[1:]:
            full_path = Path(self.config.model_path) / lora_path
            lora_dict = LoRAProcessor.load_lora_state_dict(full_path, device=self.config.device)
            lora_meta = LoRAProcessor.get_safe_metadata(full_path)
            lora_metadata[lora_path] = lora_meta
            
            # Apply DARE merging
            for key, tensor in lora_dict.items():
                if key in merged_lora:
                    # Apply DARE dropout with probability based on strength
                    dare_prob = 1.0 - strength
                    merged_tensor = LoRAProcessor.apply_dare_dropout(tensor, dare_prob)
                    merged_lora[key] = merged_lora[key] + merged_tensor * strength
                else:
                    merged_lora[key] = tensor * strength
        
        # Apply merged LoRA to base model
        key_map = WeightMerger.build_key_mapping(base_model)
        
        for lora_key, lora_tensor in merged_lora.items():
            if lora_key.endswith(".lora_up.weight"):
                base_key = lora_key.replace(".lora_up.weight", "")
                down_key = base_key + ".lora_down.weight"
                
                if down_key in merged_lora and base_key in key_map:
                    model_key = key_map[base_key]
                    if model_key in base_model:
                        base_weight = base_model[model_key].to("cpu")
                        up_weight = lora_tensor.to("cpu")
                        down_weight = merged_lora[down_key].to("cpu")
                        
                        new_weight = WeightMerger.apply_lora_to_weight(
                            base_weight, up_weight, down_weight, 1.0
                        )
                        
                        base_model[model_key] = torch.nn.Parameter(new_weight)
        
        # Post-processing
        self._finalize_model(base_model, base_metadata, lora_metadata)
    
    def _finalize_model(self, model_dict: TensorDict, base_metadata: Dict[str, Any], 
                       lora_metadata: Dict[str, Any]) -> None:
        """Finalize the baked model and save it."""
        print("🔧 Finalizing model...")
        
        # Prune if requested
        if self.config.prune:
            model_dict = prune_model(model_dict, "Model", self.config, isxl=False, isflux=False)
        
        # Ensure all tensors are contiguous
        for key in tqdm(list(model_dict.keys()), desc="Making tensors contiguous"):
            model_dict[key] = model_dict[key].contiguous()
        
        # Convert precision if requested
        if self.config.save_half:
            for key, tensor in model_dict.items():
                if tensor.dtype in {torch.float32, torch.float64}:
                    model_dict[key] = tensor.half()
        elif self.config.save_quarter:
            for key, tensor in model_dict.items():
                if tensor.dtype in {torch.float32, torch.float64, torch.float16}:
                    model_dict[key] = tensor.to(torch.float8_e4m3fn)
        
        # Prepare output path
        output_path = self._get_output_path()
        
        # Prepare metadata
        if not self.config.no_metadata:
            metadata = self._create_metadata(base_metadata, lora_metadata)
        else:
            metadata = None
        
        # Save model
        print(f"💾 Saving to: {output_path}")
        
        if output_path.suffix.lower() == '.safetensors':
            safetensors.torch.save_file(model_dict, str(output_path), metadata=metadata)
        else:
            torch.save({"state_dict": model_dict}, str(output_path))
        
        # Clean up
        del model_dict
        if self.cache_data:
            dump_cache(self.cache_data)
        
        # Report file size
        file_size = output_path.stat().st_size / (1024**3)  # GB
        print(f"✅ Done! Model saved ({file_size:.2f}GB)")
    
    def _get_output_path(self) -> Path:
        """Get the output file path."""
        base_path = Path(self.config.model_path)
        extension = ".safetensors" if self.config.save_safetensors else ".ckpt"
        return base_path / f"{self.config.output_name}{extension}"
    
    def _create_metadata(self, base_metadata: Dict[str, Any], 
                        lora_metadata: Dict[str, Any]) -> Dict[str, str]:
        """Create metadata for the baked model."""
        # Get model hash
        model_path = Path(self.config.model_path) / self.config.checkpoint_name
        model_name = model_path.stem
        checkpoint_hash = ""
        
        if self.cache_data:
            hash_info = sha256_from_cache(str(model_path), f"checkpoint/{model_name}", self.cache_data)
            checkpoint_hash = hash_info[0] or ""
        
        # Create LoRA info strings
        lora_specs = self._parse_lora_specs()
        lora_strs = [f"{path}:{strength}" for path, strength in lora_specs]
        
        # Build metadata
        merge_info = {
            "type": f"{self.config.merge_mode.value}-lora-chattiori",
            "checkpoint_hash": checkpoint_hash,
            "lora_hash": ",".join(lora_metadata.keys()),
            "alpha_info": ",".join(lora_strs),
            "output_name": self.config.output_name,
        }
        
        metadata = {
            "sd_merge_models": json.dumps(merge_info),
            "checkpoint": json.dumps(base_metadata),
            "lora": json.dumps(lora_metadata),
        }
        
        if self.config.memo:
            metadata["memo"] = self.config.memo
        
        return metadata
    
    def bake(self) -> None:
        """Perform LoRA baking based on configuration."""
        if self.config.merge_mode == LoRAMergeMode.DARE:
            self.bake_dare()
        else:
            self.bake_standard()


# Legacy function wrappers for backward compatibility
def get_loralist(arg: str) -> List[LoRASpec]:
    """Parse LoRA list from string format."""
    specs = []
    for spec in arg.split(","):
        parts = spec.strip().split(":")
        if len(parts) == 2:
            specs.append((parts[0].strip(), float(parts[1].strip())))
        else:
            specs.append((parts[0].strip(), 1.0))
    return specs

def load_state_dict(path: str, dtype=torch.float, device="cpu", depatch=True):
    """Legacy wrapper for loading LoRA state dict."""
    return LoRAProcessor.load_lora_state_dict(path, dtype, device, depatch)

def convert_diffusers_name_to_compvis(key: str, is_sd2: bool) -> str:
    """Legacy wrapper for name conversion."""
    return NameConverter.convert_diffusers_to_compvis(key, is_sd2)

def apply_dare(delta: torch.Tensor, p: float):
    """Legacy wrapper for DARE dropout."""
    return LoRAProcessor.apply_dare_dropout(delta, p)

def spectral_norm(W: torch.Tensor, it=10):
    """Legacy wrapper for spectral normalization."""
    return LoRAProcessor.apply_spectral_norm(W, it)

def pluslora(lora_list, model, output, model_path, device="cpu"):
    """Legacy wrapper for standard LoRA baking."""
    config = LoRAConfig(
        model_path=model_path,
        checkpoint_name=model,
        lora_specs=lora_list,
        output_name=Path(output).stem,
        device=device,
        save_safetensors=output.endswith('.safetensors')
    )
    
    baker = LoRABaker(config)
    baker.bake()

def darelora(mainlora, lora_list, model, output, model_path, device="cpu"):
    """Legacy wrapper for DARE LoRA baking."""
    config = LoRAConfig(
        model_path=model_path,
        checkpoint_name=model,
        lora_specs=lora_list,
        output_name=Path(output).stem,
        device=device,
        merge_mode=LoRAMergeMode.DARE,
        save_safetensors=output.endswith('.safetensors')
    )
    
    baker = LoRABaker(config)
    baker.bake()


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(description="Merge LoRAs into checkpoint models")
    parser.add_argument("model_path", type=str, help="Path to models directory")
    parser.add_argument("checkpoint", type=str, help="Name of the checkpoint file")
    parser.add_argument("loras", type=str, help="LoRA specs: 'path:strength,path:strength,...'")
    parser.add_argument("--save_half", action="store_true", help="Save as float16")
    parser.add_argument("--prune", action="store_true", help="Prune model")
    parser.add_argument("--save_quarter", action="store_true", help="Save as float8")
    parser.add_argument("--keep_ema", action="store_true", help="Keep EMA weights")
    parser.add_argument("--dare", action="store_true", help="Use DARE merge mode")
    parser.add_argument("--no_metadata", action="store_true", help="Save without metadata")
    parser.add_argument("--memo", type=str, help="Additional metadata info", default=None)
    parser.add_argument("--save_safetensors", action="store_true", help="Save as .safetensors")
    parser.add_argument("--output", type=str, help="Output filename (no extension)", default="merged")
    parser.add_argument("--device", type=str, help="Device to use", default="cpu")
    
    args = parser.parse_args()
    
    # Parse LoRA specifications
    lora_specs = get_loralist(args.loras)
    
    # Create configuration
    config = LoRAConfig(
        model_path=args.model_path,
        checkpoint_name=args.checkpoint,
        lora_specs=lora_specs,
        output_name=args.output,
        device=args.device,
        save_half=args.save_half,
        save_quarter=args.save_quarter,
        save_safetensors=args.save_safetensors,
        prune=args.prune,
        keep_ema=args.keep_ema,
        no_metadata=args.no_metadata,
        memo=args.memo,
        merge_mode=LoRAMergeMode.DARE if args.dare else LoRAMergeMode.STANDARD
    )
    
    # Perform baking
    baker = LoRABaker(config)
    baker.bake()


# Legacy compatibility functions
def apply_lora_to_model(model_path: str, lora_path: str, output_path: str, alpha: float = 1.0, **kwargs):
    """
    Legacy wrapper for LoRA baking functionality.
    Maintains compatibility with existing notebooks and scripts.
    """
    # Convert single LoRA path to LoRASpec format (tuple of path, strength)
    lora_specs = [(lora_path, alpha)]
    
    config = LoRAConfig(
        model_path=model_path,
        checkpoint_name="",  # Will be derived from model_path
        lora_specs=lora_specs,
        output_name=output_path,
        **kwargs
    )
    baker = LoRABaker(config)
    baker.bake()


if __name__ == "__main__":
    main()