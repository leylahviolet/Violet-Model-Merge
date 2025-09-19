#!/usr/bin/env python3
"""
💜 Violet Model Merge - VAE Merging Module
✨ Beautiful, pythonic VAE merging for AI artists!

This module provides advanced VAE (Variational Autoencoder) merging capabilities,
allowing artists to blend different VAEs for enhanced image generation quality.

Author: Violet Tools (Original implementation by Faildes)
License: MIT
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
import safetensors.torch

# Import from the utils module
try:
    from .utils import ModelLoader, TensorDict, PathLike, ModelUtils
except ImportError:
    # For standalone execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils import ModelLoader, TensorDict, PathLike, ModelUtils


class VaeMergeMode(Enum):
    """Enumeration of available VAE merge modes with their configurations."""
    
    WS = ("Weighted Sum", False)
    LERP = ("Linear Interpolation", False)
    COMP = ("Component Specific", True)  # Separate encoder/decoder weights
    SLERP = ("Spherical Linear", False)
    
    def __init__(self, description: str, needs_component_weights: bool):
        self.description = description
        self.needs_component_weights = needs_component_weights


@dataclass
class VaeMergeConfig:
    """Configuration class for VAE merge operations."""
    
    # Core settings
    mode: VaeMergeMode
    vae_path: str
    vae_0: str
    vae_1: str
    vae_2: Optional[str] = None
    output: str = "merged_vae"
    device: str = "cpu"
    
    # Merge parameters
    alpha: float = 0.5
    encoder_alpha: Optional[float] = None  # For component-specific merging
    decoder_alpha: Optional[float] = None  # For component-specific merging
    
    # Output format
    save_half: bool = False
    save_safetensors: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate the merge configuration."""
        # Check if VAE path exists
        vae_path = Path(self.vae_path)
        if not vae_path.exists():
            raise FileNotFoundError(f"🚨 VAE directory not found: {self.vae_path}")
        
        # Check if VAE files exist
        for vae_name in [self.vae_0, self.vae_1]:
            if vae_name:
                vae_file = vae_path / vae_name
                if not vae_file.exists():
                    raise FileNotFoundError(f"🚨 VAE file not found: {vae_file}")
        
        # Check component-specific weights
        if self.mode == VaeMergeMode.COMP:
            if self.encoder_alpha is None or self.decoder_alpha is None:
                raise ValueError(
                    "🚨 Component-specific mode requires both encoder_alpha and decoder_alpha"
                )
        
        # Validate alpha values
        for alpha_name, alpha_value in [
            ("alpha", self.alpha),
            ("encoder_alpha", self.encoder_alpha),
            ("decoder_alpha", self.decoder_alpha)
        ]:
            if alpha_value is not None and not (0.0 <= alpha_value <= 1.0):
                raise ValueError(f"🚨 {alpha_name} must be between 0.0 and 1.0, got {alpha_value}")


class VaeMergeAlgorithms:
    """Collection of VAE merging algorithms."""
    
    @staticmethod
    def weighted_sum(
        vae_0: TensorDict, 
        vae_1: TensorDict, 
        alpha: float,
        save_half: bool = False
    ) -> TensorDict:
        """
        Weighted sum merge: result = (1-alpha) * vae_0 + alpha * vae_1
        
        Args:
            vae_0: First VAE state dict
            vae_1: Second VAE state dict  
            alpha: Mixing ratio (0.0 = full vae_0, 1.0 = full vae_1)
            save_half: Convert to half precision
            
        Returns:
            Merged VAE state dict
        """
        result = {}
        all_keys = set(vae_0.keys()) | set(vae_1.keys())
        
        for key in tqdm(all_keys, desc="🎨 Weighted Sum VAE merge"):
            if key in vae_0 and key in vae_1:
                # Both VAEs have this key - blend them
                tensor_0 = vae_0[key].float()
                tensor_1 = vae_1[key].float()
                
                if tensor_0.shape == tensor_1.shape:
                    merged = (1.0 - alpha) * tensor_0 + alpha * tensor_1
                    result[key] = ModelUtils.to_half_precision(merged, save_half)
                else:
                    print(f"⚠️ Shape mismatch for {key}, using vae_0")
                    result[key] = ModelUtils.to_half_precision(tensor_0, save_half)
            elif key in vae_0:
                # Only in vae_0
                result[key] = ModelUtils.to_half_precision(vae_0[key].float(), save_half)
            elif key in vae_1:
                # Only in vae_1
                result[key] = ModelUtils.to_half_precision(vae_1[key].float(), save_half)
        
        return result
    
    @staticmethod
    def linear_interpolation(
        vae_0: TensorDict,
        vae_1: TensorDict,
        alpha: float,
        save_half: bool = False
    ) -> TensorDict:
        """
        Linear interpolation merge with smoother blending.
        Similar to weighted sum but with additional smoothing.
        """
        # For now, LERP is identical to weighted sum
        # Future enhancement: Add smoothing functions
        return VaeMergeAlgorithms.weighted_sum(vae_0, vae_1, alpha, save_half)
    
    @staticmethod
    def component_specific(
        vae_0: TensorDict,
        vae_1: TensorDict,
        encoder_alpha: float,
        decoder_alpha: float,
        save_half: bool = False
    ) -> TensorDict:
        """
        Component-specific merge with different weights for encoder and decoder.
        
        Args:
            vae_0: First VAE state dict
            vae_1: Second VAE state dict
            encoder_alpha: Mixing ratio for encoder components
            decoder_alpha: Mixing ratio for decoder components
            save_half: Convert to half precision
            
        Returns:
            Merged VAE state dict
        """
        result = {}
        all_keys = set(vae_0.keys()) | set(vae_1.keys())
        
        def is_encoder_key(key: str) -> bool:
            """Determine if a key belongs to the encoder."""
            encoder_patterns = [
                'encoder', 'enc', 'down', 'input', 'conv_in',
                'in_layers', 'downsample'
            ]
            return any(pattern in key.lower() for pattern in encoder_patterns)
        
        def is_decoder_key(key: str) -> bool:
            """Determine if a key belongs to the decoder."""
            decoder_patterns = [
                'decoder', 'dec', 'up', 'output', 'conv_out',
                'out_layers', 'upsample'
            ]
            return any(pattern in key.lower() for pattern in decoder_patterns)
        
        for key in tqdm(all_keys, desc="🎭 Component-specific VAE merge"):
            if key in vae_0 and key in vae_1:
                tensor_0 = vae_0[key].float()
                tensor_1 = vae_1[key].float()
                
                if tensor_0.shape == tensor_1.shape:
                    # Choose alpha based on component type
                    if is_encoder_key(key):
                        alpha = encoder_alpha
                        component = "encoder"
                    elif is_decoder_key(key):
                        alpha = decoder_alpha
                        component = "decoder"
                    else:
                        # Unknown component, use average
                        alpha = (encoder_alpha + decoder_alpha) / 2
                        component = "unknown"
                    
                    merged = (1.0 - alpha) * tensor_0 + alpha * tensor_1
                    result[key] = ModelUtils.to_half_precision(merged, save_half)
                else:
                    print(f"⚠️ Shape mismatch for {key}, using vae_0")
                    result[key] = ModelUtils.to_half_precision(tensor_0, save_half)
            elif key in vae_0:
                result[key] = ModelUtils.to_half_precision(vae_0[key].float(), save_half)
            elif key in vae_1:
                result[key] = ModelUtils.to_half_precision(vae_1[key].float(), save_half)
        
        return result
    
    @staticmethod
    def spherical_linear(
        vae_0: TensorDict,
        vae_1: TensorDict,
        alpha: float,
        save_half: bool = False
    ) -> TensorDict:
        """
        Spherical linear interpolation (SLERP) for smoother blending.
        Particularly useful for high-dimensional tensor spaces.
        """
        result = {}
        all_keys = set(vae_0.keys()) | set(vae_1.keys())
        
        def slerp_tensors(t0: torch.Tensor, t1: torch.Tensor, alpha: float) -> torch.Tensor:
            """Perform SLERP between two tensors."""
            # Flatten tensors for SLERP calculation
            original_shape = t0.shape
            t0_flat = t0.flatten()
            t1_flat = t1.flatten()
            
            # Normalize tensors
            t0_norm = F.normalize(t0_flat, dim=0)
            t1_norm = F.normalize(t1_flat, dim=0)
            
            # Calculate angle between tensors
            dot_product = torch.clamp(torch.dot(t0_norm, t1_norm), -1.0, 1.0)
            omega = torch.acos(torch.abs(dot_product))
            
            # Handle parallel vectors
            if omega < 1e-6:
                return (1.0 - alpha) * t0 + alpha * t1
            
            # SLERP calculation
            sin_omega = torch.sin(omega)
            scale_0 = torch.sin((1.0 - alpha) * omega) / sin_omega
            scale_1 = torch.sin(alpha * omega) / sin_omega
            
            # Get magnitudes
            mag_0 = torch.norm(t0_flat)
            mag_1 = torch.norm(t1_flat)
            target_magnitude = (1.0 - alpha) * mag_0 + alpha * mag_1
            
            # Interpolate and scale
            result_flat = scale_0 * t0_norm + scale_1 * t1_norm
            result_flat = result_flat * target_magnitude
            
            return result_flat.reshape(original_shape)
        
        for key in tqdm(all_keys, desc="🔮 SLERP VAE merge"):
            if key in vae_0 and key in vae_1:
                tensor_0 = vae_0[key].float()
                tensor_1 = vae_1[key].float()
                
                if tensor_0.shape == tensor_1.shape:
                    try:
                        merged = slerp_tensors(tensor_0, tensor_1, alpha)
                        result[key] = ModelUtils.to_half_precision(merged, save_half)
                    except Exception as e:
                        print(f"⚠️ SLERP failed for {key}, falling back to linear: {e}")
                        merged = (1.0 - alpha) * tensor_0 + alpha * tensor_1
                        result[key] = ModelUtils.to_half_precision(merged, save_half)
                else:
                    print(f"⚠️ Shape mismatch for {key}, using vae_0")
                    result[key] = ModelUtils.to_half_precision(tensor_0, save_half)
            elif key in vae_0:
                result[key] = ModelUtils.to_half_precision(vae_0[key].float(), save_half)
            elif key in vae_1:
                result[key] = ModelUtils.to_half_precision(vae_1[key].float(), save_half)
        
        return result


class VaeMerger:
    """
    Advanced VAE merging system for Violet Model Merge.
    
    Provides multiple merging algorithms optimized for VAE architectures,
    with support for component-specific blending and advanced interpolation.
    """
    
    def __init__(self, config: VaeMergeConfig):
        """
        Initialize the VAE merger.
        
        Args:
            config: VAE merge configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        self.loader = ModelLoader()
        
        print("💜 Violet Model Merge - VAE Merger")
        print("✨ Beautiful VAE blending for enhanced generation!")
        
    def merge_vaes(self) -> Tuple[str, Dict[str, Any]]:
        """
        Perform VAE merging based on configuration.
        
        Returns:
            Tuple of (output_path, merge_metadata)
        """
        print(f"🎨 Starting {self.config.mode.description} VAE merge...")
        
        # Load VAEs
        vaes = self._load_vaes()
        
        # Perform merge based on mode
        if self.config.mode == VaeMergeMode.WS:
            result = VaeMergeAlgorithms.weighted_sum(
                vaes['vae_0'], vaes['vae_1'], 
                self.config.alpha, self.config.save_half
            )
        elif self.config.mode == VaeMergeMode.LERP:
            result = VaeMergeAlgorithms.linear_interpolation(
                vaes['vae_0'], vaes['vae_1'],
                self.config.alpha, self.config.save_half
            )
        elif self.config.mode == VaeMergeMode.COMP:
            # Ensure we have valid alpha values for component-specific merging
            encoder_alpha = self.config.encoder_alpha if self.config.encoder_alpha is not None else 0.5
            decoder_alpha = self.config.decoder_alpha if self.config.decoder_alpha is not None else 0.5
            result = VaeMergeAlgorithms.component_specific(
                vaes['vae_0'], vaes['vae_1'],
                encoder_alpha, decoder_alpha,
                self.config.save_half
            )
        elif self.config.mode == VaeMergeMode.SLERP:
            result = VaeMergeAlgorithms.spherical_linear(
                vaes['vae_0'], vaes['vae_1'],
                self.config.alpha, self.config.save_half
            )
        else:
            raise ValueError(f"🚨 Unsupported VAE merge mode: {self.config.mode}")
        
        # Save result
        output_path = self._save_result(result, vaes)
        
        # Generate metadata
        metadata = self._generate_metadata(vaes, result)
        
        print(f"✅ VAE merge complete: {output_path}")
        return output_path, metadata
    
    def _load_vaes(self) -> Dict[str, TensorDict]:
        """Load VAE files for merging."""
        vaes = {}
        vae_path = Path(self.config.vae_path)
        
        # Load primary VAEs
        for i, vae_name in enumerate([self.config.vae_0, self.config.vae_1]):
            if vae_name:
                vae_file = vae_path / vae_name
                print(f"📥 Loading VAE {vae_name}...")
                vae_data = self.loader.load_model(vae_file, self.config.device, verify_hash=False)
                vaes[f'vae_{i}'] = vae_data
                vaes[f'vae_{i}_name'] = vae_name
                vaes[f'vae_{i}_path'] = str(vae_file)
        
        # Load optional third VAE for future 3-way merging
        if self.config.vae_2:
            vae_file = vae_path / self.config.vae_2
            print(f"📥 Loading VAE {self.config.vae_2}...")
            vae_data = self.loader.load_model(vae_file, self.config.device, verify_hash=False)
            vaes['vae_2'] = vae_data
            vaes['vae_2_name'] = self.config.vae_2
            vaes['vae_2_path'] = str(vae_file)
        
        return vaes
    
    def _save_result(self, result: TensorDict, vaes: Dict[str, Any]) -> str:
        """Save the merged VAE result."""
        # Determine output path
        vae_path = Path(self.config.vae_path)
        if self.config.save_safetensors:
            output_name = f"{self.config.output}.safetensors"
        else:
            output_name = f"{self.config.output}.pt"
        
        output_path = vae_path / output_name
        
        print(f"💾 Saving merged VAE: {output_name}")
        
        # Create metadata for the merged VAE
        merge_metadata = {
            "violet_model_merge": {
                "version": "1.0.0",
                "merge_type": "vae",
                "mode": self.config.mode.name,
                "mode_description": self.config.mode.description,
                "source_vaes": [vaes.get('vae_0_name', ''), vaes.get('vae_1_name', '')],
                "alpha": self.config.alpha,
                "save_half": self.config.save_half,
                "device": self.config.device
            }
        }
        
        if self.config.mode == VaeMergeMode.COMP:
            merge_metadata["violet_model_merge"]["encoder_alpha"] = self.config.encoder_alpha
            merge_metadata["violet_model_merge"]["decoder_alpha"] = self.config.decoder_alpha
        
        # Save the model using the same pattern as merge_model.py
        if self.config.save_safetensors:
            with torch.no_grad():
                safetensors.torch.save_file(
                    result, 
                    str(output_path),
                    metadata={k: str(v) for k, v in merge_metadata["violet_model_merge"].items()}
                )
        else:
            torch.save({"state_dict": result}, str(output_path))
        
        return str(output_path)
    
    def _generate_metadata(self, vaes: Dict[str, Any], result: TensorDict) -> Dict[str, Any]:
        """Generate comprehensive metadata for the merge."""
        return {
            "merge_info": {
                "mode": self.config.mode.name,
                "mode_description": self.config.mode.description,
                "alpha": self.config.alpha,
                "encoder_alpha": getattr(self.config, 'encoder_alpha', None),
                "decoder_alpha": getattr(self.config, 'decoder_alpha', None),
                "source_vaes": [vaes.get('vae_0_name', ''), vaes.get('vae_1_name', '')],
                "tensor_count": len(result),
                "save_half": self.config.save_half,
                "device": self.config.device
            }
        }


def create_vae_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for VAE merging CLI."""
    parser = argparse.ArgumentParser(
        description="💜 Violet Model Merge - VAE Merging Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎨 VAE Merge Modes:
  WS     Weighted Sum - Simple alpha blending
  LERP   Linear Interpolation - Smooth transitions  
  COMP   Component Specific - Different encoder/decoder weights
  SLERP  Spherical Linear - Advanced smooth blending

✨ Examples:
  python merge_vae.py WS vae1.safetensors vae2.safetensors --alpha 0.3
  python merge_vae.py COMP vae1.pt vae2.pt --encoder-alpha 0.2 --decoder-alpha 0.7
  python merge_vae.py SLERP vae1.safetensors vae2.safetensors --alpha 0.5 --save-half
"""
    )
    
    # Positional arguments
    parser.add_argument("mode", choices=['WS', 'LERP', 'COMP', 'SLERP'], 
                       help="VAE merge mode")
    parser.add_argument("vae_0", help="First VAE file")
    parser.add_argument("vae_1", help="Second VAE file")
    
    # Optional VAE
    parser.add_argument("--vae-2", help="Third VAE file (for future 3-way merging)")
    
    # Merge parameters
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Merge ratio (0.0-1.0, default: 0.5)")
    parser.add_argument("--encoder-alpha", type=float,
                       help="Encoder merge ratio for COMP mode")
    parser.add_argument("--decoder-alpha", type=float,
                       help="Decoder merge ratio for COMP mode")
    
    # Paths
    parser.add_argument("--vae-path", default="vae",
                       help="VAE directory path (default: vae)")
    parser.add_argument("--output", default="merged_vae",
                       help="Output filename (default: merged_vae)")
    
    # Device and precision
    parser.add_argument("--device", default="cpu",
                       help="Device for processing (default: cpu)")
    parser.add_argument("--save-half", action="store_true",
                       help="Save in half precision (FP16)")
    parser.add_argument("--save-safetensors", action="store_true", default=True,
                       help="Save as safetensors format (default: True)")
    
    return parser


def main():
    """Main entry point for VAE merging CLI."""
    try:
        parser = create_vae_argument_parser()
        args = parser.parse_args()
        
        # Create configuration
        config = VaeMergeConfig(
            mode=VaeMergeMode[args.mode],
            vae_path=args.vae_path,
            vae_0=args.vae_0,
            vae_1=args.vae_1,
            vae_2=args.vae_2,
            output=args.output,
            alpha=args.alpha,
            encoder_alpha=args.encoder_alpha,
            decoder_alpha=args.decoder_alpha,
            device=args.device,
            save_half=args.save_half,
            save_safetensors=args.save_safetensors
        )
        
        # Perform merge
        merger = VaeMerger(config)
        output_path, metadata = merger.merge_vaes()
        
        print("\n🎉 VAE merge completed successfully!")
        print(f"📁 Output: {output_path}")
        print(f"🎨 Mode: {config.mode.description}")
        print(f"⚖️ Alpha: {config.alpha}")
        
        if config.mode == VaeMergeMode.COMP:
            print(f"🔧 Encoder α: {config.encoder_alpha}")
            print(f"🔧 Decoder α: {config.decoder_alpha}")
        
    except KeyboardInterrupt:
        print("\n🚫 VAE merge cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n🚨 VAE merge failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()