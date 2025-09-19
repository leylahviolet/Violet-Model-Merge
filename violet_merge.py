#!/usr/bin/env python3
"""
💜 Violet Model Merge - Unified CLI Entry Point
✨ Beautiful model and VAE merging for AI artists!

This script provides a unified interface for both model and VAE merging,
automatically routing to the appropriate merger based on the merge type.

Usage:
  python violet_merge.py model <model_args>  # Model merging
  python violet_merge.py vae <vae_args>      # VAE merging
  
Author: Violet Tools (Original implementation by Faildes)
License: MIT
"""

import argparse
import sys
import os

# Add lib directory to path for imports
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib')
sys.path.insert(0, lib_path)

def create_unified_parser() -> argparse.ArgumentParser:
    """Create the unified argument parser for both model and VAE merging."""
    
    parser = argparse.ArgumentParser(
        description="💜 Violet Model Merge - Unified merging interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎨 Merge Types:
  model    Merge AI model checkpoints (SD, SDXL, Flux, etc.)
  vae      Merge VAE files for enhanced generation

✨ Examples:
  Model merging:
    python violet_merge.py model WS model_a.safetensors model_b.safetensors --alpha 0.3
    
  VAE merging:
    python violet_merge.py vae WS vae_a.safetensors vae_b.safetensors --alpha 0.5
    
  Component-specific VAE merge:
    python violet_merge.py vae COMP vae_a.pt vae_b.pt --encoder-alpha 0.2 --decoder-alpha 0.7

💜 For detailed help on each merge type:
    python violet_merge.py model --help
    python violet_merge.py vae --help
"""
    )
    
    # Add subcommands for model and vae merging
    subparsers = parser.add_subparsers(dest='merge_type', help='Type of merge to perform')
    
    # Model merge subcommand - just forward to merge_model.py
    model_parser = subparsers.add_parser('model', help='Merge AI model checkpoints')
    model_parser.add_argument('args', nargs=argparse.REMAINDER, 
                             help='Arguments for model merging (see merge_model.py --help)')
    
    # VAE merge subcommand - just forward to merge_vae.py  
    vae_parser = subparsers.add_parser('vae', help='Merge VAE files')
    vae_parser.add_argument('args', nargs=argparse.REMAINDER,
                           help='Arguments for VAE merging (see merge_vae.py --help)')
    
    return parser

def main():
    """Main entry point for unified merging interface."""
    
    parser = create_unified_parser()
    args = parser.parse_args()
    
    if not args.merge_type:
        parser.print_help()
        print("\n🚨 Please specify merge type: 'model' or 'vae'")
        sys.exit(1)
    
    # Route to appropriate merger
    if args.merge_type == 'model':
        print("💜 Routing to model merger...")
        from merge_model import main as model_main
        
        # Replace sys.argv with the model arguments
        sys.argv = ['merge_model.py'] + args.args
        model_main()
        
    elif args.merge_type == 'vae':
        print("💜 Routing to VAE merger...")
        from merge_vae import main as vae_main
        
        # Replace sys.argv with the VAE arguments
        sys.argv = ['merge_vae.py'] + args.args
        vae_main()
        
    else:
        print(f"🚨 Unknown merge type: {args.merge_type}")
        sys.exit(1)

if __name__ == "__main__":
    main()