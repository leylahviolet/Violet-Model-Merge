# 💜 Violet Model Merge

> *A sophisticated, artist-friendly model merging toolkit for Stable Diffusion and Flux.1*
>
> *Derived from [Chattiori Model Merger](https://github.com/faildes) by Chattiori*

[![Version](https://img.shields.io/badge/version-1.3.0-8A2BE2?style=for-the-badge&logoColor=white)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-Optional-green.svg)](https://developer.nvidia.com/cuda-zone)

**Violet Model Merge** is a fast, deterministic checkpoint merger for **Stable Diffusion** (SD1.x/2.x/XL) and **Flux.1** models. Built for AI artists who want precise control over model blending with an intuitive interface.

---

## ✨ Features

🚀 **Multiple Merge Algorithms** — 20+ sophisticated merge modes from simple weighted sums to advanced cosine structure blending  
💻 **Interactive Notebook** — Artist-friendly Jupyter interface with comprehensive documentation and clean error handling  
📊 **Metadata Editor** — CSV-based SafeTensors metadata management for batch editing descriptions, authors, and tags  
⚡ **GPU Acceleration** — CUDA support for faster merging of large models  
🛡️ **Robust Error Handling** — Clear, friendly error messages with detailed logging  
🎯 **Deterministic Results** — Consistent, reproducible merges every time  
🔧 **Flexible I/O** — Support for `.safetensors`, `.ckpt`, and multiple precision formats  
📊 **Progress Tracking** — Real-time progress bars and status updates  
🎨 **VAE Integration** — Seamless VAE baking and model pruning  

### Supported Merge Methods

| **Method** | **Description** | **Best For** |
|------------|-----------------|--------------|
| **WS** (Weighted Sum) | Linear interpolation between models | Basic style blending |
| **AD** (Add Difference) | Preserves structure while adding characteristics | Advanced style transfer |
| **SIG** (Sigmoid) | Smooth, non-linear blending curves | Natural artistic transitions |
| **DARE** | Experimental drop-and-rescale algorithm | Novel model combinations |
| **FREQ** | Frequency-domain convolution merging | Technical fine-tuning |
| **TIES** | Structured parameter combination | Complex multi-model blends |
| And 15+ more... | Including cosine, sparse, and geometric methods | Every creative workflow |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/leylahviolet/Violet-Model-Merge.git
cd Violet-Model-Merge

# Install dependencies
pip install -r requirements.txt
```

### Choose Your Interface

#### 📓 **[Interactive Notebook](#-jupyter-notebook-usage-recommended) (Recommended)**
Perfect for AI artists who want a guided, visual experience:

```bash
jupyter lab violet_merge.ipynb
```

#### ⚡ **[Command Line Interface](#-command-line-usage)**
For advanced users who prefer terminal workflows:

```bash
python lib/merge_model.py WS models "model_a.safetensors" "model_b.safetensors" --alpha 0.4
```

#### 📊 **[Metadata Manager](#-metadata-management)**
CSV-based SafeTensors metadata editing for batch operations:

```bash
jupyter lab metadata_manager.ipynb
```

---

## 📊 Metadata Management

The **metadata manager notebook** provides a powerful, user-friendly way to edit SafeTensors metadata in bulk using CSV files. Perfect for maintaining clean, consistent metadata across your entire model collection! 💜

### 🎯 Metadata Features

- 📤 **Export to CSV** — Extract metadata from models/VAEs into editable spreadsheets
- ✏️ **Easy Editing** — Use Excel, Google Sheets, or any CSV editor you prefer  
- 📥 **Safe Import** — Apply changes back with automatic backups and validation
- 🔒 **Non-destructive** — Original files are backed up before any changes
- 🎯 **Batch Processing** — Edit hundreds of models at once efficiently

### 🚀 Quick Workflow

1. **📊 Export** → `metadata_manager.ipynb` creates editable CSV from your models
2. **✏️ Edit** → Open CSV in your favorite editor, update descriptions, authors, tags
3. **💾 Import** → Apply changes back to your models with automatic validation

### Example Use Cases

- **🏷️ Bulk Tagging** — Add consistent tags across model collections
- **👤 Author Updates** — Set proper attribution for all your models
- **📝 Descriptions** — Add detailed descriptions for better organization
- **🔄 Version Control** — Track model versions and modifications
- **🧹 Cleanup** — Standardize metadata formatting across collections

---

## 📋 Requirements

**Core Dependencies:**
- **Python** 3.8+ (3.12+ recommended)
- **PyTorch** 2.0+ with CUDA support (optional but recommended)
- **safetensors** for fast, secure model loading
- **diffusers** for modern Stable Diffusion support

**Optional Enhancements:**
- **CUDA** toolkit for GPU acceleration
- **Jupyter Lab** for the interactive notebook experience

**All dependencies are automatically installed via:**
```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
Violet-Model-Merge/
├── 📒 violet_merge.ipynb          # Main interactive notebook
├── � metadata_manager.ipynb      # CSV-based metadata editor
├── �📁 lib/                        # Core Python modules
│   ├── merge_model.py             # Main merging engine
│   ├── utils.py                   # Utility functions
│   ├── metadata_csv.py            # Metadata management
│   └── lora_bake.py              # LoRA integration
├── 📁 models/                     # Your model files (.safetensors, .ckpt)
├── 📁 vae/                        # VAE files for baking
├── 📄 pyproject.toml             # Modern Python packaging
├── 📄 CHANGELOG.md               # Version history and updates
└── 📄 requirements.txt           # Python dependencies
```

---

## 🎨 Jupyter Notebook Usage (Recommended)

The **interactive notebook** provides the best experience for AI artists with guided examples, comprehensive documentation, and clean error handling.

### Getting Started

1. **Launch Jupyter Lab**:
   ```bash
   jupyter lab violet_merge.ipynb
   ```

2. **Configure Your Paths** (First cell):
   ```python
   # Set your model and VAE directories
   models_path = "../../ComfyUI/models/checkpoints"
   vae_path = "../../ComfyUI/models/vae"
   
   # Choose your models
   model_1 = "realistic_base.safetensors"
   model_2 = "anime_style.safetensors"
   vae_model = "anything-v4.0.vae.pt"
   ```

3. **Run Merges with Beautiful Progress Tracking**:
   ```python
   # Simple weighted merge
   result = run_merge(
       mode="WS",
       model0=model_1,
       model1=model_2,
       alpha=0.3,                        # 70% model_1, 30% model_2
       output_name="realistic_anime_blend"
   )
   ```

### ✨ Notebook Features

- 📚 **Comprehensive Documentation** — Each merge method explained with when/why to use it
- 🔧 **Ready-to-Run Examples** — Copy, paste, and customize for your models
- 📊 **Clean Progress Tracking** — Real-time updates without technical noise
- 🎛️ **Easy Configuration** — Set paths once, use everywhere
- 🛡️ **Friendly Error Handling** — Clear, non-blocking error messages
- 📝 **Detailed Logging** — Full technical details saved to `last_merge.log`

### Example Workflows

#### **Basic Style Blending**
```python
# Blend two artistic styles
result = run_merge(
    mode="SIG",                          # Sigmoid for smooth blending
    model0="photoreal_base.safetensors",
    model1="artistic_style.safetensors", 
    alpha=0.25,                          # Subtle artistic influence
    output_name="photoreal_with_style"
)
```

#### **Advanced 3-Model Structure Merge**
```python
# Preserve base structure, inject style and details
result = run_merge(
    mode="AD",                           # Add Difference
    model0="base_structure.safetensors", # Foundation model
    model1="style_details.safetensors",  # Style characteristics
    model2="fine_details.safetensors",   # Additional features
    alpha=0.3,                           # Style strength
    cosine0=True,                        # Preserve model0 structure
    memo="Complex artistic blend"
)
```

---

## ⚡ Command Line Usage

For advanced users who prefer terminal workflows and automation scripts.

### Basic Syntax

```bash
python lib/merge_model.py <MODE> <model_path> <model_0> <model_1> [OPTIONS]
```

### Quick Examples

#### **Simple Weighted Merge**
```bash
python lib/merge_model.py WS models "A.safetensors" "B.safetensors" \
  --alpha 0.45 --output merged_ws --save_safetensors --save_half
```

#### **Cosine Structure Merge**
```bash
python lib/merge_model.py WS models "A.safetensors" "B.safetensors" \
  --cosine1 --alpha 0.35 --output merged_cos1
```

#### **3-Model Advanced Merge**
```bash
python lib/merge_model.py AD models "A.safetensors" "B.safetensors" \
  --model_2 "C.safetensors" --alpha 0.25 --beta 0.15 --output merged_blend
```

#### **GPU-Accelerated with VAE**
```bash
python lib/merge_model.py SIG models "base.safetensors" "style.safetensors" \
  --alpha 0.4 --vae "vae.pt" --device cuda --output gpu_blend
```

### Advanced Examples

#### **Experimental DARE Merge**
```bash
python lib/merge_model.py DARE models "A.safetensors" "B.safetensors" \
  --alpha 0.4 --beta 0.3 --seed 42 --output dare_experiment
```

#### **Frequency-Band Blending**
```bash
python lib/merge_model.py FREQ models "A.safetensors" "B.safetensors" \
  --model_2 "C.safetensors" --alpha 0.4 --output freq_blend
```

#### **No Interpolation + Finetuning**
```bash
python lib/merge_model.py NoIn models "FluxModel.safetensors" "dummy.safetensors" \
  --fine "2,0,1,0,0,5" --output flux_finetuned
```

#### **Metadata Reading**
```bash
python lib/merge_model.py RM models "A.safetensors" "dummy.safetensors" --output meta_dump
```

---

## 🔧 Merge Methods Reference

### Core Methods
- **WS** (Weighted Sum): Basic linear interpolation
- **AD** (Add Difference): A + (B - C) structure preservation
- **SIG** (Sigmoid): Non-linear smooth blending
- **INV** (Inverse Sigmoid): Reverse sigmoid curves

### Advanced Algorithms  
- **DARE**: Drop and Re-scale experimental method
- **TIES**: Task Interference-aware merging
- **FREQ**: Frequency domain convolution
- **SPRSE**: Sparse top-k parameter selection

### Structure Control
- **Cosine**: Preserve model structure while blending parameters
- **Tensors**: Fine-grained layer-wise control
- **Block Weight**: Precision control over UNet blocks

### Special Operations
- **NoIn**: No interpolation (finetuning only)
- **RM**: Read metadata without merging
- **PINS**: Pin specific parameters during merge

---

## 🎛️ Configuration Options

### Alpha/Beta Blending
- **--alpha**: Primary blend ratio (0.0 = model0, 1.0 = model1)
- **--beta**: Secondary ratio for 3-model merges
- **--gamma**: Tertiary ratio for complex algorithms

### Structure Preservation  
- **--cosine0**, **--cosine1**, **--cosine2**: Preserve specific model structure
- **--use_dif_10**, **--use_dif_20**, **--use_dif_21**: Reuse model diffs internally

### Output Control
- **--output**: Output filename (without extension)
- **--save_safetensors**: Save in safetensors format
- **--save_half**: Use fp16 precision
- **--save_quarter**: Use fp8 precision (experimental)

### Performance

- **⚡ Optimized Execution** — Dramatically faster processing with streamlined merge logic
- **🚀 Real Performance** — 3-model merges complete in seconds, not hours
- **💾 Memory Efficient** — Optimized for large model handling with reduced memory footprint  
- **--device**: Choose cpu/cuda/auto for processing
- **--vae**: Bake VAE into the merged model
- **--prune**: Remove unnecessary parameters

> **Performance Note**: Version 1.2.1 includes major performance optimizations resulting in dramatically faster merge times. What used to take hours now completes in seconds! 🎉

---

## VAE, Pruning & DTypes

- **VAE bake:** `--vae path/to/vae.safetensors` replaces `first_stage_model.*`
- **Pruning:** `--prune` (with `--keep_ema`) to slim the checkpoint; architecture-aware
- **DTypes:** `--save_half` (fp16) is widely supported; `--save_quarter` (fp8) is experimental

---

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed version history and updates.

**Latest highlights:**

- ✨ **v1.2.1** — Major performance boost! Optimized merge execution with dramatically faster processing
- ✨ **v1.2.0** — Interactive Jupyter notebook with artist-friendly interface
- 🛡️ **v1.1.0** — Enhanced error handling and progress tracking
- 🔧 **v1.0.0** — Project restructure with modern packaging

---

## 🙏 Acknowledgments

The **Violet Model Merge** builds upon the incredible work of the open-source AI community. Special thanks to:

### 💜 **[Chattiori](https://github.com/faildes)** — Original Architect
*The brilliant mind behind the core merging algorithms and mathematical foundations that make this tool possible*

### 🌟 **Core Contributors**
- **[eyriewow](https://github.com/eyriewow/merge-models)** — Original merge-models architecture
- **[hako-mikan](https://github.com/hako-mikan/sd-webui-supermerger)** — Advanced merge algorithms and cosine methods
- **[lopho](https://github.com/lopho/stable-diffusion-prune)** & **[arenasys](https://github.com/arenasys/stable-diffusion-webui-model-toolkit)** — Model pruning techniques
- **[idelairre](https://github.com/idelairre/sd-merge-models)** — Geometric, Sigmoid, and Max merge implementations
- **[s1dlx](https://github.com/s1dlx/meh)** — Multiply Difference and Similarity Add Difference algorithms
- **[bbc-mc](https://github.com/bbc-mc/sdweb-merge-block-weighted-gui)** — Block-weighted merging interface
- **[martyn](https://github.com/martyn/safetensors-merge-supermario)** — DARE algorithm implementation
- **[mlfoundations/wise-ft](https://github.com/mlfoundations/wise-ft)** — Theoretical foundation for DARE method
- **[LatteLeopard](https://civitai.com/user/LatteLeopard)** - Merge method descriptions

This project stands on the shoulders of giants. 💜

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Built with 💜 by the AI art community** — *For artists, by artists*