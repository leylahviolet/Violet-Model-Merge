# 🎨 Chatt---

## ✨ Features

🚀 **Multiple Merge Algorithms** — 20+ sophisticated merge modes from simple weighted sums to advanced cosine structure blending  
💻 **Interactive Notebook** — Artist-friendly Jupyter interface with comprehensiv### Basic Syntax

```bash
python lib/merge.py <MODE> <model_## ⚡ Command Line Usage

For advanced users who prefer terminal workflows and automation scripts.

### Basic Syntax

```bash
python lib/merge.py <MODE> <model_path> <model_0> <model_1> [OPTIONS]
```

### Quick Examples

#### **Simple Weighted Merge**

```bash
python lib/merge.py WS models "A.safetensors" "B.safetensors" \
  --alpha 0.45 --output merged_ws --save_safetensors --save_half
```

#### **Cosine Structure Merge**

```bash
python lib/merge.py WS models "A.safetensors" "B.safetensors" \
  --cosine1 --alpha 0.35 --output merged_cos1
```

#### **3-Model Advanced Merge**

```bash
python lib/merge.py AD models "A.safetensors" "B.safetensors" \
  --model_2 "C.safetensors" --alpha 0.25 --beta 0.15 --output merged_blend
```

#### **GPU-Accelerated with VAE**

```bash
python lib/merge.py SIG models "base.safetensors" "style.safetensors" \
  --alpha 0.4 --vae "vae.pt" --device cuda --output gpu_blend
```

### Advanced Examples

#### **Experimental DARE Merge**

```bash
python lib/merge.py DARE models "A.safetensors" "B.safetensors" \
  --alpha 0.4 --beta 0.3 --seed 42 --output dare_experiment
```

#### **Frequency-Band Blending**

```bash
python lib/merge.py FREQ models "A.safetensors" "B.safetensors" \
  --model_2 "C.safetensors" --alpha 0.4 --output freq_blend
```

#### **No Interpolation + Finetuning**

```bash
python lib/merge.py NoIn models "FluxModel.safetensors" "dummy.safetensors" \
  --fine "2,0,1,0,0,5" --output flux_finetuned
```

#### **Metadata Reading**

```bash
python lib/merge.py RM models "A.safetensors" "dummy.safetensors" --output meta_dump
```

---PTIONS]
```

### Quick Examples

#### **Simple Weighted Merge**

```bash
python lib/merge.py WS models "A.safetensors" "B.safetensors" \
  --alpha 0.45 --output merged_ws --save_safetensors --save_half
```

#### **Cosine Structure Merge**

```bash
python lib/merge.py WS models "A.safetensors" "B.safetensors" \
  --cosine1 --alpha 0.35 --output merged_cos1
```

#### **3-Model Advanced Merge**

```bash
python lib/merge.py AD models "A.safetensors" "B.safetensors" \
  --model_2 "C.safetensors" --alpha 0.25 --beta 0.15 --output merged_blend
```

#### **GPU-Accelerated with VAE**

```bash
python lib/merge.py SIG models "base.safetensors" "style.safetensors" \
  --alpha 0.4 --vae "vae.pt" --device cuda --output gpu_blend
```

### Advanced Examples

#### **Experimental DARE Merge**

```bash
python lib/merge.py DARE models "A.safetensors" "B.safetensors" \
  --alpha 0.4 --beta 0.3 --seed 42 --output dare_experiment
```

#### **Frequency-Band Blending**

```bash
python lib/merge.py FREQ models "A.safetensors" "B.safetensors" \
  --model_2 "C.safetensors" --alpha 0.4 --output freq_blend
```

#### **No Interpolation + Finetuning**

```bash
python lib/merge.py NoIn models "FluxModel.safetensors" "dummy.safetensors" \
  --fine "2,0,1,0,0,5" --output flux_finetuned
```

#### **Metadata Reading**

```bash
python lib/merge.py RM models "A.safetensors" "dummy.safetensors" --output meta_dump
```

--- documentation and examples  
🔧 **Flexible I/O** — Supports `.ckpt`, `.safetensors`, and `.pt` VAE files with automatic architecture detection  
⚡ **GPU Acceleration** — CUDA support for faster processing with large models  
🎯 **Precise Control** — Block-weighted merging, elemental ratios, and structure-preserving cosine modes  
📊 **Rich Metadata** — Comprehensive merge tracking with SHA256 verification and recipe storage  
🛡️ **Robust Error Handling** — Clean, non-technical error reporting with detailed logging

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/leylahviolet/Chattiori-Model-Merger.git
cd Chattiori-Model-Merger

# Install dependencies
pip install -r requirements.txt
```

### Choose Your Interface

#### 🎨 **[Interactive Notebook](#-jupyter-notebook-usage-recommended) (Recommended)**
Perfect for AI artists who want a guided, visual experience:

```bash
jupyter lab merge_runner.ipynb
```

#### ⚡ **[Command Line Interface](#-command-line-usage)**
For advanced users who prefer terminal workflows:

```bash
python lib/merge.py WS models "model_a.safetensors" "model_b.safetensors" --alpha 0.4
```l Merger

> *A sophisticated, artist-friendly model merging toolkit for Stable Diffusion and Flux.1*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-Optional-green.svg)](https://developer.nvidia.com/cuda-zone)

**Chattiori Model Merger** is a fast, deterministic checkpoint merger for **Stable Diffusion** (SD1.x/2.x/XL) and **Flux.1** models. Built for AI artists who want precise control over model blending with an intuitive interface.

---

## Highlights

- ✅ 2-model and 3-model merges  
- ✅ **Cosine structure modes** (`--cosine0/1/2`) to keep one model’s **structure** while blending **details** from others  
- ✅ Block-weighted / Elemental ratios, plus random ratio samplers  
- ✅ Extra modes: orthogonal delta, sparse top-k, channel-wise cosine gate, frequency-band mix, etc.  
- ✅ VAE bake-in, pruning, EMA keep, fp16/fp8 saving, rich metadata

---

## Architectures & Formats

- **Architectures:** SD 1.x / 2.x / XL and **Flux.1** (via `detect_arch`)
- **Checkpoints:** `.ckpt` (PyTorch) and `.safetensors`
- **VAE:** Optional bake-in with `--vae`
- **DTypes:** fp32 (default), **fp16** (`--save_half`), **fp8** (`--save_quarter`, experimental)

---

## Modes

Pick one `mode` (first positional arg). Some require a third model or `beta`.

| Code   | Name                         | Needs 3rd model | Needs `beta` | Notes |
|-------:|------------------------------|:-------------:|:------------:|------|
| `WS`   | Weighted Sum                 |       ✗       |      ✗       | Standard linear mix |
| `SIG`  | Sigmoid                      |       ✗       |      ✗       | Smooth non-linear |
| `GEO`  | Geometric                    |       ✗       |      ✗       | Geometric mean |
| `MAX`  | Max                          |       ✗       |      ✗       | Element-wise max |
| `AD`   | Add Difference               |       ✓       |      ✗       | Uses (model1 − model2) |
| `sAD`  | Smooth Add Difference        |       ✓       |      ✗       | Median+Gaussian filtered diff |
| `MD`   | Multiply Difference          |       ✓       |      ✓       | Non-linear diff product |
| `SIM`  | Similarity Add Difference    |       ✓       |      ✓       | Similarity-aware lerp |
| `TD`   | Train Difference             |       ✓       |      ✗       | Data-dependent scaling |
| `TRS`  | Triple Sum                   |       ✓       |      ✓       | α/β tri-blend |
| `TS`   | Tensor Sum                   |       ✗       |      ✓       | Splice by ranges |
| `ST`   | Sum Twice                    |       ✓       |      ✓       | Two-stage sum |
| `NoIn` | No Interpolation             |       ✗       |      ✗       | No blending; combine with `--fine` |
| `RM`   | Read Metadata                |       ✗       |      ✗       | Prints SHA256 + metadata and exits |
| `DARE` | DARE                         |       ✗       |      ✓       | Stochastic delta resampling |
| `ORTHO`| Orthogonalized Delta         |       ✗       |      ✗       | Apply diff orthogonal to base |
| `SPRSE`| Sparse Top-k Delta           |   ✓    |      ✗       | Applies largest diffs (uses α) |
| `NORM` | Norm/Direction Split         |       ✗       |      ✗       | Blend magnitude & direction separately |
| `CHAN` | Channel-wise Cosine Gate     |   ✓    |      ✗       | Gate per output channel (Conv/Linear) |
| `FREQ` | Frequency-Band Blend         |   ✓    |      ✗       | Low/high-freq mix for Conv kernels |

---

## Cosine Structure Modes

Use **exactly one** of: `--cosine0`, `--cosine1`, `--cosine2` (mutually exclusive).

- `--cosine0`: **Model 0** defines the **structure**; inject details from Model 1 (and Model 2 if present)
- `--cosine1`: **Model 1** defines the **structure**; inject details from the others
- `--cosine2`: **Model 2** defines the **structure**; inject details from Model 0 then Model 1 (**requires** `--model_2`)

These modes compute per-layer cosine stats between the **base (structure)** and **detail** models, then blend using your **`alpha`** (detail A) and **`beta`** (detail B) with block/elemental weights.

---

## Ratios & Block Weights

- `--alpha`, `--beta` accept:
  - A **float** (`0.45`, etc.)
  - **Merge Block Weights** string (19/25-length, etc.)
  - **Elemental Merge** syntax
- `--rand_alpha`, `--rand_beta` accept `"MIN, MAX[, SEED][, elemental...]"`  
  If `SEED` omitted, a random seed is generated.
- **XL note:** 25-length weights auto-convert for XL; 19-length is supported.

Rule of thumb: `--alpha 0.5` ≈ 50/50 of model0 & model1 (lower alpha → closer to model0).

---

## CLI

### Positional

```
mode model_path model_0 model_1
```

- `mode`: one of the codes above
- `model_path`: directory containing the checkpoints
- `model_0`: filename of the first model
- `model_1`: filename of the second model (required for merging modes)
- `--model_2`: filename of the third model (required by some modes, or when using `--cosine2`)

### Options

- Ratios:
  - `--alpha <v>` / `--rand_alpha "<min,max[,seed][,elemental...]>"`
  - `--beta  <v>` / `--rand_beta  "<min,max[,seed][,elemental...]>"`
- Cosine structure (mutually exclusive):
  - `--cosine0` | `--cosine1` | `--cosine2` (`--cosine2` requires `--model_2`)
- Finetune:
  - `--fine "<comma,separated,values>"`  
    SDXL: classic pattern; Flux.1: pattern-based scaling for keys like `double_block`, `img_in`, `txt_in`, `time`, `out`, etc.
- I/O & formats:
  - `--vae <path>` bake into `first_stage_model.*`
  - `--save_safetensors` (otherwise saves `.ckpt`)
  - `--save_half` (fp16), `--save_quarter` (fp8, experimental)
  - `--output <name>` output filename (no extension), `--force` to overwrite/autoname
  - `--delete_source` delete source checkpoints after saving
  - `--no_metadata` save without metadata
- Pruning:
  - `--prune` (optionally `--keep_ema` to keep EMA only)
- Names:
  - `--m0_name`, `--m1_name`, `--m2_name`
- Device:
  - `--device cpu|cuda:x` (default `cpu`)
- DARE reproducibility:
  - `--seed <int>`
- Differences:
  - `--use_dif_10`, `--use_dif_20`, `--use_dif_21` to reuse model diffs internally

---

## 🎨 Jupyter Notebook Usage (Recommended)

The **interactive notebook** provides the best experience for AI artists with guided examples, comprehensive documentation, and clean error handling.

### Getting Started

1. **Launch Jupyter Lab**:
   ```bash
   jupyter lab merge_runner.ipynb
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
```bash
python lib/merge.py WS models "A.safetensors" "B.safetensors"   --alpha 0.45 --output merged_ws --save_safetensors --save_half
```

### 2) Cosine structure: keep model1’s structure, inject model0 details
```bash
python lib/merge.py WS models "A.safetensors" "B.safetensors"   --cosine1 --alpha 0.35 --output merged_cos1
```

### 3) Cosine2 (3 models): keep model2’s structure; inject 0 then 1 with α/β




---

## VAE, Pruning & DTypes

- **VAE bake:** `--vae path/to/vae.safetensors` replaces `first_stage_model.*`
- **Pruning:** `--prune` (with `--keep_ema`) to slim the checkpoint; architecture-aware
- **DTypes:** `--save_half` (fp16) is widely supported; `--save_quarter` (fp8) is experimental

---

## Output & Metadata

- **`.safetensors`** (recommended) with metadata (unless `--no_metadata`), or **`.ckpt`** (`{"state_dict": ...}`)
- Metadata includes:
  - SHA256s, legacy hashes
  - `sd_merge_recipe` (method, α/β info, fp, VAE, pruning flags, etc.)
  - `sd_merge_models` (per-model provenance)

---

## Colab Quickstart

```bash
pip install torch safetensors

git clone https://github.com/Faildes/merge-models
cd merge-models

python lib/merge.py WS models "A.safetensors" "B.safetensors" --alpha 0.45 --output merged
```

> Depending on your environment, you may need `python3`, `py`, or `conda run -n <env> python`.

---

## Troubleshooting

- **Cosine switches are exclusive:** use one of `--cosine0/1/2`.  
- **3-model requirements:** `AD`, `sAD`, `MD`, `SIM`, `TD`, `TRS`, `ST` (and in this build: `SPRSE`, `CHAN`, `FREQ`) require `--model_2`.  
- **Low VRAM:** stick to `--device cpu` (default). GPU merging needs VRAM ≈ model sizes (safetensors) or ~1.15× (ckpt).  
- **XL block weights:** 25-length auto-converted; 19-length supported.  
- **FREQ mode:** applies only to Conv tensors with kernel ≥ 3×3; others fall back to simple blending.  
- **Flux finetune:** If some keys don’t react, extend the key substrings (`double_block`, `img_in`, `txt_in`, `time`, `out`, etc.) to match your checkpoint.

---

## 🙏 Acknowledgements

Special thanks to **[Faildes](https://github.com/Faildes)** for the original implementation that serves as the foundation for this enhanced version.

Additional acknowledgements to the incredible community contributors:

- **[AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)** — Design inspiration and ecosystem leadership
- **[eyriewow](https://github.com/eyriewow/merge-models)** — Original merge-models architecture
- **[hako-mikan](https://github.com/hako-mikan/sd-webui-supermerger)** — Advanced merge algorithms and cosine methods
- **[lopho](https://github.com/lopho/stable-diffusion-prune)** & **[arenasys](https://github.com/arenasys/stable-diffusion-webui-model-toolkit)** — Model pruning techniques
- **[idelairre](https://github.com/idelairre/sd-merge-models)** — Geometric, Sigmoid, and Max merge implementations
- **[s1dlx](https://github.com/s1dlx/meh)** — Multiply Difference and Similarity Add Difference algorithms
- **[bbc-mc](https://github.com/bbc-mc/sdweb-merge-block-weighted-gui)** — Block-weighted merging interface
- **[martyn](https://github.com/martyn/safetensors-merge-supermario)** — DARE algorithm implementation
- **[mlfoundations/wise-ft](https://github.com/mlfoundations/wise-ft)** — Theoretical foundation for DARE method
