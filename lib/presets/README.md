# 💜 Violet Model Merge - Preset Library

This directory contains **MBW (Model Block Weight)** presets for advanced model merging techniques.

## 📁 Files

- **`mbwpresets_master.txt`** — Master template with all default preset patterns
- **`mbwpresets.txt`** — User-customizable presets (auto-created from master if missing)

## 🎨 What Are MBW Presets?

MBW presets define **26 weight values** corresponding to each UNet block in Stable Diffusion models. They enable **surgical precision** in model merging by controlling exactly how much each model contributes to specific network layers.

### 🧬 Preset Types

#### **Gradient Patterns**
- **`GRAD_V`** — V-shaped gradient (starts high, dips to zero, ends high)
- **`GRAD_A`** — A-shaped gradient (starts low, peaks in middle, ends low)

#### **Flat Patterns**
- **`FLAT_25`** — Constant 25% blend across all blocks
- **`FLAT_75`** — Constant 75% blend across all blocks

#### **Wrap Patterns** (Target specific layer ranges)
- **`WRAP08`** — Only affects first 4 and last 4 blocks
- **`WRAP12`** — Only affects first 6 and last 6 blocks
- **`WRAP14`** — Only affects first 7 and last 7 blocks
- **`WRAP16`** — Only affects first 8 and last 8 blocks

#### **Structural Patterns**
- **`MID12_50`** — Only affects middle 12 blocks at 50% strength
- **`OUT07`** — Only affects final 7 output layers
- **`OUT12`** — Only affects final 12 layers
- **`RING08_SOFT`** — Ring pattern with soft transitions

#### **Mathematical Curves**
- **`SMOOTHSTEP`** — Smooth S-curve transition
- **`COSINE`** — Cosine wave pattern
- **`TRUE_CUBIC_HERMITE`** — Cubic Hermite interpolation
- **`FAKE_CUBIC_HERMITE`** — Simplified cubic curve

#### **Special Patterns**
- **`ALL_A`** — 100% Model A (all zeros)
- **`ALL_B`** — 100% Model B (all ones)

## 🔧 Usage in Violet Model Merge

These presets are automatically loaded by the `PresetManager` class and used for:

1. **Block-wise merging** — Control which layers get affected
2. **Structural preservation** — Keep certain network parts intact
3. **Creative effects** — Apply mathematical curves to blending
4. **Professional workflows** — Use industry-tested patterns

## ✨ Customization

You can:
- **Edit `mbwpresets.txt`** to customize existing presets
- **Add new presets** following the same format
- **Create backup copies** before experimenting
- **Reset to defaults** by deleting `mbwpresets.txt` (auto-recreated from master)

## 📝 Format

Each preset line follows this format:
```
preset_name<TAB>weight1,weight2,weight3,...,weight26
```

Example:
```
CUSTOM_PRESET	0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0,0,0,0,0
```

---

*Part of **💜 Violet Model Merge** — A sophisticated, artist-friendly model merging toolkit*