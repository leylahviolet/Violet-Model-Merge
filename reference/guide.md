# Introduction

### The goal of the article to explain how I make model merges and how you can too! My intentions are to make this guide understandable to both noob and pro users

# Setting up / tools

1. Chattiori Model Merger [ [https://github.com/Faildes/Chattiori-Model-Merger](https://github.com/Faildes/Chattiori-Model-Merger) ]

    Git Clone to directory of your choice.

    create a 'models' and 'vae' folder anywhere you like.  

2. Download models and SDXL_vae and put them in the appropriate folders

# How to Use Chattiori Model Merger

### First things first, lets discuss the different types of merges and when you should use them

_NOTE: yes chatGPT summarized this. I can confirm it is 99% accurate._

## 1️⃣ **WS - Weighted Sum (2 Models)**

### 🛠️ How It Works

- Directly blends **two models** at a specified alpha.

### ✅ When to Use

- When you want a **simple, balanced mix**.

- Ideal for merging **two models of similar style or purpose**.

- Great for combining **two complementary styles (like realism and anime)**.

### ⚠️ Why Use It

- **Fast, simple, predictable.**

- Best when you want **even mixing of features** without overcomplication.

---

## 2️⃣ **SIG - Sigmoid Merge (2 Models)**

### 🛠️ How It Works

- Uses a **sigmoid curve function** to blend models.

- It **favors details from the stronger model** more than Weighted Sum.

### ✅ When to Use

- If you want to preserve more detail from **one dominant model**, but still have a **touch** of the other model.

- Great for preserving **fine textures or shading from one source**.

### ⚠️ Why Use It

- Provides a more **organic blend**.

- Useful when one model is **higher quality or more detailed**, and you don’t want to lose that.

---

## 3️⃣ **GEO - Geometric Merge (2 Models)**

### 🛠️ How It Works

- Multiplies the latent spaces of the models, which often **smooths out artifacts** and enhances compatibility.

- Often results in a **cleaner but less creative blend**.

### ✅ When to Use

- When you want **more structural stability**.

- Useful if you’re merging **two very different models**, such as realism + anime.

### ⚠️ Why Use It

- Reduces **glitches and noise** from mismatched styles.

- Keeps **core structure stable**, even if styles differ.

---

## 4️⃣ **MAX - Maximum Merge (2 Models)**

### 🛠️ How It Works

- Picks the **maximum value from each model's weights**.

- This strongly favors **the stronger model** in each part of the latent space.

### ✅ When to Use

- When you want to **preserve maximum detail from both models**.

- Can be **unstable**, but great for maximizing detail.

### ⚠️ Why Use It

- Best when working with **high-detail models**.

- Useful for **testing upper limits** of merges.

---

## 5️⃣ **AD - Add Difference (3 Models)**

### 🛠️ How It Works

- Uses **Model 3** as a "difference model" to guide the blend between **Model 1** and **Model 2**.

- This lets you **add features from Model 3 without fully blending it**.

### ✅ When to Use

- When you have a **base blend (Model 1 + Model 2)** and want to **inject features from Model 3**.

- Example: Combine two anime models, but add **realism shading from Model 3**.

### ⚠️ Why Use It

- Preserves control over the core blend while still adding new features.

- One of the most **flexible triple merges**.

---

## 6️⃣ **sAD - Smooth Add Difference (3 Models)**

### 🛠️ How It Works

- Similar to **AD**, but with smoother interpolation between models.

- It softens the impact of Model 3.

### ✅ When to Use

- When Model 3 has **very different features**, and you want to **soften its influence**.

- Best for introducing **stylistic tweaks without overpowering the blend**.

### ⚠️ Why Use It

- Good for subtle refinements in **texture or color**.

---

## 7️⃣ **MD - Multiply Difference (3 Models + Beta)**

### 🛠️ How It Works

- Multiplies the differences between **Model 1 and 2**, then adds them to **Model 3**.

- **Beta** controls how strongly Model 3 influences the final mix.

### ✅ When to Use

- When Model 3 acts as a **refinement base**, and you want to **multiply stylistic differences** into it.

- Example: Apply the stylistic contrast between **anime and realism models** into a **semi-realistic base model**.

### ⚠️ Why Use It

- Very creative but hard to predict.

- Ideal for **experimental merges**.

---

## 8️⃣ **SIM - Similarity Add Difference (3 Models + Beta)**

### 🛠️ How It Works

- Adds the difference between **Model 1 and 2**, but favors **similar features** between them.

- **Beta** controls how much Model 3 influences the mix.

### ✅ When to Use

- If you want to keep **similar structure across all models**, while using Model 3 to add **new elements**.

- Best for combining **closely related models (same base but different styles)**.

### ⚠️ Why Use It

- Helps avoid **artifacting when combining models from the same family**.

---

## 9️⃣ **TD - Train Difference (3 Models)**

### 🛠️ How It Works

- Uses the **difference between Model 1 and 2 as training data**, then applies it to **Model 3**.

- It’s more about **transferring traits than direct blending**.

### ✅ When to Use

- For transferring **aesthetic styles** from one pair to another model.

- Example: Transfer **anime lighting** from one anime pair into a **realism model**.

### ⚠️ Why Use It

- Very powerful for **style transfer**.

- Can get unstable if models differ too much.

---

## 🔟 **TRS - Triple Sum (3 Models + Beta)**

### 🛠️ How It Works

- Averages 3 models, with **beta controlling how much Model 3 contributes**.

- Simple 3-way average.

### ✅ When to Use

- When you want a **balanced, even blend** between all 3 models.

- Good for combining **three models of equal quality**.

### ⚠️ Why Use It

- Simple and reliable if you want an **even mix of all sources**.

---

## 1️⃣1️⃣ **TS - Tensor Sum (2 Models + Beta)**

### 🛠️ How It Works

- Similar to Weighted Sum, but uses tensor-based averaging.

- **Beta** controls how much Model 2 contributes.

### ✅ When to Use

- If you want a more **mathematically stable version** of WS.

- Great for **preserving structure**.

### ⚠️ Why Use It

- Rarely used unless you’re trying to **fine-tune precision merges**.

---

## 1️⃣2️⃣ **ST - Sum Twice (3 Models + Beta)**

### 🛠️ How It Works

- Blends Model 1 and 2, then **blends that result with Model 3**.

- **Beta** controls how strong Model 3 is.

### ✅ When to Use

- When Model 3 is a **refinement or enhancement model**, and you want it added **after the base merge**.

- Example: Combine two toon models, then add a refinement model like **RealVisXL**.

### ⚠️ Why Use It

- Sequential refinement control.

---

## 1️⃣3️⃣ **NoIn - No Interpolation (2 Models)**

### 🛠️ How It Works

- Just takes Model 1 — no merge actually happens.

### ✅ When to Use

- To output Model 1 **as-is** with some metadata tweaking.

- Debugging tool.

---

## 1️⃣4️⃣ **RM - Read Metadata**

### ✅ When to Use

- Reads **merge data from metadata of a model**.

- Rarely used directly.

---

## 1️⃣5️⃣ **DARE - DARE Merge (2 Models)**

### 🛠️ How It Works

- Applies **advanced blending techniques** combining 2 models.

- Designed for **experimental blending**.

### ✅ When to Use

- When you want a **smooth out the final step of a merge**.

- Can produce unexpected but unique results.

---

# 📝 Final Tip

✅ **For stability and quality:** Use **WS**, **GEO**, or **SIG**.  
✅ **For artistic fusion:** Use **AD**, **sAD**, or **TD**.  
✅ **For experimental creativity:** Use **MD**, **DARE**, or **SIM**.

# How do I make my first merge, now?

you need to now write a small script to make the merge.

Using terminal or CMD navigate to the Chattiori-Model-Merge folder  
  
Here is the first part of the script. Feel free to modify this.

This is a basic Wieghted Sum Merge, it uses only two models

```
python lib/merge.py "WS" "/path/to-your/models/" "modelName.safetensors" "ModelName2.safetensors" \
```

## but were not done yet, we need to add arguements

```
python lib/merge.py "WS" "/path/to-your/models/" "modelName.safetensors" "modelName2.safetensors" \
--alpha 0.4 \
--vae "/path/to/vae/sdxl_vae.safetensors" \
--prune --save_half --save_safetensors \
--output "Part A"
```

Whats going on here?

"WS" - merge type. You always define the merge type by its initials, refer to the explanation above of merge types for the different initials.

--Alpha: this controls ratio of model0 and model1  
0.3 = 70/30 ratio, meaning 70% of model0 and 30% of model1  
0.7 = 30/70 ratio meaning 30% of model0 and 70% of model1  
  
--Vae: this is the path to SDXL_vae.safetensors  
  
--prune: strips the models of their original vae and other crap. Always use this.  
  
--save_half: saves the model as FP16 which is best and most used.  
  
--save_safetensors: saves output as .safetensors  
  
--output: name of merged file

# Advanced Merge Arguments

Some merge types require model0, model1 and model2 (meaning three models)

some merges also will requires both an ALPHA and BETA.

I have marked merge types in the above list with how many models are required and if alpha and beta are required.

## How do use beta and a third model?

Example: You want to do an Smooth Add Difference merge

```
python lib/merge.py "sAD" "/path/to-your/models/" "PartA.safetensors" "PartB.safetensors" \
--model_2 "partC.safetensors" \
--alpha 0.55 \
--beta 0.35 \
--vae "/path/to-your/vae/sdxl_vae.safetensors" \
--prune --save_half --save_safetensors \
--output "finalMerge"
```

--Beta: is the ratio between (model0 + model1) & Model 2  
example: --beta 0.35 = 65% (model0 + model1) & 35% Model2

--Model_2: this is how you define the third model (only usable on merges that require 3 models)

## More Optional Arguements

- Optional: `--rand_alpha` randomizes weight put on the second model, if omitted  
    Need to be written in str like `"MIN, MAX, SEED"`.  
    If SEED is not setted, it will be completely random (generates seed).  
    Or `"MIN, MAX, SEED, [Elemental merge args]"` if you want to specify.  
    Check out [Elemental Random](https://github.com/Faildes/merge-models/blob/main/elemental_random.md) for Elemental merge args.

- Optional: `--beta` controls how much weight is put on the third model. Defaults to 0, if omitted  
    Can be written in float value, Merge Block Weight type writing and Elemental Merge type writing.

- Optional: `--rand_beta` randomizes weight put on the third model, if omitted  
    Need to be written in str like `"MIN, MAX, SEED"`.  
    If SEED is not setted, it will be completely random (generates seed).  
    Or `"MIN, MAX, SEED, [Elemental merge args]"` if you want to specify.  
    Check out [Elemental Random](https://github.com/Faildes/merge-models/blob/main/elemental_random.md) for Elemental merge args.

- Optional: `--vae` sets the vae file by set the path, if omitted.  
    If not, the vae stored inside the model will automatically discarded.

- Optional: `--m0_name` determines the name that to write in the data for the model0, if omitted

- Optional: `--m1_name` determines the name that to write in the data for the model1, if omitted

- Optional: `--m2_name` determines the name that to write in the data for the model2, if omitted

- Optional: `--cosine0` determines to favor model0's structure with details from 1, if omitted  
    Check out [Calcmode](https://github.com/hako-mikan/sd-webui-supermerger/blob/main/calcmode_en.md) by hako-mikan for the information.

- Optional: `--cosine1` determines to favor model1's structure with details from 0, if omitted  
    Check out [Calcmode](https://github.com/hako-mikan/sd-webui-supermerger/blob/main/calcmode_en.md) by hako-mikan for the information.

- Optional: `--use_dif_10` determines to use the difference between model0 and model1 as model1, if omitted

- Optional: `--use_dif_20` determines to use the difference between model0 and model2 as model2, if omitted

- Optional: `--use_dif_21` determines to use the difference between model2 and model1 as model2, if omitted

- Optional: `--fine` determines adjustment of details, if omitted  
    Check out [Elemental EN](https://github.com/hako-mikan/sd-webui-supermerger/blob/main/elemental_en.md#adjust) by hako-mikan for the information.

- Optional: `--functn` determines whether add merge function names, if omitted

- Optional: `--delete_source` determines whether to delete the source checkpoint files, not vae file, if omitted

- Optional: `--no_metadata` saves the checkpoint without metadata, if omitted
