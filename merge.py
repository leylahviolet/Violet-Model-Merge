from __future__ import annotations
import os
import numpy as np
import json
import copy
import argparse
import torch
import torch.nn.functional as F
import scipy.ndimage
import safetensors.torch
import safetensors
from tqdm.auto import tqdm

from Utils import wgt, rand_ratio, sha256, read_metadata_from_safetensors \
    , load_model, parse_ratio, qdtyper, maybe_to_qdtype, np_trim_percentiles \
    , diff_inplace, clone_dict_tensors, fineman, weighttoxl, BLOCKID, BLOCKIDFLUX \
    , BLOCKIDXLL, blockfromkey, checkpoint_dict_skip_on_merge, FINETUNES, elementals \
    , to_half, to_half_k, prune_model, cache, merge_cache_json, detect_arch

# Mode Functions

def weight_max(theta0, theta1, *args):
    return torch.max(theta0, theta1)

def geometric(theta0, theta1, alpha):
    return torch.pow(theta0, 1 - alpha) * torch.pow(theta1, alpha)

def sigmoid(theta0, theta1, alpha):
    return (1 / (1 + torch.exp(-4 * alpha))) * (theta0 + theta1) - (1 / (1 + torch.exp(-alpha))) * theta0

def weighted_sum(theta0, theta1, alpha):
    return (1 - alpha) * theta0 + alpha * theta1

def sum_twice(theta0, theta1, theta2, alpha, beta):
    return (1 - beta) * ((1 - alpha) * theta0 + alpha * theta1) + beta * theta2

def triple_sum(theta0, theta1, theta2, alpha, beta):
    return (1 - alpha - beta) * theta0 + alpha * theta1 + beta * theta2

def get_difference(theta1, theta2):
    return theta1 - theta2

def add_difference(theta0, theta1_2_diff, alpha):
    return theta0 + (alpha * theta1_2_diff)

def multiply_difference(theta0, theta1, theta2, alpha, beta):
    theta0_float, theta1_float = theta0.float(), theta1.float()
    diff = (theta0_float - theta2).abs().pow(1 - alpha) * (theta1_float - theta2).abs().pow(alpha)
    sign = weighted_sum(theta0, theta1, beta) - theta2
    return theta2 + torch.copysign(diff, sign).to(theta2.dtype)

def similarity_add_difference(a, b, c, alpha, beta):
    threshold = torch.maximum(a.abs(), b.abs())
    similarity = torch.nan_to_num(((a * b)/(threshold ** 2) + 1) * beta / 2, nan = beta)
    ab_diff = a + alpha * (b - c)
    ab_sum = a * (1 - alpha / 2) + b * (alpha / 2)
    return torch.lerp(ab_diff, ab_sum, similarity)

def dare_merge(theta0, theta1, alpha, beta):
    if theta0.dim() in (1, 2):
        dw = theta1.shape[-1] - theta0.shape[-1]
        if dw > 0:
            theta0 = F.pad(theta0, (0, dw, 0, 0))
        elif dw < 0: 
            theta1 = F.pad(theta1, (0, -dw, 0, 0))
        dh = theta1.shape[0] - theta0.shape[0]
        if dh > 0:
            theta0 = F.pad(theta0, (0, 0, 0, dh))
        elif dh < 0:
            theta1 = F.pad(theta1, (0, 0, 0, -dh))
    delta = theta1 - theta0
    m = torch.bernoulli(torch.full(delta.shape, float(beta), dtype=torch.float32, device=theta0.device))
    denom = max(1.0 - float(beta), 1e-6)
    delta_hat = (m * delta) / denom
    return theta0 + alpha * delta_hat.to(theta0.dtype)

def ortho_merge(a, b, alpha):
    a32 = a.detach().float().view(-1)
    d32 = (b.detach().float() - a.detach().float()).view(-1)
    proj = (torch.dot(d32, a32) / (a32.norm()**2 + 1e-12)) * a32
    d_ortho = (d32 - proj).view_as(a)
    return (a + alpha * d_ortho.to(a.dtype)).to(a.dtype)

def sparse_topk(a, b, alpha, beta):
    d = (b.detach().float() - a.detach().float()).abs().view(-1)
    if d.numel() == 0:
        return a
    k = max(int(d.numel() * float(beta)), 1)
    thresh = d.kthvalue(d.numel() - k).values
    mask = ((b.detach().float() - a.detach().float()).abs() >= thresh).to(a.dtype)
    return (a + alpha * (b - a) * mask).to(a.dtype)

def norm_dir_blend(a, b, alpha):
    a32 = a.detach().float().view(-1); b32 = b.detach().float().view(-1)
    an = a32.norm() + 1e-12; bn = b32.norm() + 1e-12
    au = a32 / an; bu = b32 / bn
    du = F.normalize((1 - alpha) * au + alpha * bu, dim=0)
    mag = (1 - alpha) * an + alpha * bn
    out = (du * mag).view_as(a)
    return out.to(a.dtype)

def channel_cosine_gate(a, b, alpha, beta):
    if a.dim() == 4:
        axis = (1,2,3)
    elif a.dim() == 2:
        axis = (1,)
    else:
        return (1 - alpha) * a + alpha * b

    a32 = a.detach().float()
    b32 = b.detach().float()
    num = (a32 * b32).sum(dim=axis)
    den = (a32.norm(dim=axis) * b32.norm(dim=axis) + 1e-12)
    cos = (num / den).clamp_(-1, 1)
    g = (1 - cos) * float(beta)
    while g.dim() < a.dim():
        g = g.unsqueeze(-1)
    mix = (1 - alpha) * a + alpha * b
    return (a * (1 - g) + mix * g).to(a.dtype)

def freq_band_blend(a, b, alpha, beta):
    if a.dim() != 4 or a.shape[-1] < 3 or a.shape[-2] < 3:
        return (1 - alpha) * a + alpha * b
    a32 = a.detach().float(); b32 = b.detach().float()
    A = torch.fft.rfft2(a32, norm="ortho")
    B = torch.fft.rfft2(b32, norm="ortho")
    H, W = a32.shape[-2], a32.shape[-1]
    cut = max(int(min(H, W) * float(beta)), 1)
    yy = torch.arange(A.shape[-2], device=a.device).view(-1,1).float()
    xx = torch.arange(A.shape[-1], device=a.device).view(1,-1).float()
    cy = (A.shape[-2]-1)/2; cx = (A.shape[-1]-1)/2
    dist = torch.sqrt((yy-cy)**2 + (xx-cx)**2)
    low = (dist <= cut).to(A.dtype)
    high = 1 - low
    M = low * ((1 - alpha) + alpha) + high * ((1 - alpha) + alpha)
    Aout = low * ((1 - alpha) * A + alpha * A) + high * ((1 - alpha) * A + alpha * B)
    Bout = low * ((1 - alpha) * B + alpha * A) + high * ((1 - alpha) * B + alpha * B)
    F = low * Aout + high * Bout
    out = torch.fft.irfft2(F, s=(H, W), norm="ortho")
    return out.to(a.dtype)

# Mode name assignment

theta_funcs = {
    "WS":   (None,           weighted_sum,               "Weighted Sum"),
    "AD":   (get_difference, add_difference,             "Add Difference"),
    "RM":   (None,           None,                       "Read Metedata"),
    "sAD":  (get_difference, add_difference,             "Smooth Add Difference"),
    "MD":   (None,           multiply_difference,        "Multiply Difference"),
    "SIM":  (None,           similarity_add_difference,  "Similarity Add Difference"),
    "TD":   (None,           add_difference,             "Training Difference"),
    "TS":   (None,           weighted_sum,               "Tensor Sum"),
    "TRS":  (None,           triple_sum,                 "Triple Sum"),
    "ST":   (None,           sum_twice,                  "Sum Twice"),
    "NoIn": (None,           None,                       "No Interpolation"),
    "SIG":  (None,           sigmoid,                    "Sigmoid"),
    "GEO":  (None,           geometric,                  "Geometric"),
    "MAX":  (None,           weight_max,                 "Max"),
    "DARE": (None,           dare_merge,                 "DARE"),
    "ORTHO":(None,           ortho_merge,                "Orthogonalized Delta"),
    "SPRSE":(None,           sparse_topk,                "Sparse Top-k Delta"),
    "NORM": (None,           norm_dir_blend,             "Norm/Direction Split"),
    "CHAN": (None,           channel_cosine_gate,        "Channel-wise Cosine Gate"),
    "FREQ": (None,           freq_band_blend,            "Frequency-Band Blend"),
}
modes_need_m2   = {"sAD", "AD", "TRS", "ST",  "TD", "SIM", "MD", "SPRSE", "HUB", "CHAN", "FREQ"}
modes_need_beta = {"TRS", "ST", "TS",  "SIM", "MD", "DARE"}

parser = argparse.ArgumentParser(description="Merge two or three models")

parser.add_argument("mode",         choices=list(theta_funcs.keys()),   help="Merging mode")
parser.add_argument("model_path",   type=str,                           help="Path to models")
parser.add_argument("model_0",      type=str,                           help="Name of model 0")
parser.add_argument("model_1",      type=str,                           help="Optional, Name of model 1", default=None)
parser.add_argument(f"--model_2",   type=str,                           help="Optional, Name of model 2", default=None, required=False)

for i in range(3):
    parser.add_argument(f"--m{i}_name", type=str, help=f"Custom name of model {i}", default=None, required=False)

for dif in ["10","20","21"]:
    parser.add_argument(f"--use_dif_{dif}", action="store_true", help=f"Use the difference of model {dif[0]} and model {dif[1]} as model {max(int(dif[0]), int(dif[1]))}", required=False)

for p in ["alpha","beta"]:
    parser.add_argument(f"--{p}", default=0.0, help=f"{p.capitalize()} value, optional, defaults to 0", required=False)
    parser.add_argument(f"--rand_{p}", type=str, help=f"Random {p.capitalize()} value, optional", default=None, required=False)

for flag, helpmsg in {
    "cosine0":          "Favor model 0's structure with details from the others (two/three models)",
    "cosine1":          "Favor model 1's structure with details from the others (two/three models)",
    "cosine2":          "Favor model 2's structure with details from the others (three models only)",
    "save_half":        "Save as float16",
    "save_quarter":     "Save as float8",
    "save_safetensors": "Save as .safetensors",
    "keep_ema":         "Keep ema",
    "delete_source":    "Delete the source checkpoint file",
    "no_metadata":      "Save without metadata",
    "prune":            "Prune Model",
    "force":            "Overwrite output if exists",
}.items():
    parser.add_argument(f"--{flag}", action="store_true", help=helpmsg, required=False)

parser.add_argument("--seed",   type=int,   default=None, help="Random seed for stochastic modes (e.g., DARE)")
parser.add_argument("--vae",    type=str,   help="Path of VAE", default=None, required=False)
parser.add_argument("--fine",   type=str,   help="Finetune the given keys on model 0", default=None, required=False)
parser.add_argument("--output",             help="Output file name without extension", default="merged", required=False)
parser.add_argument("--device", type=str,   help="Device to use, defaults to cpu", default="cpu", required=False)

args = parser.parse_args()
device = args.device
mode = args.mode
if mode in modes_need_m2 and (args.model_2 is None):
    raise SystemExit(f"mode '{mode}' needs 3rd model")
theta_func1, theta_func2, merge_name = theta_funcs[mode]

args.alpha, deep_a, block_a = wgt(args.alpha, [])
args.beta,  deep_b, block_b = wgt(args.beta, [])
useblocks = block_a or block_b

cos_flags = [args.cosine0, args.cosine1, args.cosine2]
if sum(1 for f in cos_flags if f) > 1:
    raise SystemExit("cosine0, cosine1 and cosine2 cannot be posed at same time, choose one only")

if args.cosine2 and (args.model_2 is None):
    raise SystemExit("--cosine2 cannot be used when there are only 2 models given")

if mode == "WS" and (args.cosine0 ^ args.cosine1):
    cosine0, cosine1 = args.cosine0, args.cosine1
else:
    cosine0 = cosine1 = False
output_name = args.output
output_file = f"{output_name}.{'safetensors' if args.save_safetensors else 'ckpt'}"
output_path = os.path.join(args.model_path, output_file)
merge_cache_json(args.model_path)
cache_data = cache("hashes", None)

i = 0
while os.path.isfile(output_path):
    if args.force:
        os.remove(output_path); break
    output_name = f"{args.output}_{i:02}"
    output_file = f"{output_name}.{'safetensors' if args.save_safetensors else 'ckpt'}"
    output_path = os.path.join(args.model_path, output_file)
    i += 1
    if not os.path.isfile(output_path): print(f"Assigned result checkpoint name as {output_file}\n")

stem = lambda p: os.path.splitext(os.path.basename(p))[0]

torch.set_grad_enabled(False)

alpha_seed = beta_seed = None
alpha_info = beta_info = ""
if args.rand_alpha is not None:
    args.alpha, alpha_seed, deep_a, alpha_info = rand_ratio(args.rand_alpha)
if args.rand_beta is not None:
    args.beta, beta_seed,  deep_b,  beta_info  = rand_ratio(args.rand_beta)

model_0_path = os.path.join(args.model_path, args.model_0)
if mode == "RM":
    print(sha256(model_0_path, f"checkpoint/{stem(model_0_path)}"))
    meta = read_metadata_from_safetensors(model_0_path)
    print(json.dumps(meta, indent=2))
    with open(f"./{output_name}.json", "a+", encoding="utf-8") as dmp:
        json.dump(meta, dmp, indent=4)
    exit()
    
interp_method = 2
model_0_name = args.m0_name or stem(model_0_path)
print(f"Loading {model_0_name}...")
theta_0, model_0_sha256, model_0_hash, model_0_meta, cache_data = load_model(model_0_path, device, cache_data=cache_data)
qd0 = qdtyper(theta_0)

theta_1 = theta_2 = None
model_1_sha256 = model_2_sha256 = None

if mode != "NoIn":
    interp_method = 0
    model_1_path = os.path.join(args.model_path, args.model_1)
    model_1_name = args.m1_name or stem(model_1_path)
    print(f"Loading {model_1_name}...")
    theta_1, model_1_sha256, model_1_hash, model_1_meta, cache_data = load_model(model_1_path, device, cache_data=cache_data)
    qd1 = qdtyper(theta_1)
    isxl, isflux = detect_arch(theta_1)

    weights_a, alpha, alpha_info = parse_ratio(args.alpha, alpha_info, deep_a)
    if mode in modes_need_m2:
        model_2_path = os.path.join(args.model_path, args.model_2)
        model_2_name = args.m2_name or stem(model_2_path)
        print(f"Loading {model_2_name}...")
        theta_2, model_2_sha256, model_2_hash, model_2_meta, cache_data = load_model(model_2_path, device, cache_data=cache_data)
        qd2 = qdtyper(theta_2)

    usebeta = mode in modes_need_beta
    if usebeta:
        weights_b, beta, beta_info = parse_ratio(args.beta, beta_info, deep_b)
    else:
        weights_b, beta = None, None
else:
    usebeta = False
    weights_a = weights_b = None
    alpha = beta = None
    isxl, isflux = False, False

if args.vae:
    vae_name = stem(args.vae)
    vae, *_ = load_model(args.vae, device, verify_hash=False)

if mode == "DARE":
    g = torch.Generator(device=device if device != "cpu" else "cpu")
    if args.seed is not None: g.manual_seed(args.seed)

def cosine_minmax(base_dict, other_dict, desc, variant=0):
    vals = []
    for k in tqdm(base_dict.keys(), desc=desc):
        if "first_stage_model" in k or "model" not in k or k not in other_dict:
            continue
        a = base_dict[k].detach().float().view(-1)
        b = other_dict[k].detach().float().view(-1)
        if a.numel() == 0 or b.numel() == 0 or a.shape != b.shape:
            continue
        simab = F.cosine_similarity(a, b, dim=0).item()
        if variant == 0:
            vals.append(simab)
        else:
            dot = torch.dot(a, b).item()
            denom = float(a.norm().item() * b.norm().item())
            mag = (dot / denom) if denom != 0.0 else 0.0
            vals.append(0.5 * (simab + mag))
    arr = np_trim_percentiles(np.asarray(vals, dtype=np.float64))
    if arr.size == 0 or not np.isfinite(arr).any():
        return 0.0, 1.0
    return float(np.nanmin(arr)), float(np.nanmax(arr))

if theta_func1:
    if isflux:
        theta_1, theta_2 = maybe_to_qdtype(theta_1, theta_2, qd1, qd2, device)
    diff_inplace(theta_1, theta_2, theta_func1, "Getting Difference of Model 1 and 2")
    del theta_2

if isflux:
    theta_0, theta_1 = maybe_to_qdtype(theta_0, theta_1, qd0, qd1, device)
    if 'theta_2' in locals() and theta_2 is not None:
        theta_0, theta_2 = maybe_to_qdtype(theta_0, theta_2, qd0, qd2, device)

if mode == "TS":
    theta_0 = clone_dict_tensors(theta_0)
    
if args.use_dif_21:
    theta_3 = copy.deepcopy(theta_1)
    diff_inplace(theta_2, theta_3, get_difference, "Getting Difference of Model 1 and 2")
    del theta_3

if args.use_dif_10:
    theta_3 = copy.deepcopy(theta_0)
    diff_inplace(theta_1, theta_3, get_difference, "Getting Difference of Model 0 and 1")
    del theta_3

if args.use_dif_20:
    theta_3 = copy.deepcopy(theta_0)
    diff_inplace(theta_2, theta_3, get_difference, "Getting Difference of Model 0 and 2")
    del theta_3

def resolve_cosine_triplet(theta_0, theta_1, theta_2, use_cos0, use_cos1, use_cos2):
    if use_cos0:
        base, dA, dB = theta_0, theta_1, theta_2
        varA, varB = 0, 0
    elif use_cos1:
        base, dA, dB = theta_1, theta_0, theta_2
        varA, varB = 1, 0
    else:
        base, dA, dB = theta_2, theta_0, theta_1
        varA, varB = 0, 0
    return base, dA, dB, varA, varB

if mode != "NoIn":
    if args.fine:
        fine = fineman([float(t) for t in args.fine.split(",")], isxl)
    else:
        fine = ""

    if isxl and useblocks:
        if len(weights_a) == 25:
            weights_a = weighttoxl(weights_a)
            print(f"alpha weight converted for XL{weights_a}")
        elif len(weights_a) == 19:
            weights_a += [0]
        if mode != "DARE" and usebeta:
            if len(weights_b) == 25:
                weights_b = weighttoxl(weights_b)
                print(f"beta weight converted for XL{weights_b}")
            elif len(weights_b) == 19:
                weights_b += [0]
        
def _resolve_weight_index(key):
    block, tag = blockfromkey(key, isxl, isflux)
    if block == "Not Merge":
        return -1
    if isflux and tag in BLOCKIDFLUX: return BLOCKIDFLUX.index(tag)
    if isxl   and tag in BLOCKIDXLL:  return BLOCKIDXLL.index(tag)
    if tag in BLOCKID:                return BLOCKID.index(tag)
    return -1

def _apply_cosine_blend(a, b, kmin, kmax, cur_alpha, variant):
    a_f = a.detach().float()
    b_f = b.detach().float()
    simab = F.cosine_similarity(a_f.view(-1), b_f.view(-1), dim=0)
    if variant == 1:
        dot = torch.dot(a_f.view(-1), b_f.view(-1))
        denom = (a_f.norm() * b_f.norm())
        mag = dot / denom if denom != 0 else torch.tensor(0., device=a.device)
        simab = 0.5 * (simab + mag)
    k = ((simab - kmin) / max(kmax - kmin, 1e-6) - abs(float(cur_alpha))).clamp_(0, 1)
    return b * (1 - k) + a * k

def _finetune_inplace(key, tens, fine):
    if not fine:
        return tens

    if isinstance(fine, dict) and isflux:
        mul = fine.get("mul", {})
        m = 1.0

        if any(s in key for s in ("double_block", "double_blocks", "db.")):
            m *= mul.get("double_block", 1.0)
        if any(s in key for s in (".img_in", "image_in", "image_proj")):
            m *= mul.get("img_in", 1.0)
        if any(s in key for s in (".txt_in", "context_in", "text_in", "clip_proj")):
            m *= mul.get("txt_in", 1.0)
        if any(s in key for s in ("time_in", "time_embed", "timestep", "vector_in")):
            m *= mul.get("time", 1.0)
        if any(s in key for s in (".out.", "final_layer", "vector_out")) or key.endswith(".out"):
            m *= mul.get("out", 1.0)

        if m != 1.0:
            tens = tens * torch.as_tensor(m, device=tens.device, dtype=tens.dtype)

        add = float(fine.get("add", 0.0) or 0.0)
        if add and (key.endswith(".bias") or ".bias" in key):
            tens = tens + torch.as_tensor(add, device=tens.device, dtype=tens.dtype)

        return tens

    if isinstance(fine, list):
        idx = next((i for i, pat in enumerate(FINETUNES) if pat in key), -1)
        if idx == -1:
            return tens

        if idx < 5:
            return tens * torch.as_tensor(fine[idx], device=tens.device, dtype=tens.dtype)
        else:
            f5 = fine[5]
            if isinstance(f5, (list, tuple)) and len(f5) > 0:
                bias = float(f5[0])
            elif isinstance(f5, (int, float)):
                bias = float(f5)
            else:
                bias = 0.0
            if bias:
                return tens + torch.as_tensor(bias, device=tens.device, dtype=tens.dtype)
            return tens

    return tens

use_cos0 = bool(args.cosine0)
use_cos1 = bool(args.cosine1)
use_cos2 = bool(args.cosine2)

if use_cos0 or use_cos1 or use_cos2:
    base, dA, dB, varA, varB = resolve_cosine_triplet(theta_0, theta_1, theta_2, use_cos0, use_cos1, use_cos2)

    kminA, kmaxA = cosine_minmax(base, dA, "Cosine(base vs A)", variant=varA)
    if dB is not None:
        kminB, kmaxB = cosine_minmax(base, dB, "Cosine(base vs B)", variant=varB)
    else:
        kminB = kmaxB = None
        
    theta_res = clone_dict_tensors(base)

    for key in tqdm(base.keys(), desc="Cosine structure-based blending..."):
        if "first_stage_model" in key or "model" not in key:
            continue
        if key not in dA:
            continue

        wi = _resolve_weight_index(key)
        if wi < 0:
            continue

        cur_a, cur_b = alpha, beta
        if wi > 0:
            if weights_a is not None:            cur_a = weights_a[wi - 1]
            if (weights_b is not None) and dB is not None: cur_b = weights_b[wi - 1]
        if deep_a: cur_a = elementals(key, wi, deep_a, cur_a)
        if deep_b and dB is not None: cur_b = elementals(key, wi, deep_b, cur_b)

        a = base[key]
        b = dA[key]

        out = _apply_cosine_blend(a, b, kminA, kmaxA, cur_a, variant=varA)

        if dB is not None and (key in dB) and (cur_b is not None):
            out = _apply_cosine_blend(out, dB[key], kminB, kmaxB, cur_b, variant=varB)

        out = _finetune_inplace(key, out)
        theta_res[key] = out

    theta_0 = theta_res

if mode != "NoIn":
    for key in tqdm(theta_0.keys(), desc=f"{merge_name} Merging..."):
        if args.vae is None and "first_stage_model" in key:
            continue
        if not (theta_1 and "model" in key and key in theta_1):
            continue
        if mode != "DARE" and (usebeta or mode == "TD") and (theta_2 is not None) and key not in theta_2:
            continue
        if key in checkpoint_dict_skip_on_merge:
            continue

        a, b = theta_0[key], theta_1[key]
        al, bl = list(a.shape), list(b.shape)

        wi = _resolve_weight_index(key)
        if wi < 0:
            continue

        cur_a, cur_b = alpha, beta
        if wi > 0:
            if weights_a is not None:       cur_a = weights_a[wi - 1]
            if usebeta and weights_b is not None: cur_b = weights_b[wi - 1]
        if deep_a: cur_a = elementals(key, wi, deep_a, cur_a)
        if deep_b: cur_b = elementals(key, wi, deep_b, cur_b)

        if mode == "sAD":
            bf = b.detach().float()
            filt = scipy.ndimage.gaussian_filter(bf.cpu().numpy(), sigma=1)
            theta_0[key] = a + cur_a * torch.from_numpy(filt).to(a.device, dtype=a.dtype)
            theta_0[key] = _finetune_inplace(key, theta_0[key]); continue

        if mode == "TD":
            t1 = theta_1[key].float()
            t2 = theta_2[key].float()
            t0 = theta_0[key].float()
            if torch.allclose(t1, t2, rtol=0, atol=0):
                theta_2[key] = theta_0[key]
                continue
            diff_AB = (t1 - t2).abs()
            dist_A0 = (t1 - t0).abs()
            dist_A2 = (t1 - t2).abs()
            denom = dist_A0 + dist_A2
            scale = torch.where(denom != 0, dist_A0 / denom, torch.tensor(0., device=t0.device))
            scale = torch.sign(t1 - t2) * scale.abs()
            theta_0[key] = (t0 + (scale * diff_AB) * (cur_a * 1.8)).to(theta_0[key].dtype)
            theta_0[key] = _finetune_inplace(key, theta_0[key]); continue

        if mode == "TS":
            if a.dim() == 0:
                continue
            n = a.shape[0]
            if cur_a + cur_b <= 1:
                s, e = int(n * cur_b), int(n * (cur_a + cur_b))
                theta_0[key][s:e, ...] = b[s:e, ...].clone()
            else:
                s, e = int(n * (cur_a + cur_b - 1)), int(n * cur_b)
                t = b.clone()
                t[s:e, ...] = a[s:e, ...].clone()
                theta_0[key] = t
            theta_0[key] = _finetune_inplace(key, theta_0[key])
            continue

        if al != bl and len(al) == 4 and len(bl) == 4 and al[0]==bl[0] and al[2:]==bl[2:]:
            use = min(al[1], bl[1], 4)
            ad = a[:, :use, ...]
        else:
            ad = a

        if usebeta and mode != "DARE":
            c = theta_2[key]
            theta_0[key] = theta_func2(ad, b, c, cur_a, cur_b)
        elif usebeta:
            theta_0[key] = theta_func2(ad, b, cur_a, cur_b)
        else:
            theta_0[key] = theta_func2(ad, b, cur_a)

        theta_0[key] = _finetune_inplace(key, theta_0[key])

    if mode != "DARE":
        for key in tqdm(theta_1.keys(), desc="Remerging..."):
            if isflux or key in checkpoint_dict_skip_on_merge or "model" not in key or key in theta_0:
                continue
            try:
                if mode in {"TRS", "ST"} and theta_2 is not None and key in theta_2:
                    b, c = theta_1[key], theta_2[key]
                    cur_b = beta
                    wi = _resolve_weight_index(key)
                    if wi >= 0 and weights_b is not None:
                        cur_b = weights_b[wi]
                    if deep_b:
                        cur_b = elementals(key, wi, deep_b, cur_b)
                    theta_0[key] = weighted_sum(b, c, cur_b)
                else:
                    theta_0[key] = theta_1[key]
            except NameError:
                theta_0[key] = theta_1[key]

    del theta_1
    try:
        if theta_2:
            for key in tqdm(theta_2.keys(), desc="Remerging..."):
                if key in checkpoint_dict_skip_on_merge:
                    continue
                if "model" in key and key not in theta_0:
                    theta_0[key] = theta_2[key]
            del theta_2
    except NameError:
        pass

else:
    isxl, isflux = detect_arch(theta_0)
    if args.fine:
        fine = fineman([float(t) for t in args.fine.split(",")], isxl, isflux)
        for key in tqdm(theta_0.keys(), desc="Fine Tuning ..."):
            if args.vae is None and "first_stage_model" in key:
                continue
            theta_0[key] = _finetune_inplace(key, theta_0[key], fine)
    else:
        fine = ""
        
if args.vae:
    for k in tqdm(vae.keys(), desc=f"Baking in VAE[{vae_name}] ..."):
        tk = 'first_stage_model.' + k
        if tk in theta_0:
            theta_0[tk] = to_half(vae[k], args.save_half)
    del vae

isxl, isflux = detect_arch(theta_0)

if isxl:
    for k in tqdm([k for k in theta_0.keys() if "cond_stage_model." in k], desc="Cond resolving..."):
        del theta_0[k]

theta_0 = to_half_k(theta_0, args.save_half)
if args.prune:
    theta_0 = prune_model(theta_0, "Model", args, isxl, isflux)

for k in tqdm(list(theta_0.keys()), desc="Check contiguous..."):
    theta_0[k] = theta_0[k].contiguous()

metadata = {"format": "pt", "sd_merge_models": {}, "sd_merge_recipe": None}

calcs = [
    name for flag, name in [
        (cosine0,            "cosine_0"),
        (cosine1,            "cosine_1"),
        (args.use_dif_10,    "use_dif_10"),
        (args.use_dif_20,    "use_dif_20"),
        (args.use_dif_21,    "use_dif_21"),
    ] if flag
]
if args.fine:
    calcs.append(f"fine[{fine}]")
calcl = ",".join(calcs) or None

fp = "fp8" if args.save_quarter else ("fp16" if args.save_half else "fp32")

merge_recipe = {
    "type":                 "merge-models-chattiori",
    "primary_model_hash":   model_0_sha256,
    "secondary_model_hash": model_1_sha256 if mode != "NoIn" else None,
    "tertiary_model_hash":  model_2_sha256 if mode in modes_need_m2 else None,
    "merge_method":         merge_name,
    "block_weights":        (weights_a is not None or weights_b is not None),
    "alpha_info":           alpha_info or None,
    "beta_info":            beta_info  or None,
    "calculation":          calcl,
    "fp":                   fp,
    "output_name":          output_name,
    "bake_in_vae":          (vae_name if args.vae else False),
    "pruned":               args.prune,
}
metadata["sd_merge_recipe"] = json.dumps(merge_recipe)

def add_model_metadata(s256, hashed, meta, model_name):
    metadata["sd_merge_models"][s256] = {
        "name": model_name,
        "legacy_hash": hashed,
        "sd_merge_recipe": meta.get("sd_merge_recipe"),
    }
    metadata["sd_merge_models"].update(meta.get("sd_merge_models", {}))

add_model_metadata(model_0_sha256, model_0_hash, model_0_meta, model_0_name)
if mode != "NoIn":
    add_model_metadata(model_1_sha256, model_1_hash, model_1_meta, model_1_name)
if mode in modes_need_m2:
    add_model_metadata(model_2_sha256, model_2_hash, model_2_meta, model_2_name)

metadata["sd_merge_models"] = json.dumps(metadata["sd_merge_models"])

print(f"Saving as {output_file}...")

if args.delete_source:
    for p, cond in [
        (model_0_path, True),
        (model_1_path, mode != "NoIn"),
        (model_2_path, mode in modes_need_m2),
    ]:
        if cond:
            os.remove(p)

if args.save_safetensors:
    with torch.no_grad():
        safetensors.torch.save_file(
            theta_0, output_path,
            metadata=None if args.no_metadata else metadata
        )
else:
    torch.save({"state_dict": theta_0}, output_path)

del theta_0
print(f"Done! ({round(os.path.getsize(output_path)/1073741824, 2)}G)")