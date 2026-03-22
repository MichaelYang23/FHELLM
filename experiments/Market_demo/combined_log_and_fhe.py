#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import json
import argparse
from copy import deepcopy
from typing import Dict, Any, Tuple, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

import logix
from logix.utils import DataIDGenerator, get_logger

# If InfluenceFunctionFHE is already provided in the library, import it directly.
from logix.analysis import InfluenceFunctionFHE

from utils import (
    set_seed,
    construct_mlp,
    filter_mnist_indices,
    make_subset_loader_4class,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- helpers: tree operations (clone/add/scalar multiply) ----------------
def _tree_clone(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    if isinstance(x, list):
        return [_tree_clone(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_tree_clone(v) for v in x)
    if isinstance(x, dict):
        return {k: _tree_clone(v) for k, v in x.items()}
    return deepcopy(x)

def _tree_add_(dst, src):
    if type(dst) != type(src):
        return
    if isinstance(dst, torch.Tensor) and isinstance(src, torch.Tensor):
        if dst.shape == src.shape and dst.dtype == src.dtype and dst.device == src.device:
            dst.add_(src)
        return
    if isinstance(dst, list) and isinstance(src, list):
        for a, b in zip(dst, src): _tree_add_(a, b); return
    if isinstance(dst, tuple) and isinstance(src, tuple):
        for a, b in zip(dst, src): _tree_add_(a, b); return
    if isinstance(dst, dict) and isinstance(src, dict):
        for k in dst.keys():
            if k in src: _tree_add_(dst[k], src[k]); return

def _tree_mul_scalar_(dst, alpha: float):
    if isinstance(dst, torch.Tensor):
        dst.mul_(alpha); return
    if isinstance(dst, list):
        for x in dst: _tree_mul_scalar_(x, alpha); return
    if isinstance(dst, tuple):
        for x in dst: _tree_mul_scalar_(x, alpha); return
    if isinstance(dst, dict):
        for k in dst.keys(): _tree_mul_scalar_(dst[k], alpha); return


# ---------------- dataset / loader construction ----------------
def load_partitions(partitions_dir: str) -> Dict[str, np.ndarray]:
    """Load seller/evaluation indices."""
    out = {}
    for name in ("sellerA_idx", "sellerB_idx", "sellerC_idx", "buyer_eval_idx"):
        p = os.path.join(partitions_dir, f"{name}.npy")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {name}: {p}")
        arr = np.load(p)
        if arr.size == 0:
            raise RuntimeError(f"{name} is empty.")
        out[name] = arr
    return out

def make_seller_loader_singleclass(ds_tr, tr_idx_all, tr_y_remap, subset_indices, batch, shuffle=False):
    """Training subset loader (unified 4-class remap) for logging sellers' gradients/covariance."""
    return make_subset_loader_4class(
        dataset=ds_tr,
        filtered_indices=tr_idx_all,
        remapped_labels_all=tr_y_remap,
        subset_indices=subset_indices,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )

def build_eval_loader_single_digit(root: str, partitions_dir: str, eval_digit: int, batch: int):
    """
    Filter subset from buyer_eval_idx.npy where "original label == eval_digit",
    and build 4-class remap DataLoader based on test split.
    """
    buyer_eval_idx = np.load(os.path.join(partitions_dir, "buyer_eval_idx.npy"))
    te_idx_all, te_y_remap, ds_te = filter_mnist_indices(root=root, split="test")

    # Build mapping from test global idx -> position
    idx_map = {int(g): i for i, g in enumerate(te_idx_all.tolist())}

    # remap: original {1,2,3,4} -> {0,1,2,3}
    remap = {1:0, 2:1, 3:2, 4:3}
    want_lab = remap[eval_digit]

    keep = []
    for g in buyer_eval_idx.tolist():
        pos = idx_map.get(int(g), None)
        if pos is None: continue
        if int(te_y_remap[pos]) == want_lab:
            keep.append(int(g))
    keep = np.array(keep, dtype=int)
    if keep.size == 0:
        raise RuntimeError(f"Eval subset for digit={eval_digit} is empty.")

    loader = make_subset_loader_4class(
        dataset=ds_te,
        filtered_indices=te_idx_all,
        remapped_labels_all=te_y_remap,
        subset_indices=keep,
        batch_size=batch,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return loader, te_idx_all, te_y_remap, ds_te, keep.size


# ---------------- Main workflow ----------------
def main():
    ap = argparse.ArgumentParser("Combined logging (sellers+KFAC) & one-pass FHE IF (single eval digit)")
    ap.add_argument("--project", type=str, default="mnist_market_4class")
    ap.add_argument("--buyer_ckpt", type=str, default="./checkpoints/buyer/mlp4_seed0_epoch10.pt")
    ap.add_argument("--partitions", type=str, default="./partitions")
    ap.add_argument("--root", type=str, default="./data")
    ap.add_argument("--logs_dir", type=str, default="./logs")

    # Single-digit evaluation (recommend 3)
    ap.add_argument("--eval_digit", type=int, default=3, choices=[1,2,3,4])

    # batch / training logging settings
    ap.add_argument("--seller_batch", type=int, default=512)
    ap.add_argument("--eval_batch", type=int, default=256)
    ap.add_argument("--batch_log", type=int, default=64)   # batch size for building log_loader

    # LoRA & Hessian options
    ap.add_argument("--use_lora", action="store_true", default=True)
    ap.add_argument("--hessian", type=str, default="kfac", choices=["kfac", "none"])
    ap.add_argument("--damping", type=float, default=0.0)

    # Optional greedy selection
    ap.add_argument("--greedy_k", type=int, default=0)
    ap.add_argument("--select_mode", type=str, default="max", choices=["max","min"])  # more positive is better, default max

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.logs_dir, exist_ok=True)

    lg = get_logger()
    lg.info(f"[Eval] original digit={args.eval_digit}")

    # ---------- Load partitions ----------
    parts = load_partitions(args.partitions)
    sellerA_idx = parts["sellerA_idx"]
    sellerB_idx = parts["sellerB_idx"]
    sellerC_idx = parts["sellerC_idx"]

    # ---------- Build datasets ----------
    tr_idx_all, tr_y_remap, ds_tr = filter_mnist_indices(root=args.root, split="train")
    # Three single-class seller loaders (note: still use unified 4-class remap)
    loader_A = make_seller_loader_singleclass(ds_tr, tr_idx_all, tr_y_remap, sellerA_idx, batch=args.seller_batch, shuffle=False)
    loader_B = make_seller_loader_singleclass(ds_tr, tr_idx_all, tr_y_remap, sellerB_idx, batch=args.seller_batch, shuffle=False)
    loader_C = make_seller_loader_singleclass(ds_tr, tr_idx_all, tr_y_remap, sellerC_idx, batch=args.seller_batch, shuffle=False)

    eval_loader, te_idx_all, te_y_remap, ds_te, n_eval = build_eval_loader_single_digit(
        root=args.root, partitions_dir=args.partitions, eval_digit=args.eval_digit, batch=args.eval_batch
    )
    lg.info(f"[Eval] Using original digit={args.eval_digit} | N={n_eval}")

    # ---------- Build model ----------
    model = construct_mlp(num_classes=4, seed=0).to(DEVICE)
    sd = torch.load(args.buyer_ckpt, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    # ---------- Initialize LogIX (single instance) ----------
    run = logix.init(project=args.project)
    run.watch(model)
    if args.use_lora:
        lg.info("Adding LoRA adapters…")
        run.add_lora()
        run.watch(model)  # Watch again after LoRA to ensure parameter topology matches
        lg.info("LoRA added.")

    # Configuration for logging sellers (KFAC or none)
    if args.hessian == "kfac":
        # Log covariance for both forward/backward + gradients
        run.setup({"forward": ["covariance"], "backward": ["covariance"], "grad": ["log"]})
    else:
        run.setup({"grad": ["log"]})
    run.save(True)  # Allow saving

    # ---------- Log three sellers (order A→B→C), accumulate manifest ----------
    manifest = {}
    idg = DataIDGenerator()

    def log_one(name: str, loader: DataLoader) -> int:
        seen = 0
        for xb, yb in loader:
            with run(data_id=idg(xb)):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                model.zero_grad()
                out = model(xb)
                loss = torch.nn.functional.cross_entropy(out, yb, reduction="sum")
                loss.backward()
            seen += xb.size(0)
        return seen

    offset = 0
    nA = log_one("sellerA", loader_A); manifest["sellerA"] = {"size": int(nA), "offset": int(offset)}; offset += nA
    nB = log_one("sellerB", loader_B); manifest["sellerB"] = {"size": int(nB), "offset": int(offset)}; offset += nB
    nC = log_one("sellerC", loader_C); manifest["sellerC"] = {"size": int(nC), "offset": int(offset)}; offset += nC
    manifest["total_logged"] = int(offset)

    # Save catalog indices for later consistency checks/experiment reproduction
    for name, arr in [("sellerA", sellerA_idx), ("sellerB", sellerB_idx), ("sellerC", sellerC_idx)]:
        d = os.path.join(args.logs_dir, name); os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "catalog_indices.npy"), arr)

    # Finalize logs (write out means/covariances etc.)
    run.finalize()
    with open(os.path.join(args.logs_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("\n=== Seller logging complete (single-digit eval) ===")
    print(json.dumps(manifest, indent=2))

    # ---------- Build "PLAINTEXT averaged test gradient tree" within same instance (only for eval_digit) ----------
    run.eval()               # Switch to eval mode
    run.setup({"grad": ["log"]})
    run.save(False)          # No longer generate new training logs

    acc_tree = None
    used = 0
    idg_eval = DataIDGenerator()

    # To strictly match official example interface expectations: test_log should pass a "dict tree"
    # We process per sample, backward once, and average the tree structure from run.get_log()
    for xb, yb in eval_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        for i in range(xb.size(0)):
            x = xb[i:i+1]; y = yb[i:i+1]
            with run(data_id=idg_eval(x)):
                model.zero_grad()
                out = model(x)
                loss = torch.nn.functional.cross_entropy(out, y, reduction="sum")
                loss.backward()
            cur = run.get_log()  # dict-like
            if acc_tree is None:
                acc_tree = _tree_clone(cur)
            else:
                _tree_add_(acc_tree, cur)
            used += 1

    if acc_tree is None or used == 0:
        raise RuntimeError("Eval loader yielded no samples; mean test gradient cannot be built.")
    _tree_mul_scalar_(acc_tree, 1.0 / float(used))
    lg.info(f"Built plaintext mean test gradient on {used} samples.")

    # Optional save (for traceability)
    buyer_dir = os.path.join(args.logs_dir, "buyer"); os.makedirs(buyer_dir, exist_ok=True)
    torch.save({"mean_test_tree": acc_tree, "count": int(used)}, os.path.join(buyer_dir, "mean_test_grad.pt"))

    # ---------- Build log_loader (corresponding to sellers logs) ----------
    # Suggest flatten=False to let analyzer uniformly flatten test/train sides, avoiding "one flattened one not" mismatch.
    log_loader = run.build_log_dataloader(batch_size=args.batch_log, num_workers=0, flatten=True)

    # ---------- FHE IF (one pass) ----------
    run.add_analysis({"influence_fhe": InfluenceFunctionFHE})
    lg.info("Compute FHE IF (ONE pass)…")
    out = run.influence_fhe.compute_influence_all_fhe(
        test_log=acc_tree,                # Pass "tree" directly, not (ids, tree) tuple
        log_loader=log_loader,
        decrypt_results=True,
        hessian=args.hessian,             # "kfac" will use the covariance just written by run.finalize
        damping=args.damping,
    )
    if not isinstance(out["influence"], torch.Tensor):
        raise RuntimeError("FHE results are not decrypted. Set decrypt_results=True.")
    scores = out["influence"].detach().cpu().numpy().reshape(-1)
    assert scores.size == int(manifest["total_logged"])

    # ---------- Save & summarize ----------
    tag = f"eval{args.eval_digit}"
    np.save(os.path.join(args.logs_dir, f"scores_{tag}_fhe.npy"), scores)
    print(f"Saved FHE scores ({tag}) → ./logs/scores_{tag}_fhe.npy")

    # Slice by seller
    def slice_by_seller(vec: np.ndarray, man: Dict[str, Any]) -> Dict[str, np.ndarray]:
        out = {}
        for s in ("sellerA", "sellerB", "sellerC"):
            off = int(man[s]["offset"]); size = int(man[s]["size"])
            out[s] = vec[off:off+size]
            if out[s].shape[0] != size:
                raise RuntimeError(f"[{s}] slice length mismatch.")
        return out

    byseller = slice_by_seller(scores, manifest)
    agg_sum  = {k: float(np.sum(v))  for k, v in byseller.items()}
    agg_mean = {k: float(np.mean(v)) for k, v in byseller.items()}
    print(f"\n[{tag}] IF sum:  " + json.dumps({k: f"{v:+.6f}" for k, v in agg_sum.items()}))
    print(f"[{tag}] IF mean: " + json.dumps({k: f"{v:+.6f}" for k, v in agg_mean.items()}))

    # Optional greedy selection (more positive is better)
    if args.greedy_k and args.greedy_k > 0:
        if args.select_mode == "max":
            idx = np.argpartition(-scores, args.greedy_k)[:args.greedy_k]
        else:
            idx = np.argpartition(scores, args.greedy_k)[:args.greedy_k]
        plan = {
            "target": tag,
            "k": int(args.greedy_k),
            "select_mode": args.select_mode,
            "chosen_global_indices": [int(i) for i in idx.tolist()],
        }
        with open(os.path.join(args.logs_dir, "purchase_plan.json"), "w") as f:
            json.dump(plan, f, indent=2)
        print("\nSaved greedy plan → logs/purchase_plan.json")

    print("\n=== Done (single-pass FHE on plaintext-averaged test-grad, single LogIX run) ===")

if __name__ == "__main__":
    main()
