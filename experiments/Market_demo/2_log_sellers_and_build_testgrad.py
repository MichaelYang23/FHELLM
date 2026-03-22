#!/usr/bin/env python3


import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from utils import (
    set_seed,
    KEEP, REMAP, INV_REMAP, MNIST_NORM,
    construct_mlp,
    filter_mnist_indices,
    make_subset_loader_4class,
)

import logix
from logix.utils import DataIDGenerator, get_logger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser("Log sellers (1/2/3) and build mean test gradient on eval digit")
    ap.add_argument("--partitions", type=str, default="./partitions")
    ap.add_argument("--root", type=str, default="./data")
    ap.add_argument("--buyer_ckpt", type=str, default="./checkpoints/buyer/mlp4_seed0_epoch10.pt")
    ap.add_argument("--project", type=str, default="mnist_market_4class")
    ap.add_argument("--outdir", type=str, default="./logs")

    # logging / eval batch sizes
    ap.add_argument("--seller_batch", type=int, default=512)
    ap.add_argument("--eval_batch", type=int, default=256)

    # LoRA / Hessian logging knobs
    ap.add_argument("--use_lora", action="store_true", default=True)
    ap.add_argument("--hessian", type=str, default="kfac", choices=["kfac", "none"])

    # which eval digit to average (original MNIST label)
    ap.add_argument("--eval_digit", type=int, default=3, choices=[1, 2, 3, 4])

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # ---------- Load partition indices ----------
    # Single-class sellers by construction:
    sellerA_idx = np.load(os.path.join(args.partitions, "sellerA_idx.npy"))  # digit 1
    sellerB_idx = np.load(os.path.join(args.partitions, "sellerB_idx.npy"))  # digit 2
    sellerC_idx = np.load(os.path.join(args.partitions, "sellerC_idx.npy"))  # digit 3

    # Buyer eval (already sampled as only the chosen eval_digit in step 1)
    buyer_eval = np.load(os.path.join(args.partitions, "buyer_eval_idx.npy"))

    # ---------- Prepare 4-class datasets (train/test) ----------
    tr_idx_all, tr_y_remap, ds_tr = filter_mnist_indices(root=args.root, split="train")
    te_idx_all, te_y_remap, ds_te = filter_mnist_indices(root=args.root, split="test")

    # ---------- Seller loaders (4-class remap) ----------
    def seller_loader(subset_idx):
        return make_subset_loader_4class(
            dataset=ds_tr,
            filtered_indices=tr_idx_all,
            remapped_labels_all=tr_y_remap,
            subset_indices=subset_idx,
            batch_size=args.seller_batch,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    loader_A = seller_loader(sellerA_idx)  # digit-1 only
    loader_B = seller_loader(sellerB_idx)  # digit-2 only
    loader_C = seller_loader(sellerC_idx)  # digit-3 only

    # ---------- Eval loader (only the chosen original digit) ----------
    # Step-1 already guaranteed buyer_eval contains only this digit; directly build 4-class remap loader
    eval_loader = make_subset_loader_4class(
        dataset=ds_te,
        filtered_indices=te_idx_all,
        remapped_labels_all=te_y_remap,
        subset_indices=buyer_eval,
        batch_size=args.eval_batch,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    get_logger().info(f"[Eval] Using original digit={args.eval_digit} | N={buyer_eval.size}")

    # ---------- Build buyer model (4-class head) ----------
    model = construct_mlp(num_classes=4, seed=0).to(DEVICE)
    state = torch.load(args.buyer_ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # ---------- Init one LogIX run (shared) ----------
    # NOTE: Do not pass non-existent parameters like storage_dir / resume
    run = logix.init(project=args.project)
    get_logger().info("Watching model for seller logging…")
    run.watch(model)

    if args.use_lora:
        get_logger().info("Adding LoRA adapters…")
        # Adding LoRA clears previous handlers (LogIX behavior), so watch again after adding
        run.add_lora()
        run.watch(model)
        get_logger().info("LoRA added.")

    # Covariance + gradient logging
    if args.hessian == "kfac":
        # Log covariance for both forward/backward, and also log grad (for later test-grad/IF computation)
        run.setup({"forward": ["covariance"], "backward": ["covariance"], "grad": ["log"]})
    else:
        run.setup({"grad": ["log"]})
    run.save(True)   # Allow writing seller logs to disk

    # ---------- Log sellers in fixed order A → B → C ----------
    manifest = {}
    idg = DataIDGenerator()

    def log_one(name, loader):
        count_seen = 0
        for x, y in loader:
            with run(data_id=idg(x)):
                x, y = x.to(DEVICE), y.to(DEVICE)
                model.zero_grad()
                out = model(x)
                # y is already 4-class remapped ({1,2,3,4}->{0,1,2,3})
                loss = torch.nn.functional.cross_entropy(out, y, reduction="sum")
                loss.backward()
            count_seen += x.size(0)
        return count_seen

    off = 0
    size_A = log_one("sellerA", loader_A); manifest["sellerA"] = {"size": int(size_A), "offset": int(off)}; off += size_A
    size_B = log_one("sellerB", loader_B); manifest["sellerB"] = {"size": int(size_B), "offset": int(off)}; off += size_B
    size_C = log_one("sellerC", loader_C); manifest["sellerC"] = {"size": int(size_C), "offset": int(off)}; off += size_C
    manifest["total_logged"] = int(off)

    # Complete seller logging - this step writes covariance_state etc. to disk at ./logix_logs/<project>/state/
    run.finalize()

    # Save catalog indices (for later slicing by seller)
    for name, arr in [("sellerA", sellerA_idx), ("sellerB", sellerB_idx), ("sellerC", sellerC_idx)]:
        d = os.path.join(args.outdir, name)
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "catalog_indices.npy"), arr)

    # ---------- Build mean test gradient on eval(digit=k) ----------
    # Use the same run in eval mode, backward pass per sample, average the structure (maintain 4-class remap)
    run.eval()
    run.setup({"grad": ["log"]})
    run.save(False)

    idg_eval = DataIDGenerator()
    acc_tree = None
    n_seen = 0

    # Small utilities for structured "add, scale" operations
    def tree_clone(x):
        if isinstance(x, torch.Tensor): return x.detach().clone()
        if isinstance(x, list):  return [tree_clone(v) for v in x]
        if isinstance(x, tuple): return tuple(tree_clone(v) for v in x)
        if isinstance(x, dict):  return {k: tree_clone(v) for k, v in x.items()}
        return x

    def tree_add_(dst, src):
        if type(dst) != type(src): return
        if isinstance(dst, torch.Tensor) and isinstance(src, torch.Tensor):
            if dst.shape == src.shape and dst.dtype == src.dtype:
                dst.add_(src.detach())
            return
        if isinstance(dst, list):
            for a, b in zip(dst, src): tree_add_(a, b); return
        if isinstance(dst, tuple):
            for a, b in zip(dst, src): tree_add_(a, b); return
        if isinstance(dst, dict):
            for k in dst.keys():
                if k in src: tree_add_(dst[k], src[k]); return

    def tree_mul_(dst, alpha: float):
        if isinstance(dst, torch.Tensor): dst.mul_(alpha); return
        if isinstance(dst, list):
            for x in dst: tree_mul_(x, alpha); return
        if isinstance(dst, tuple):
            for x in dst: tree_mul_(x, alpha); return
        if isinstance(dst, dict):
            for k in dst.keys(): tree_mul_(dst[k], alpha); return

    for xb, yb in eval_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        for i in range(xb.size(0)):
            x = xb[i:i+1]; y = yb[i:i+1]
            with run(data_id=idg_eval(x)):
                model.zero_grad()
                out = model(x)
                loss = torch.nn.functional.cross_entropy(out, y, reduction="sum")
                loss.backward()
            blob = run.get_log()
            if acc_tree is None:
                acc_tree = tree_clone(blob)
            else:
                tree_add_(acc_tree, blob)
            n_seen += 1

    if n_seen == 0:
        raise RuntimeError("Eval set empty; cannot compute mean test gradient.")

    tree_mul_(acc_tree, 1.0 / float(n_seen))

    buyer_dir = os.path.join(args.outdir, "buyer")
    os.makedirs(buyer_dir, exist_ok=True)
    torch.save(
        {"mean_test_tree": acc_tree, "count": int(n_seen), "note": f"Mean eval(digit={args.eval_digit}) gradient; labels remapped {{1..4}}→{{0..3}}."},
        os.path.join(buyer_dir, "mean_test_grad.pt"),
    )

    # ---------- Save manifest ----------
    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # ---------- Friendly printouts ----------
    print("\n=== Seller logging complete (single-class sellers) ===")
    print(json.dumps(manifest, indent=2))
    print(f"Logs saved under: {args.outdir}/sellerA, sellerB, sellerC")
    print(f"Saved mean eval (digit={args.eval_digit}) gradient over {n_seen} samples to {buyer_dir}/mean_test_grad.pt")
    print("=========================================\n")
    # Inform step-3 where to load covariance from
    print(f"[INFO] LogIX state (incl. covariance) saved to: ./logix_logs/{args.project}/state/")
    print("      In step-3, call: run._state.load_state(f'./logix_logs/%s')  # with project name" % args.project)


if __name__ == "__main__":
    main()
