#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, json, argparse
import numpy as np
import torch
from torch import nn
from torch.optim import SGD

from utils import (
    set_seed,
    construct_mlp,
    filter_mnist_indices,
    make_subset_loader_4class,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------- Eval helpers ----------------------
@torch.no_grad()
def eval_stats(model, loader):
    """Mean CE loss + accuracy on the given loader (labels already remapped to {0..3})."""
    model.eval().to(DEVICE)
    ce_sum = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        total_loss += ce_sum(logits, y).item()
        pred = logits.argmax(1)
        total_correct += (pred == y).sum().item()
        total += y.size(0)
    avg_loss = total_loss / max(1, total)
    acc = total_correct / max(1, total)
    return acc, avg_loss


# ---------------------- Adaptation helpers ----------------------
def freeze_except_head(model):
    """Freeze all params except the final Linear in our MLP scaffold."""
    for _, p in model.named_parameters():
        p.requires_grad = False
    # Unfreeze the last nn.Linear
    last_lin = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_lin = m
    if last_lin is not None:
        for p in last_lin.parameters():
            p.requires_grad = True


def adapt_once(model, train_loader, mode="head", epochs=6, lr=5e-3, weight_decay=0.0, clip_grad=1.0):
    """One-shot small adaptation on a seller bundle."""
    if mode not in ("head", "full"):
        raise ValueError("mode must be 'head' or 'full'")
    model = model.to(DEVICE)
    model.train()
    if mode == "head":
        freeze_except_head(model)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters for adaptation.")
    opt = SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    ce = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = ce(out, y)
            loss.backward()
            if clip_grad is not None and clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(params, clip_grad)
            opt.step()

    model.eval()
    return model


# ---------------------- IO helpers ----------------------
def load_manifest_and_catalogs(logs_dir):
    man = os.path.join(logs_dir, "manifest.json")
    if not os.path.exists(man):
        raise FileNotFoundError(f"Missing manifest: {man}")
    with open(man, "r") as f:
        manifest = json.load(f)

    cats = {}
    for s in ("sellerA","sellerB","sellerC"):
        p = os.path.join(logs_dir, s, "catalog_indices.npy")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing catalog for {s}: {p}")
        arr = np.load(p)
        if arr.size == 0:
            raise RuntimeError(f"Empty catalog for {s}.")
        cats[s] = arr
    return manifest, cats


def slice_by_seller(vec, manifest):
    out = {}
    for s in ("sellerA","sellerB","sellerC"):
        off = int(manifest[s]["offset"]); size = int(manifest[s]["size"])
        out[s] = vec[off:off+size]
        if out[s].shape[0] != size:
            raise RuntimeError(f"[{s}] slice length mismatch.")
    return out


# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser("IF vs observed ΔAcc/ΔLoss consistency (single eval digit)")
    ap.add_argument("--buyer_ckpt", type=str, default="./checkpoints/buyer/mlp4_seed0_epoch10.pt")
    ap.add_argument("--partitions", type=str, default="./partitions")
    ap.add_argument("--logs_dir", type=str, default="./logs")
    ap.add_argument("--root", type=str, default="./data")

    # single eval digit (e.g., 3)
    ap.add_argument("--eval_digit", type=int, default=3, choices=[1,2,3,4])
    ap.add_argument("--score_backend", type=str, default="fhe", choices=["fhe", "plain"])

    # Adaptation knobs
    ap.add_argument("--adapt_mode", type=str, default="head", choices=["head","full"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--clip_grad", type=float, default=1.0)

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)

    score_path = os.path.join(args.logs_dir, f"scores_eval{args.eval_digit}_{args.score_backend}.npy")
    if not os.path.exists(score_path):
        raise FileNotFoundError(
            f"Missing IF scores for eval digit {args.eval_digit}: {score_path}. "
            f"Run the matching scoring script with --eval_digit {args.eval_digit} first."
        )
    scores = np.load(score_path)

    manifest, catalogs = load_manifest_and_catalogs(args.logs_dir)
    byseller = slice_by_seller(scores, manifest)

    def sum_dict(d): return {k: float(np.sum(v)) for k,v in d.items()}
    print(f"=== IF aggregates on eval digit={args.eval_digit} [{args.score_backend}] (more positive ⇒ predicted helpful) ===")
    print({k: f"{v:+.6f}" for k,v in sum_dict(byseller).items()})
    print()

    # ---- Build eval loader (ONLY this digit) with the SAME 4-class remap ----
    te_idx_all, te_y_remap, ds_te = filter_mnist_indices(root=args.root, split="test")
    buyer_eval_idx = np.load(os.path.join(args.partitions, "buyer_eval_idx.npy"))

    idx_map = {int(g): i for i, g in enumerate(te_idx_all.tolist())}
    only_eval = []
    target_remap = {1:0, 2:1, 3:2, 4:3}[args.eval_digit]
    for g in buyer_eval_idx.tolist():
        pos = idx_map.get(int(g), None)
        if pos is None: 
            continue
        if int(te_y_remap[pos]) == target_remap:
            only_eval.append(int(g))
    only_eval = np.array(only_eval, dtype=int)
    if only_eval.size == 0:
        raise RuntimeError("Eval set for the chosen digit is empty.")

    eval_loader = make_subset_loader_4class(
        ds_te, te_idx_all, te_y_remap, only_eval,
        batch_size=512, shuffle=False, num_workers=0, pin_memory=False
    )

    # ---- Build training (seller) loaders with SAME remap ----
    tr_idx_all, tr_y_remap, ds_tr = filter_mnist_indices(root=args.root, split="train")
    tr_pos_map = {int(g): i for i, g in enumerate(tr_idx_all.tolist())}

    def make_seller_loader(seller_key):
        seller_orig = catalogs[seller_key].tolist()
        pos = [tr_pos_map[i] for i in seller_orig if i in tr_pos_map]
        subset = tr_idx_all[pos]
        return make_subset_loader_4class(
            ds_tr, tr_idx_all, tr_y_remap, subset,
            batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=False
        )

    loaders_train = {
        "sellerA": make_seller_loader("sellerA"),
        "sellerB": make_seller_loader("sellerB"),
        "sellerC": make_seller_loader("sellerC"),
    }

    # ---- Baseline model (4-class) ----
    base = construct_mlp(num_classes=4, seed=0).to(DEVICE)
    sd = torch.load(args.buyer_ckpt, map_location="cpu")
    base.load_state_dict(sd); base.eval()

    acc0, loss0 = eval_stats(base, eval_loader)
    print(f"Baseline on eval(digit={args.eval_digit}): Acc={acc0*100:.2f}% | Loss={loss0:.4f}\n")

    # ---- One-shot per seller ----
    results = {}
    for s in ("sellerA","sellerB","sellerC"):
        print(f"[ONESHOT] {s}")
        model = construct_mlp(num_classes=4, seed=0).to(DEVICE)
        model.load_state_dict(sd)

        model = adapt_once(
            model,
            loaders_train[s],
            mode=args.adapt_mode,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            clip_grad=args.clip_grad,
        )

        acc_s, loss_s = eval_stats(model, eval_loader)
        print(f"  Acc={acc_s*100:.2f}% (Δ {(acc_s-acc0)*100:+.2f} pp) | "
              f"Loss={loss_s:.4f} (Δ {(loss_s-loss0):+.4f})\n")

        results[s] = {
            "acc": float(acc_s),
            "loss": float(loss_s),
            "delta_acc": float(acc_s - acc0),
            "delta_loss": float(loss_s - loss0),
        }

    # ---- Summarize IF vs Observed (single eval digit) ----
    table_if = {k: float(np.sum(v)) for k, v in byseller.items()}
    # Positive IF => helpful => rank in descending order
    rank_if_desc = [k for k,_ in sorted(table_if.items(), key=lambda kv: kv[1], reverse=True)]
    rank_acc_desc = [k for k,_ in sorted(results.items(), key=lambda kv: kv[1]["acc"], reverse=True)]
    rank_loss_asc = [k for k,_ in sorted(results.items(), key=lambda kv: kv[1]["loss"])]

    report = {
        "eval_digit": int(args.eval_digit),
        "score_backend": args.score_backend,
        "adapt_mode": args.adapt_mode,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch": args.batch,
        "weight_decay": args.weight_decay,
        "clip_grad": args.clip_grad,

        "baseline": {"acc": float(acc0), "loss": float(loss0)},
        "if_sum": table_if,
        "rank_if_desc_positive_good": rank_if_desc,
        "results": results,
        "rank_observed_acc_desc": rank_acc_desc,
        "rank_observed_loss_asc": rank_loss_asc,
        "note": "Positive IF ⇒ predicted helpful. Single-digit eval setup.",
    }

    os.makedirs(args.logs_dir, exist_ok=True)
    out = os.path.join(args.logs_dir, "if_vs_acc_report_single_eval.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved consistency report → {out}")
    print("Done.")


if __name__ == "__main__":
    main()
