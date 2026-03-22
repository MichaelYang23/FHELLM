#!/usr/bin/env python3


import argparse
import json
import os
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets

from utils import (
    set_seed,
    KEEP, REMAP, INV_REMAP, MNIST_NORM,
    construct_mlp,
    filter_mnist_indices,
    make_subset_loader_4class,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------- small helpers ----------------------------
def class_index_pools(
    dataset: datasets.MNIST,
    filtered_indices: np.ndarray,
) -> Dict[int, np.ndarray]:
    """
    Build per-original-digit pools (global indices) for the filtered 4-class pool.
    Keys are original digits in {1,2,3,4}.
    """
    targets = dataset.targets
    y_all = np.array([int(targets[i].item()) for i in filtered_indices], dtype=int)
    pools = {}
    for d in sorted(list(KEEP)):
        pools[d] = filtered_indices[y_all == d]
    return pools


def sample_without_overlap(
    pool: np.ndarray,
    used: set,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample n unique global indices from 'pool' that are not in 'used'.
    Returns np.ndarray of length n and updates 'used'.
    """
    if n == 0:
        return np.empty((0,), dtype=int)
    # remove used
    if used:
        used_arr = np.fromiter(used, dtype=int)
        avail = np.setdiff1d(pool, used_arr, assume_unique=False)
    else:
        avail = pool
    if avail.size < n:
        raise ValueError(f"Insufficient availability: need {n}, have {avail.size}")
    picks = rng.choice(avail, size=n, replace=False)
    used.update(int(i) for i in picks)
    return picks.astype(int)


def count_by_original_class(dataset: datasets.MNIST, indices: np.ndarray) -> Dict[int, int]:
    if indices.size == 0:
        return {}
    y = dataset.targets
    vals, cnts = np.unique([int(y[i]) for i in indices], return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnts)}


@torch.no_grad()
def compute_metrics_4class(model: torch.nn.Module, loader: DataLoader):
    """
    Evaluate mean CE loss and accuracy on a loader whose labels are remapped {0..3}.
    """
    model.eval().to(DEVICE)
    ce = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = ce(logits, y).item()
        preds = logits.argmax(dim=1)
        total_loss += loss
        total += y.size(0)
        total_correct += preds.eq(y).sum().item()
    avg_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1)
    return {"loss": float(avg_loss), "accuracy": float(acc), "num": int(total)}


# ---------------------------- main ----------------------------
def main():
    ap = argparse.ArgumentParser("Partition MNIST (4-class) and train buyer baseline")
    ap.add_argument("--root", type=str, default="./data")
    ap.add_argument("--outdir", type=str, default="./partitions")
    ap.add_argument("--seed", type=int, default=123)

    # Buyer: we keep {1,2}; configurable size & mix
    ap.add_argument("--buyer_size", type=int, default=6000)
    ap.add_argument("--buyer_p1", type=float, default=0.30)  # fraction of digit 1
    ap.add_argument("--buyer_p2", type=float, default=0.70)  # fraction of digit 2

    # Sellers (SINGLE-CLASS): A=1 only, B=2 only, C=3 only
    ap.add_argument("--seller_size", type=int, default=4000,
                    help="Per-seller size (each seller is single-class: A=1, B=2, C=3).")

    # Eval: SINGLE original digit from test split (cleaner demo)
    ap.add_argument("--eval_size", type=int, default=2000)
    ap.add_argument("--eval_digit", type=int, default=4, choices=[1,2,3,4],
                    help="Which original digit to use for eval set (single-class eval).")

    # Train hyperparams
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--batch_train", type=int, default=128)
    ap.add_argument("--batch_eval", type=int, default=256)

    # Checkpoint/output
    ap.add_argument("--ckpt_dir", type=str, default="./checkpoints/buyer")
    ap.add_argument("--history_file", type=str, default=None)
    args = ap.parse_args()

    # ---------------- RNG & IO ----------------
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ---------------- Load filtered MNIST (train/test) ----------------
    # train pool (4-class)
    tr_idx_all, tr_y_remap, ds_tr = filter_mnist_indices(root=args.root, split="train")
    # test pool (4-class)
    te_idx_all, te_y_remap, ds_te = filter_mnist_indices(root=args.root, split="test")

    # Build per-original-digit pools on the TRAIN filtered pool
    pools_tr = class_index_pools(ds_tr, tr_idx_all)  # keys in {1,2,3,4}

    # ---------------- Feasibility checks ----------------
    if not np.isclose(args.buyer_p1 + args.buyer_p2, 1.0):
        raise ValueError("buyer_p1 + buyer_p2 must equal 1.0")

    # Seller demand on the train split (global indices):
    # A:1 only, B:2 only, C:3 only
    demand = {1: args.seller_size, 2: args.seller_size, 3: args.seller_size, 4: 0}
    # availability per digit within filtered train pool:
    avail = {d: pools_tr[d].size for d in pools_tr.keys()}
    for d in sorted(KEEP):
        if demand.get(d, 0) > avail.get(d, 0):
            raise ValueError(
                f"Seller demand exceeds train availability for digit {d}: "
                f"demand={demand.get(d,0)}, available={avail.get(d,0)}"
            )

    # Max feasible buyer (digits 1 & 2 left after sellers)
    max_buyer_1 = avail[1] - demand[1]
    max_buyer_2 = avail[2] - demand[2]
    cap1 = int(np.floor(max_buyer_1 / args.buyer_p1)) if args.buyer_p1 > 0 else 10**12
    cap2 = int(np.floor(max_buyer_2 / args.buyer_p2)) if args.buyer_p2 > 0 else 10**12
    buyer_cap = min(cap1, cap2)
    buyer_size_requested = int(args.buyer_size)
    if args.buyer_size > buyer_cap:
        print(f"[auto_shrink_buyer] Requested buyer_size={args.buyer_size} > feasible={buyer_cap}.")
        args.buyer_size = buyer_cap
        print(f"[auto_shrink_buyer] Shrink buyer_size → {args.buyer_size}.")

    # ---------------- Sample disjoint TRAIN partitions ----------------
    used = set()

    # Buyer (digits 1 & 2 with configured mix)
    n_b1 = int(round(args.buyer_size * args.buyer_p1))
    n_b2 = args.buyer_size - n_b1
    buyer_1 = sample_without_overlap(pools_tr[1], used, n_b1, rng)
    buyer_2 = sample_without_overlap(pools_tr[2], used, n_b2, rng)
    buyer_train_idx = np.concatenate([buyer_1, buyer_2])
    rng.shuffle(buyer_train_idx)

    # Sellers (single-class)
    def sample_seller_single(digit, name):
        idx = sample_without_overlap(pools_tr[digit], used, args.seller_size, rng)
        print(f"{name}: size={idx.size} | counts={{ {digit}: {idx.size} }}")
        return idx

    sellerA_idx = sample_seller_single(1, "sellerA")
    sellerB_idx = sample_seller_single(2, "sellerB")
    sellerC_idx = sample_seller_single(3, "sellerC")

    # Final disjointness sanity (optional but cheap)
    all_tr = np.concatenate([buyer_train_idx, sellerA_idx, sellerB_idx, sellerC_idx])
    if all_tr.size != np.unique(all_tr).size:
        raise AssertionError("Partitions are not disjoint (train).")

    # ---------------- Sample EVAL set (from TEST pool, SINGLE digit) ----------------
    te_targets = ds_te.targets
    te_y_all = np.array([int(te_targets[i].item()) for i in te_idx_all], dtype=int)
    pool_eval = te_idx_all[te_y_all == int(args.eval_digit)]

    if args.eval_size > pool_eval.size:
        print(f"[auto_shrink_eval] Requested eval_size={args.eval_size} but cap={pool_eval.size}.")
        args.eval_size = int(pool_eval.size)
        print(f"[auto_shrink_eval] Shrink eval_size → {args.eval_size}")

    buyer_eval_idx = rng.choice(pool_eval, size=int(args.eval_size), replace=False)

    # ---------------- Save partitions ----------------
    np.save(os.path.join(args.outdir, "buyer_train_idx.npy"), buyer_train_idx)
    np.save(os.path.join(args.outdir, "sellerA_idx.npy"), sellerA_idx)
    np.save(os.path.join(args.outdir, "sellerB_idx.npy"), sellerB_idx)
    np.save(os.path.join(args.outdir, "sellerC_idx.npy"), sellerC_idx)
    np.save(os.path.join(args.outdir, "buyer_eval_idx.npy"), buyer_eval_idx)

    report = {
        "seed": args.seed,
        "keep_digits": sorted(list(KEEP)),
        "remap": {str(k): int(v) for k, v in REMAP.items()},
        "buyer": {
            "size_req": buyer_size_requested,
            "size_final": int(buyer_train_idx.size),
            "mix_req": {"1": args.buyer_p1, "2": args.buyer_p2},
            "counts": count_by_original_class(ds_tr, buyer_train_idx),
        },
        "sellerA_counts": count_by_original_class(ds_tr, sellerA_idx),
        "sellerB_counts": count_by_original_class(ds_tr, sellerB_idx),
        "sellerC_counts": count_by_original_class(ds_tr, sellerC_idx),
        "eval": {
            "digit": int(args.eval_digit),
            "size_req": int(args.eval_size),
            "size_final": int(buyer_eval_idx.size),
            "counts": count_by_original_class(ds_te, buyer_eval_idx),
        },
        "availability_train": {int(d): int(pools_tr[d].size) for d in pools_tr.keys()},
        "seller_demand": {int(k): int(v) for k, v in demand.items()},
        "buyer_cap_train": int(buyer_cap),
    }
    with open(os.path.join(args.outdir, "partition_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Partition summary (4-class, single-class sellers, single-class eval) ===")
    print(json.dumps(report, indent=2))

    # ---------------- Build loaders (4-class remapped) ----------------
    # IMPORTANT: for loaders we must use 4-class wrappers so labels are {0..3}
    train_loader = make_subset_loader_4class(
        dataset=ds_tr,
        filtered_indices=tr_idx_all,
        remapped_labels_all=tr_y_remap,
        subset_indices=buyer_train_idx,
        batch_size=args.batch_train,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    eval_loader = make_subset_loader_4class(
        dataset=ds_te,
        filtered_indices=te_idx_all,
        remapped_labels_all=te_y_remap,
        subset_indices=buyer_eval_idx,
        batch_size=args.batch_eval,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ---------------- Train buyer baseline (4-class MLP) ----------------
    model = construct_mlp(num_classes=4, seed=0).to(DEVICE)
    opt = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        total, correct, tot_loss = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * y.size(0)
            total += y.size(0)
            correct += logits.argmax(1).eq(y).sum().item()
        train_loss = tot_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        eval_metrics = compute_metrics_4class(model, eval_loader)

        history.append(
            {
                "epoch": ep,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "eval_loss": float(eval_metrics["loss"]),
                "eval_acc": float(eval_metrics["accuracy"]),
            }
        )
        print(
            f"[Epoch {ep:02d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
            f"eval_loss={eval_metrics['loss']:.4f} eval_acc={eval_metrics['accuracy']*100:.2f}%"
        )

    # Save checkpoint + history + final eval
    ckpt_path = os.path.join(args.ckpt_dir, f"mlp4_seed{args.seed}_epoch{args.epochs}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved buyer checkpoint → {ckpt_path}")

    final_eval = compute_metrics_4class(model, eval_loader)

    hist_path = args.history_file or os.path.join(args.ckpt_dir, f"training_history_seed{args.seed}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history → {hist_path}")

    baseline_path = os.path.join(args.ckpt_dir, "baseline_metrics.json")
    with open(baseline_path, "w") as f:
        json.dump(
            {
                "ckpt": ckpt_path,
                "seed": args.seed,
                "epochs": args.epochs,
                "train_final": history[-1] if history else {},
                "eval_final": final_eval,
                "note": "4-class setup; labels remapped {1,2,3,4}→{0,1,2,3}. Sellers are single-class (A=1,B=2,C=3).",
            },
            f,
            indent=2,
        )
    print(
        f"Eval on buyer_eval (digit={int(args.eval_digit)}): "
        f"acc={final_eval['accuracy']*100:.2f}% (n={final_eval['num']}) "
        f"loss={final_eval['loss']:.4f}"
    )


if __name__ == "__main__":
    main()
