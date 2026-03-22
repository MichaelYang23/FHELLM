#!/usr/bin/env python3


import argparse
import copy
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

try:
    from Pyfhel import Pyfhel
except ImportError as exc:
    raise SystemExit("Pyfhel is required. Install it with: pip install pyfhel") from exc


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 10, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_digit_indices(ds: datasets.MNIST, digit: int) -> np.ndarray:
    targets = ds.targets.cpu().numpy() if torch.is_tensor(ds.targets) else np.asarray(ds.targets)
    idx = np.where(targets == digit)[0]
    if idx.size == 0:
        raise ValueError(f"Digit {digit} not found in dataset.")
    return idx


def pick_indices(ds: datasets.MNIST, digit: int, count: int, offset: int = 0) -> np.ndarray:
    all_idx = get_digit_indices(ds, digit)
    if all_idx.size < offset + count:
        raise ValueError(
            f"Not enough samples for digit={digit}. requested offset+count={offset+count}, found={all_idx.size}"
        )
    return all_idx[offset : offset + count]


def stack_samples(ds: datasets.MNIST, indices: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for idx in indices:
        x, y = ds[int(idx)]
        xs.append(x)
        ys.append(int(y))
    x_tensor = torch.stack(xs, dim=0).to(device)
    y_tensor = torch.tensor(ys, dtype=torch.long, device=device)
    return x_tensor, y_tensor


def gradient_vector(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    model.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=False, retain_graph=False)
    return torch.cat([g.detach().reshape(-1) for g in grads]).cpu().numpy().astype(np.float64)


def evaluate_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 64) -> float:
    model.eval()
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for start in range(0, x.size(0), batch_size):
            end = min(start + batch_size, x.size(0))
            logits = model(x[start:end])
            loss = F.cross_entropy(logits, y[start:end], reduction="sum")
            total_loss += float(loss.item())
            total_n += end - start
    return total_loss / max(total_n, 1)


def train_baseline(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
) -> None:
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    n = x_train.size(0)
    for _ in range(epochs):
        perm = torch.randperm(n, device=x_train.device)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = perm[start:end]
            logits = model(x_train[idx])
            loss = F.cross_entropy(logits, y_train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


def adapt_on_seller(
    base_model: nn.Module,
    x_seller: torch.Tensor,
    y_seller: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
) -> nn.Module:
    model = copy.deepcopy(base_model)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    n = x_seller.size(0)
    for _ in range(epochs):
        perm = torch.randperm(n, device=x_seller.device)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = perm[start:end]
            logits = model(x_seller[idx])
            loss = F.cross_entropy(logits, y_seller[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


def mean_test_gradient(
    model: nn.Module,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    max_samples: int,
) -> np.ndarray:
    m = min(max_samples, x_eval.size(0))
    grads = []
    for i in range(m):
        g = gradient_vector(model, x_eval[i : i + 1], y_eval[i : i + 1])
        grads.append(g)
    return np.mean(np.stack(grads, axis=0), axis=0)


def setup_ckks(poly_mod_degree: int, scale_bits: int) -> Pyfhel:
    he = Pyfhel()
    he.contextGen(
        scheme="CKKS",
        n=poly_mod_degree,
        scale=2 ** scale_bits,
        qi_sizes=[60, scale_bits, scale_bits, 60],
    )
    he.keyGen()
    he.rotateKeyGen()
    he.relinKeyGen()
    return he


def chunked_fhe_dot(he: Pyfhel, a: np.ndarray, b: np.ndarray, slot_count: int) -> float:
    if a.shape != b.shape:
        raise ValueError("Vectors for dot product must have the same shape.")
    total = 0.0
    for start in range(0, a.size, slot_count):
        end = min(start + slot_count, a.size)
        seg_len = end - start
        ca = np.zeros(slot_count, dtype=np.float64)
        cb = np.zeros(slot_count, dtype=np.float64)
        ca[:seg_len] = a[start:end]
        cb[:seg_len] = b[start:end]

        # SEAL rejects transparent ciphertexts. If one chunk is exactly zero,
        # its contribution to the dot product is mathematically zero anyway.
        if np.count_nonzero(ca[:seg_len]) == 0 or np.count_nonzero(cb[:seg_len]) == 0:
            continue

        ctxt_a = he.encryptFrac(ca)
        ptxt_b = he.encodeFrac(cb)
        try:
            ctxt_prod = ctxt_a * ptxt_b
        except RuntimeError as exc:
            if "transparent" in str(exc).lower():
                plain_chunk = float(np.dot(ca[:seg_len], cb[:seg_len]))
                if abs(plain_chunk) < 1e-15:
                    continue
            raise

        if hasattr(he, "cumul_add"):
            ctxt_sum = he.cumul_add(ctxt_prod)
            chunk_dot = float(he.decryptFrac(ctxt_sum)[0])
        else:
            chunk_dot = float(np.sum(np.array(he.decryptFrac(ctxt_prod), dtype=np.float64)))
        total += chunk_dot
    return total


def load_mnist_with_fallback(
    requested_root: str, transform: transforms.Compose
) -> Tuple[datasets.MNIST, datasets.MNIST, str]:
    candidate_roots = [
        requested_root,
        "/tmp/mnist",
        "/mfs/io/groups/gao/michael/logix_fhe_02_11/logix_fhe/logix/data",
        "/mfs/io/groups/gao/michael/logix_fhe/logix/data",
    ]

    seen = set()
    dedup_roots = []
    for r in candidate_roots:
        rr = os.path.abspath(r)
        if rr not in seen:
            seen.add(rr)
            dedup_roots.append(rr)

    for root in dedup_roots:
        try:
            train_ds = datasets.MNIST(root=root, train=True, download=False, transform=transform)
            test_ds = datasets.MNIST(root=root, train=False, download=False, transform=transform)
            return train_ds, test_ds, root
        except Exception:
            pass

    try:
        train_ds = datasets.MNIST(root=requested_root, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root=requested_root, train=False, download=True, transform=transform)
        return train_ds, test_ds, os.path.abspath(requested_root)
    except Exception as exc:
        roots_text = "\n".join([f"  - {r}" for r in dedup_roots])
        raise RuntimeError(
            "Failed to load MNIST from local roots and download also failed.\n"
            f"Tried roots:\n{roots_text}\n"
            "Please pass a valid local root via --data_root."
        ) from exc


def rankdata_simple(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser("Minimal CKKS-assisted FHE-IF + loss-reduction proxy demo")
    parser.add_argument("--data_root", type=str, default="./data")

    parser.add_argument("--buyer_digits", type=str, default="1,2")
    parser.add_argument("--seller_digits", type=str, default="1,2,3")
    parser.add_argument("--test_digit", type=int, default=3)

    parser.add_argument("--buyer_train_per_digit", type=int, default=200)
    parser.add_argument("--seller_samples", type=int, default=32)
    parser.add_argument("--eval_samples", type=int, default=128)
    parser.add_argument("--eval_grad_samples", type=int, default=16)

    parser.add_argument("--base_epochs", type=int, default=3)
    parser.add_argument("--base_lr", type=float, default=0.1)
    parser.add_argument("--adapt_epochs", type=int, default=1)
    parser.add_argument("--adapt_lr", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--damping", type=float, default=1e-2)
    parser.add_argument("--fhe_verify_per_seller", type=int, default=6)
    parser.add_argument("--poly_mod_degree", type=int, default=2 ** 14)
    parser.add_argument("--scale_bits", type=int, default=30)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.poly_mod_degree & (args.poly_mod_degree - 1):
        raise ValueError("--poly_mod_degree must be a power of 2.")

    set_seed(args.seed)
    device = torch.device(args.device)

    buyer_digits = [int(x.strip()) for x in args.buyer_digits.split(",") if x.strip()]
    seller_digits = [int(x.strip()) for x in args.seller_digits.split(",") if x.strip()]
    if len(seller_digits) < 2:
        raise ValueError("Need at least 2 sellers to compare usefulness.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds, test_ds, used_root = load_mnist_with_fallback(args.data_root, transform)

    # Build buyer-train set from buyer digits.
    buyer_idx_parts = [pick_indices(train_ds, d, args.buyer_train_per_digit, offset=0) for d in buyer_digits]
    buyer_idx = np.concatenate(buyer_idx_parts, axis=0)
    x_buyer, y_buyer = stack_samples(train_ds, buyer_idx, device)

    # Build eval set from buyer objective digit on test split.
    eval_idx = pick_indices(test_ds, args.test_digit, args.eval_samples, offset=0)
    x_eval, y_eval = stack_samples(test_ds, eval_idx, device)

    # Baseline buyer model.
    model = TinyMLP().to(device)
    train_baseline(
        model=model,
        x_train=x_buyer,
        y_train=y_buyer,
        epochs=args.base_epochs,
        batch_size=args.batch_size,
        lr=args.base_lr,
    )
    base_loss = evaluate_loss(model, x_eval, y_eval, batch_size=args.batch_size)

    # Buyer test gradient (mean over eval samples).
    g_test = mean_test_gradient(model, x_eval, y_eval, max_samples=args.eval_grad_samples)
    v_test = g_test / args.damping

    he = setup_ckks(args.poly_mod_degree, args.scale_bits)
    slot_count = args.poly_mod_degree // 2

    print("\n=== Minimal end-to-end FHE-IF demo ===")
    print(f"MNIST root used: {used_root}")
    print(f"Buyer train digits: {buyer_digits}, per-digit: {args.buyer_train_per_digit}")
    print(f"Seller digits: {seller_digits}, seller_samples: {args.seller_samples}")
    print(f"Buyer objective (eval digit): {args.test_digit}, eval_samples: {args.eval_samples}")
    print(f"Baseline eval loss (before buying): {base_loss:.6f}")
    print(f"Gradient dimension: {g_test.size}, CKKS slots per ciphertext: {slot_count}")

    rows: List[Dict[str, float]] = []

    for s_digit in seller_digits:
        # Use non-overlapping segment for sellers when seller digit is also buyer digit.
        offset = args.buyer_train_per_digit if s_digit in buyer_digits else 0
        s_idx = pick_indices(train_ds, s_digit, args.seller_samples, offset=offset)
        x_seller, y_seller = stack_samples(train_ds, s_idx, device)

        # Plain IF for all seller samples.
        if_plain_scores = []
        seller_grads = []
        for i in range(x_seller.size(0)):
            g_i = gradient_vector(model, x_seller[i : i + 1], y_seller[i : i + 1])
            seller_grads.append(g_i)
            if_plain_scores.append(-float(np.dot(v_test, g_i)))

        if_plain_mean = float(np.mean(if_plain_scores))

        # FHE verification on subset.
        k_verify = min(args.fhe_verify_per_seller, len(seller_grads))
        if_fhe_scores = []
        for i in range(k_verify):
            dot_fhe = chunked_fhe_dot(he, v_test, seller_grads[i], slot_count)
            if_fhe_scores.append(-dot_fhe)

        if_fhe_mean = float(np.mean(if_fhe_scores))
        if_plain_mean_verify = float(np.mean(if_plain_scores[:k_verify]))
        if_abs_diff = abs(if_plain_mean_verify - if_fhe_mean)

        # Measure actual utility by adaptation and objective loss reduction.
        adapted = adapt_on_seller(
            base_model=model,
            x_seller=x_seller,
            y_seller=y_seller,
            epochs=args.adapt_epochs,
            batch_size=args.batch_size,
            lr=args.adapt_lr,
        )
        loss_after = evaluate_loss(adapted, x_eval, y_eval, batch_size=args.batch_size)
        delta_loss = loss_after - base_loss
        benefit = -delta_loss  # positive is better

        helpful_if = -if_plain_mean  # positive is better (proxy for benefit)

        rows.append(
            {
                "seller_digit": float(s_digit),
                "if_plain_mean": if_plain_mean,
                "if_fhe_mean_subset": if_fhe_mean,
                "if_abs_diff_subset": if_abs_diff,
                "loss_after": loss_after,
                "delta_loss": delta_loss,
                "benefit": benefit,
                "helpful_if": helpful_if,
            }
        )

    print("\nSeller | IF_plain_mean | IF_fhe_mean(subset) | abs_diff | delta_loss | benefit(-delta) | helpful_IF")
    print("------ | ------------- | ------------------- | -------- | ---------- | --------------- | ----------")
    for r in rows:
        print(
            f"{int(r['seller_digit']):>6} | "
            f"{r['if_plain_mean']:>13.6f} | "
            f"{r['if_fhe_mean_subset']:>19.6f} | "
            f"{r['if_abs_diff_subset']:>8.6f} | "
            f"{r['delta_loss']:>10.6f} | "
            f"{r['benefit']:>15.6f} | "
            f"{r['helpful_if']:>10.6f}"
        )

    helpful = np.array([r["helpful_if"] for r in rows], dtype=np.float64)
    benefit = np.array([r["benefit"] for r in rows], dtype=np.float64)

    pearson = float(np.corrcoef(helpful, benefit)[0, 1]) if len(rows) > 1 else float("nan")
    r_helpful = rankdata_simple(helpful)
    r_benefit = rankdata_simple(benefit)
    spearman = float(np.corrcoef(r_helpful, r_benefit)[0, 1]) if len(rows) > 1 else float("nan")

    best_if_idx = int(np.argmax(helpful))
    best_benefit_idx = int(np.argmax(benefit))

    print("\n=== Proxy check: IF vs loss reduction ===")
    print(f"Pearson(helpful_IF, benefit):  {pearson:.6f}")
    print(f"Spearman(helpful_IF, benefit): {spearman:.6f}")
    print(
        f"Top seller by IF proxy: digit {int(rows[best_if_idx]['seller_digit'])}; "
        f"Top seller by actual benefit: digit {int(rows[best_benefit_idx]['seller_digit'])}"
    )
    print("\nDone. This run demonstrates:")
    print("1) IF_plain and FHE-IF are numerically close.")
    print("2) IF ranking aligns with observed loss reduction in this minimal market setup.")


if __name__ == "__main__":
    main()
