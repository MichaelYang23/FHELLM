#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


SELLERS: Tuple[str, ...] = ("sellerA", "sellerB", "sellerC")
PAIRWISE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("sellerC", "sellerA"),
    ("sellerC", "sellerB"),
    ("sellerB", "sellerA"),
)
SELLER_METRICS: Tuple[str, ...] = ("if_sum", "if_mean", "benefit", "delta_acc", "delta_loss")
RANK_EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run repeated MNIST market experiments")

    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="./replicate_runs")
    parser.add_argument("--project_prefix", type=str, default="mnist_market_replicates")
    parser.add_argument("--score_backend", type=str, default="fhe", choices=["fhe", "plain"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--eval_digit", type=int, default=3, choices=[1, 2, 3, 4])

    parser.add_argument("--buyer_size", type=int, default=6000)
    parser.add_argument("--buyer_p1", type=float, default=0.30)
    parser.add_argument("--buyer_p2", type=float, default=0.70)
    parser.add_argument("--seller_size", type=int, default=4000)
    parser.add_argument("--eval_size", type=int, default=2000)

    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--train_lr", type=float, default=0.1)
    parser.add_argument("--train_momentum", type=float, default=0.9)
    parser.add_argument("--train_wd", type=float, default=1e-3)
    parser.add_argument("--batch_train", type=int, default=128)
    parser.add_argument("--batch_eval", type=int, default=256)

    parser.add_argument("--seller_batch", type=int, default=512)
    parser.add_argument("--eval_batch", type=int, default=256)
    parser.add_argument("--batch_log", type=int, default=64)
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_use_lora", action="store_false", dest="use_lora")
    parser.add_argument("--hessian", type=str, default="kfac", choices=["kfac", "none"])
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--greedy_k", type=int, default=0)
    parser.add_argument("--select_mode", type=str, default="max", choices=["max", "min"])

    parser.add_argument("--adapt_mode", type=str, default="head", choices=["head", "full"])
    parser.add_argument("--adapt_epochs", type=int, default=6)
    parser.add_argument("--adapt_lr", type=float, default=5e-3)
    parser.add_argument("--adapt_batch", type=int, default=256)
    parser.add_argument("--adapt_weight_decay", type=float, default=0.0)
    parser.add_argument("--adapt_clip_grad", type=float, default=1.0)

    return parser.parse_args()


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def build_rep_dirs(root: Path, rep: int) -> Dict[str, Path]:
    rep_dir = root / f"rep_{rep:04d}"
    return {
        "rep_dir": rep_dir,
        "partitions": rep_dir / "partitions",
        "ckpt_dir": rep_dir / "checkpoints" / "buyer",
        "logs_dir": rep_dir / "logs",
    }


def final_report_path(paths: Dict[str, Path]) -> Path:
    return paths["logs_dir"] / "if_vs_acc_report_single_eval.json"


def run_command(cmd: Sequence[str], cwd: Path, dry_run: bool) -> None:
    print("$", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True, env=build_subprocess_env())


def build_subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    repo_root = script_dir().parent.parent
    existing = env.get("PYTHONPATH", "")
    entries = [str(repo_root)]
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_if_stats(logs_dir: Path, eval_digit: int, score_backend: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    manifest = load_json(logs_dir / "manifest.json")
    scores = np.load(logs_dir / f"scores_eval{eval_digit}_{score_backend}.npy")

    if_sum: Dict[str, float] = {}
    if_mean: Dict[str, float] = {}
    for seller in SELLERS:
        offset = int(manifest[seller]["offset"])
        size = int(manifest[seller]["size"])
        vec = scores[offset : offset + size]
        if vec.shape[0] != size:
            raise RuntimeError(f"Score slice mismatch for {seller}: expected {size}, got {vec.shape[0]}")
        if_sum[seller] = float(np.sum(vec))
        if_mean[seller] = float(np.mean(vec))
    return if_sum, if_mean


def sorted_sellers_desc(metric_map: Dict[str, float]) -> List[str]:
    return [seller for seller, _ in sorted(metric_map.items(), key=lambda item: item[1], reverse=True)]


def sorted_sellers_asc(metric_map: Dict[str, float]) -> List[str]:
    return [seller for seller, _ in sorted(metric_map.items(), key=lambda item: item[1])]


def rank_positions(order: Sequence[str]) -> Dict[str, int]:
    return {seller: idx + 1 for idx, seller in enumerate(order)}


def top_sellers(metric_map: Dict[str, float], mode: str) -> List[str]:
    values = list(metric_map.values())
    target = max(values) if mode == "max" else min(values)
    return sorted(
        [seller for seller, value in metric_map.items() if math.isclose(value, target, abs_tol=RANK_EPS)],
    )


def summarize_replicate(
    rep: int,
    seed: int,
    paths: Dict[str, Path],
    eval_digit: int,
    score_backend: str,
) -> Tuple[List[dict], dict]:
    report = load_json(final_report_path(paths))
    if_sum, if_mean = load_if_stats(paths["logs_dir"], eval_digit, score_backend)

    observed = report["results"]
    benefit = {seller: float(-observed[seller]["delta_loss"]) for seller in SELLERS}
    delta_acc = {seller: float(observed[seller]["delta_acc"]) for seller in SELLERS}
    delta_loss = {seller: float(observed[seller]["delta_loss"]) for seller in SELLERS}

    if_order = sorted_sellers_desc(if_sum)
    benefit_order = sorted_sellers_desc(benefit)
    acc_order = sorted_sellers_desc(delta_acc)
    loss_order = sorted_sellers_asc(delta_loss)

    if_ranks = rank_positions(if_order)
    benefit_ranks = rank_positions(benefit_order)
    acc_ranks = rank_positions(acc_order)
    loss_ranks = rank_positions(loss_order)

    if_top = top_sellers(if_sum, mode="max")
    benefit_top = top_sellers(benefit, mode="max")

    replicate_rows: List[dict] = []
    for seller in SELLERS:
        obs = observed[seller]
        replicate_rows.append(
            {
                "replicate": rep,
                "seed": seed,
                "seller": seller,
                "eval_digit": eval_digit,
                "score_backend": score_backend,
                "baseline_acc": float(report["baseline"]["acc"]),
                "baseline_loss": float(report["baseline"]["loss"]),
                "if_sum": if_sum[seller],
                "if_mean": if_mean[seller],
                "acc": float(obs["acc"]),
                "loss": float(obs["loss"]),
                "delta_acc": float(obs["delta_acc"]),
                "delta_loss": float(obs["delta_loss"]),
                "benefit": benefit[seller],
                "if_rank": if_ranks[seller],
                "benefit_rank": benefit_ranks[seller],
                "acc_rank": acc_ranks[seller],
                "loss_rank": loss_ranks[seller],
                "is_unique_if_top": int(len(if_top) == 1 and seller == if_top[0]),
                "is_unique_benefit_top": int(len(benefit_top) == 1 and seller == benefit_top[0]),
                "is_any_if_top": int(seller in if_top),
                "is_any_benefit_top": int(seller in benefit_top),
            }
        )

    replicate_summary = {
        "replicate": rep,
        "seed": seed,
        "eval_digit": eval_digit,
        "score_backend": score_backend,
        "baseline_acc": float(report["baseline"]["acc"]),
        "baseline_loss": float(report["baseline"]["loss"]),
        "if_top_sellers": if_top,
        "benefit_top_sellers": benefit_top,
        "if_top_unique": if_top[0] if len(if_top) == 1 else "",
        "benefit_top_unique": benefit_top[0] if len(benefit_top) == 1 else "",
        "top1_unique_match": int(len(if_top) == 1 and len(benefit_top) == 1 and if_top[0] == benefit_top[0]),
        "top1_any_overlap": int(bool(set(if_top) & set(benefit_top))),
        "sellerC_top_if": int("sellerC" in if_top),
        "sellerC_top_benefit": int("sellerC" in benefit_top),
    }
    for left, right in PAIRWISE_PAIRS:
        left_short = left[-1]
        right_short = right[-1]
        replicate_summary[f"if_sum_gap_{left_short}_{right_short}"] = if_sum[left] - if_sum[right]
        replicate_summary[f"if_mean_gap_{left_short}_{right_short}"] = if_mean[left] - if_mean[right]
        replicate_summary[f"benefit_gap_{left_short}_{right_short}"] = benefit[left] - benefit[right]
        replicate_summary[f"delta_acc_gap_{left_short}_{right_short}"] = delta_acc[left] - delta_acc[right]
        replicate_summary[f"delta_loss_gap_{left_short}_{right_short}"] = delta_loss[left] - delta_loss[right]

    return replicate_rows, replicate_summary


def mean_sd_se_ci(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "se": None, "ci95_low": None, "ci95_high": None, "min": None, "max": None}
    mean = float(np.mean(arr))
    if n == 1:
        sd = se = 0.0
    else:
        sd = float(np.std(arr, ddof=1))
        se = float(sd / math.sqrt(n))
    ci_half = 1.96 * se
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95_low": mean - ci_half,
        "ci95_high": mean + ci_half,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float | None, float | None]:
    if n == 0:
        return None, None
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n))
    return center - margin, center + margin


def binom_tail_greater(k: int, n: int, p0: float) -> float:
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, i) * (p0**i) * ((1.0 - p0) ** (n - i)) for i in range(k, n + 1)))


def binom_pmf(k: int, n: int, p0: float) -> float:
    return float(math.comb(n, k) * (p0**k) * ((1.0 - p0) ** (n - k)))


def binom_two_sided(k: int, n: int, p0: float) -> float:
    if n == 0:
        return 1.0
    pmf_k = binom_pmf(k, n, p0)
    total = 0.0
    for i in range(n + 1):
        pmf_i = binom_pmf(i, n, p0)
        if pmf_i <= pmf_k + 1e-15:
            total += pmf_i
    return float(min(total, 1.0))


def summarize_binary_indicator(name: str, values: Iterable[int], p0: float) -> dict:
    arr = np.asarray(list(values), dtype=int)
    n = int(arr.size)
    k = int(arr.sum())
    rate = float(k / n) if n else None
    ci_low, ci_high = wilson_interval(k, n)
    return {
        "metric": name,
        "n": n,
        "successes": k,
        "rate": rate,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "null_rate": p0,
        "p_value_greater": binom_tail_greater(k, n, p0),
        "p_value_two_sided": binom_two_sided(k, n, p0),
    }


def summarize_pairwise(metric: str, left: str, right: str, values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=float)
    stats = mean_sd_se_ci(arr)
    positives = int(np.sum(arr > RANK_EPS))
    negatives = int(np.sum(arr < -RANK_EPS))
    zeros = int(arr.size - positives - negatives)
    n_nonzero = positives + negatives
    sign_p_greater = binom_tail_greater(positives, n_nonzero, 0.5)
    sign_p_two_sided = binom_two_sided(max(positives, negatives), n_nonzero, 0.5)
    return {
        "metric": metric,
        "left": left,
        "right": right,
        "comparison": f"{left}-{right}",
        "n": stats["n"],
        "mean_diff": stats["mean"],
        "sd_diff": stats["sd"],
        "se_diff": stats["se"],
        "ci95_low": stats["ci95_low"],
        "ci95_high": stats["ci95_high"],
        "min_diff": stats["min"],
        "max_diff": stats["max"],
        "positives": positives,
        "negatives": negatives,
        "zeros": zeros,
        "n_nonzero": n_nonzero,
        "sign_test_p_greater": sign_p_greater,
        "sign_test_p_two_sided": sign_p_two_sided,
    }


def build_summary(all_rows: List[dict], rep_rows: List[dict], args: argparse.Namespace) -> Tuple[List[dict], List[dict], List[dict], dict]:
    seller_summary_rows: List[dict] = []
    for metric in SELLER_METRICS:
        for seller in SELLERS:
            values = [float(row[metric]) for row in all_rows if row["seller"] == seller]
            stats = mean_sd_se_ci(values)
            seller_summary_rows.append(
                {
                    "metric": metric,
                    "seller": seller,
                    **stats,
                }
            )

    pairwise_rows: List[dict] = []
    metric_prefixes = ("if_sum", "if_mean", "benefit", "delta_acc", "delta_loss")
    for metric in metric_prefixes:
        for left, right in PAIRWISE_PAIRS:
            left_short = left[-1]
            right_short = right[-1]
            key = f"{metric}_gap_{left_short}_{right_short}"
            values = [float(row[key]) for row in rep_rows]
            pairwise_rows.append(summarize_pairwise(metric, left, right, values))

    ranking_rows = [
        summarize_binary_indicator("top1_unique_match", [row["top1_unique_match"] for row in rep_rows], p0=1.0 / 3.0),
        summarize_binary_indicator("top1_any_overlap", [row["top1_any_overlap"] for row in rep_rows], p0=1.0 / 3.0),
        summarize_binary_indicator("sellerC_top_if", [row["sellerC_top_if"] for row in rep_rows], p0=1.0 / 3.0),
        summarize_binary_indicator("sellerC_top_benefit", [row["sellerC_top_benefit"] for row in rep_rows], p0=1.0 / 3.0),
    ]

    summary_json = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reps_requested": int(args.reps),
        "reps_completed": int(len(rep_rows)),
        "eval_digit": int(args.eval_digit),
        "outdir": str(Path(args.outdir).resolve()),
        "notes": {
            "if_sign_convention": "Positive IF means predicted helpful.",
            "benefit_definition": "benefit = -delta_loss; larger means better.",
            "pairwise_significance": "sign_test_p_* is computed on replicate-level paired differences.",
        },
        "seller_metric_summary": seller_summary_rows,
        "pairwise_summary": pairwise_rows,
        "ranking_summary": ranking_rows,
    }
    return seller_summary_rows, pairwise_rows, ranking_rows, summary_json


def maybe_skip_rep(paths: Dict[str, Path], resume: bool) -> bool:
    return resume and final_report_path(paths).exists()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    root_path = Path(args.root).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    base_dir = script_dir()
    script_partition = base_dir / "1_partition_and_train.py"
    script_combined = base_dir / ("combined_log_and_fhe.py" if args.score_backend == "fhe" else "combined_log_NO_FHE.py")
    script_report = base_dir / "4_report_consistency.py"

    all_rows: List[dict] = []
    rep_rows: List[dict] = []
    failures: List[dict] = []

    for rep in range(args.reps):
        seed = args.seed_start + rep
        paths = build_rep_dirs(outdir, rep)
        paths["rep_dir"].mkdir(parents=True, exist_ok=True)

        print(f"\n=== Replicate {rep + 1}/{args.reps} | seed={seed} ===")

        try:
            if maybe_skip_rep(paths, args.resume):
                print(f"Reusing existing outputs in {paths['rep_dir']}")
            else:
                ckpt_path = paths["ckpt_dir"] / f"mlp4_seed{seed}_epoch{args.train_epochs}.pt"

                cmd_partition = [
                    py,
                    str(script_partition),
                    "--root",
                    str(root_path),
                    "--outdir",
                    str(paths["partitions"]),
                    "--seed",
                    str(seed),
                    "--buyer_size",
                    str(args.buyer_size),
                    "--buyer_p1",
                    str(args.buyer_p1),
                    "--buyer_p2",
                    str(args.buyer_p2),
                    "--seller_size",
                    str(args.seller_size),
                    "--eval_size",
                    str(args.eval_size),
                    "--eval_digit",
                    str(args.eval_digit),
                    "--epochs",
                    str(args.train_epochs),
                    "--lr",
                    str(args.train_lr),
                    "--momentum",
                    str(args.train_momentum),
                    "--wd",
                    str(args.train_wd),
                    "--batch_train",
                    str(args.batch_train),
                    "--batch_eval",
                    str(args.batch_eval),
                    "--ckpt_dir",
                    str(paths["ckpt_dir"]),
                ]

                cmd_combined = [
                    py,
                    str(script_combined),
                    "--project",
                    f"{args.project_prefix}_rep{rep:04d}",
                    "--buyer_ckpt",
                    str(ckpt_path),
                    "--partitions",
                    str(paths["partitions"]),
                    "--root",
                    str(root_path),
                    "--logs_dir",
                    str(paths["logs_dir"]),
                    "--eval_digit",
                    str(args.eval_digit),
                    "--seller_batch",
                    str(args.seller_batch),
                    "--eval_batch",
                    str(args.eval_batch),
                    "--batch_log",
                    str(args.batch_log),
                    "--hessian",
                    args.hessian,
                    "--damping",
                    str(args.damping),
                    "--greedy_k",
                    str(args.greedy_k),
                    "--select_mode",
                    args.select_mode,
                    "--seed",
                    str(seed),
                ]
                if args.use_lora:
                    cmd_combined.append("--use_lora")

                cmd_report = [
                    py,
                    str(script_report),
                    "--buyer_ckpt",
                    str(ckpt_path),
                    "--partitions",
                    str(paths["partitions"]),
                    "--logs_dir",
                    str(paths["logs_dir"]),
                    "--root",
                    str(root_path),
                    "--eval_digit",
                    str(args.eval_digit),
                    "--score_backend",
                    args.score_backend,
                    "--adapt_mode",
                    args.adapt_mode,
                    "--epochs",
                    str(args.adapt_epochs),
                    "--lr",
                    str(args.adapt_lr),
                    "--batch",
                    str(args.adapt_batch),
                    "--weight_decay",
                    str(args.adapt_weight_decay),
                    "--clip_grad",
                    str(args.adapt_clip_grad),
                    "--seed",
                    str(seed),
                ]

                run_command(cmd_partition, cwd=paths["rep_dir"], dry_run=args.dry_run)
                run_command(cmd_combined, cwd=paths["rep_dir"], dry_run=args.dry_run)
                run_command(cmd_report, cwd=paths["rep_dir"], dry_run=args.dry_run)

            if args.dry_run:
                continue

            replicate_rows, replicate_summary = summarize_replicate(
                rep=rep,
                seed=seed,
                paths=paths,
                eval_digit=args.eval_digit,
                score_backend=args.score_backend,
            )
            all_rows.extend(replicate_rows)
            rep_rows.append(replicate_summary)
        except Exception as exc:  # noqa: BLE001
            failure = {
                "replicate": rep,
                "seed": seed,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            print(f"[ERROR] replicate={rep} seed={seed}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                break

    if args.dry_run:
        print("Dry run complete.")
        return

    save_csv(outdir / "replicate_results.csv", all_rows)
    save_csv(outdir / "replicate_summary.csv", rep_rows)
    save_json(outdir / "failures.json", {"failures": failures})

    seller_summary_rows, pairwise_rows, ranking_rows, summary_json = build_summary(all_rows, rep_rows, args)
    save_csv(outdir / "summary_by_seller.csv", seller_summary_rows)
    save_csv(outdir / "pairwise_summary.csv", pairwise_rows)
    save_csv(outdir / "ranking_summary.csv", ranking_rows)
    save_json(outdir / "summary.json", summary_json)

    print("\n=== Finished ===")
    print(f"Completed replicates: {len(rep_rows)}/{args.reps}")
    print(f"Per-seller rows: {outdir / 'replicate_results.csv'}")
    print(f"Replicate summary: {outdir / 'replicate_summary.csv'}")
    print(f"Seller stats: {outdir / 'summary_by_seller.csv'}")
    print(f"Pairwise stats: {outdir / 'pairwise_summary.csv'}")
    print(f"Ranking stats: {outdir / 'ranking_summary.csv'}")
    print(f"JSON summary: {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
