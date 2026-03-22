#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_scores(tsv_path: str) -> List[Tuple[str, float]]:
    rows: List[Tuple[str, float]] = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            key = r.get("book_key", r.get("book", "unknown"))
            val = r.get("IF_sum", r.get("score", None))
            if val is None:
                continue
            rows.append((key, float(val)))
    if not rows:
        raise RuntimeError(f"No scores loaded from: {tsv_path}")
    return rows


def main():
    ap = argparse.ArgumentParser("Plot TIP score distribution for books")
    ap.add_argument("--scores_tsv", type=str, required=True, help="Path to per_book_scores.tsv")
    ap.add_argument("--out_png", type=str, default="./bookmkt.png", help="Output figure path")
    ap.add_argument("--title", type=str, default="")
    ap.add_argument("--fig_w", type=float, default=16.0)
    ap.add_argument("--fig_h", type=float, default=8.5)
    ap.add_argument("--dpi", type=int, default=350)
    ap.add_argument("--label_fontsize", type=int, default=24)
    ap.add_argument("--tick_fontsize", type=int, default=18)
    ap.add_argument("--title_fontsize", type=int, default=24)
    ap.add_argument("--bar_width", type=float, default=0.8)
    ap.add_argument("--color", type=str, default="#4C72B0")
    ap.add_argument("--ref_line", type=float, default=10000.0)
    args = ap.parse_args()

    rows = load_scores(args.scores_tsv)
    rows_sorted = sorted(rows, key=lambda x: x[1], reverse=True)
    scores = np.array([v for _, v in rows_sorted], dtype=np.float64)
    x = np.arange(scores.shape[0], dtype=np.int64)

    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    plt.figure(figsize=(args.fig_w, args.fig_h), dpi=args.dpi)
    plt.bar(x, scores, width=args.bar_width, color=args.color)
    if args.ref_line is not None:
        plt.axhline(y=args.ref_line, color="red", linestyle="--", linewidth=2.0)

    plt.xlabel("Book Index", fontsize=args.label_fontsize)
    plt.ylabel("Average TIP Score", fontsize=args.label_fontsize)
    if args.title:
        plt.title(args.title, fontsize=args.title_fontsize)
    plt.xticks(fontsize=args.tick_fontsize)
    plt.yticks(fontsize=args.tick_fontsize)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=args.dpi)
    print(f"[done] saved figure: {args.out_png}")


if __name__ == "__main__":
    main()

