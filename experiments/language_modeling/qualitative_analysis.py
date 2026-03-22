#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Qualitative analysis of influence scores
# -----------------------------------------------------------------------------
# Read pre-computed influence scores (scores.pt), map them back to human-readable text,
# print each query (validation sample) along with top-k most influential training samples,
# for manual sanity-checking by researchers.
#
# Copyright 2023-present LogIX team.
# License: Apache 2.0
# -----------------------------------------------------------------------------

import argparse
from pathlib import Path
from typing import List

import torch
from scipy.stats import pearsonr
from transformers import AutoTokenizer

from utils import get_loaders, set_seed  # get_loaders is defined in utils.py

parser = argparse.ArgumentParser(
    "Qualitative inspection of influence scores", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--score_path", type=Path, required=True,
                    help="Path to the *.pt file that stores influence scores "
                         "(shape [n_test, n_train]).")
parser.add_argument("--score_path2", type=Path,
                    help="Optional second score file to compute Pearson correlation.")
# data / model
parser.add_argument("--model_name", type=str,
                    default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument("--data_path", type=str, default="wikitext",
                    help="HF dataset path or 'generated' / 'external' etc.")
parser.add_argument("--data_name", type=str, default="wikitext-2-v1",
                    help="HF dataset config name (e.g. wikitext-2-v1).")
parser.add_argument("--cache_dir", type=str, default="/mfs/io/groups/gao/michael/huggingface_cache",
                    help="Directory for HF datasets & tokenizer cache.")
# analysis options
parser.add_argument("--subset_size", type=int, default=128,
                    help="How many train indices to keep for qualitative display.")
parser.add_argument("--num_queries", type=int, default=16,
                    help="How many validation/test queries to print.")
parser.add_argument("--top_k", type=int, default=3,
                    help="How many top-influence training samples to show per query.")
args = parser.parse_args()

# -----------------------------------------------------------------------------#
#                            Initialise & load data                            #
# -----------------------------------------------------------------------------#
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(0)

# 1. Build DataLoaders (train / eval-train / validation)
_, eval_train_loader, test_loader = get_loaders(
    model_name=args.model_name,
    data_path=args.data_path,
    data_name=args.data_name,
    cache_dir=args.cache_dir,
    train_batch_size=1,            # full train set; we only need eval_train_loader
    test_batch_size=1,
    valid_indices=list(range(args.subset_size)),  # keep only first N train samples
)

# 2. Tokenizer - only used for decode; model doesn't need to be loaded
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name, use_fast=True, trust_remote_code=True, cache_dir=args.cache_dir
)

# -----------------------------------------------------------------------------#
#                        Load influence score tensor(s)                        #
# -----------------------------------------------------------------------------#
scores = torch.load(args.score_path, map_location="cpu")

if args.score_path2:
    scores2 = torch.load(args.score_path2, map_location="cpu")
    # Compute Pearson correlation for each query, then average
    corrs: List[float] = [
        pearsonr(s1, s2).statistic for s1, s2 in zip(scores, scores2)
    ]
    print(f"[Info] Average Pearson correlation between two score files: "
          f"{sum(corrs) / len(corrs):.4f}")

# -----------------------------------------------------------------------------#
#                           Print qualitative results                          #
# -----------------------------------------------------------------------------#
for q_idx in range(min(args.num_queries, len(test_loader.dataset))):
    print("=" * 100)
    print(f"[Query {q_idx}]")
    query_text = tokenizer.decode(test_loader.dataset[q_idx]["input_ids"], skip_special_tokens=True)
    print(f"• Query sentence:\n{query_text}\n")

    # Get top-k indices
    topk_idx = torch.argsort(scores[q_idx], descending=True)[: args.top_k]

    print(f"Top {args.top_k} most influential training examples:")
    for rank, idx in enumerate(topk_idx):
        influence_value = scores[q_idx][idx].item()
        train_text = tokenizer.decode(
            eval_train_loader.dataset[int(idx)]["input_ids"], skip_special_tokens=True
        )
        print(f"  [{rank:>2}] score = {influence_value:+.6f}\n      {train_text}\n")

    input("Press <Enter> to continue…")

print("\n[Done] Qualitative inspection complete.")
