#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import copy
import time

import torch
import torch.nn.functional as F
import numpy as np
from accelerate import Accelerator
from tqdm import tqdm

import logix
from logix.analysis import InfluenceFunction
from logix.utils import DataIDGenerator, merge_logs, get_logger

# Import your BERT loading utilities
from examples.bert.utils import construct_model, get_loaders

def main():
    parser = argparse.ArgumentParser("BERT FHE vs Plaintext Validator")
    parser.add_argument("--project",      required=True, help="LogIX project name")
    parser.add_argument("--config",       required=True, help="LogIX config YAML path")
    parser.add_argument("--data_name",    default="sst2", help="GLUE task name")
    parser.add_argument("--checkpoint",   required=True, help="Fine-tuned BERT checkpoint")
    parser.add_argument("--fhe_results",  required=True, help="FHE results .pt file path")
    parser.add_argument("--hessian",      default="raw", help="Hessian preconditioning method")
    parser.add_argument("--damping",      type=float, default=None, help="Hessian damping")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Number of test samples")
    parser.add_argument("--lora",         action="store_true", help="Whether LoRA was used during extraction")
    parser.add_argument("--data_id_mode", default="hash", choices=["hash","decode"])
    args = parser.parse_args()

    logger = get_logger()
    accel  = Accelerator()

    # 1. Initialize LogIX and restore state
    run = logix.init(project=args.project, config=args.config)
    if run is None:
        # If init() returns None, get global instance and reload config
        run = logix._LOGIX_INSTANCE
        run.config.load_config(args.config)
    run.initialize_from_log()
    logger.info(f"Loaded LogIX state from {run.log_dir}")

    # 2. Reconstruct model, LoRA, and watch
    model, tokenizer = construct_model(args.data_name, ckpt_path=args.checkpoint)
    run.model = model
    if args.lora:
        run.add_lora()
    run.watch(model, type_filter=[torch.nn.Linear])
    model = accel.prepare(model)
    model.eval()

    # 3. Build training gradient DataLoader
    log_loader = run.build_log_dataloader(batch_size=64, flatten=True)
    logger.info(f"Log DataLoader: {len(log_loader.dataset)} training samples")

    # 4. Recompute test sample gradients (usually fast)
    test_loader = get_loaders(data_name=args.data_name,
                              eval_batch_size=args.test_batch_size,
                              valid_indices=list(range(args.test_batch_size)))[-1]
    test_loader = accel.prepare(test_loader)
    id_gen = DataIDGenerator(mode=args.data_id_mode)

    run.eval()
    run.setup({"grad": ["log"]})
    test_logs = []
    for batch in tqdm(test_loader, desc="Computing Test Gradients"):
        batch_ids = id_gen(batch["input_ids"])
        labels = batch.pop("labels").view(-1)
        if "idx" in batch: batch.pop("idx")
        with run(data_id=batch_ids, mask=batch.get("attention_mask")):
            model.zero_grad()
            out = model(**batch)
            logits = out.view(-1, out.shape[-1])
            loss = F.cross_entropy(logits, labels, reduction="sum", ignore_index=-100)
            accel.backward(loss)
        test_logs.append(copy.deepcopy(run.get_log()))
    merged_test = merge_logs(test_logs) if len(test_logs)>1 else test_logs[0]
    logger.info(f"Computed {len(merged_test[0])} test gradients")

    # 5. Load saved FHE results
    saved = torch.load(args.fhe_results, map_location="cpu")
    fhe_inf   = saved["influence"]               # numpy array or tensor
    if isinstance(fhe_inf, torch.Tensor): fhe_inf = fhe_inf.numpy()
    tgt_ids   = saved["tgt_ids"]
    src_ids   = saved.get("src_ids", None)
    logger.info(f"Loaded FHE scores of shape {fhe_inf.shape}, {len(tgt_ids)} succeeded")

    # 6. Recompute with plaintext influence function and compare
    plain_calc = InfluenceFunction(state=run.state)
    damping = args.damping if args.damping is not None else run.config.influence.damping
    t0 = time.time()
    plain_res = plain_calc.compute_influence_all(
        src_log=merged_test,
        loader=log_loader,
        damping=damping,
        hessian=args.hessian
    )
    t1 = time.time()
    plain_inf = plain_res["influence"].numpy()
    plain_ids = plain_res["tgt_ids"]
    logger.info(f"Plaintext compute done in {t1-t0:.1f}s, shape={plain_inf.shape}")

    # 7. Align common successful samples and compute metrics
    set_fhe   = set(map(str, tgt_ids))
    set_plain = set(map(str, plain_ids))
    common    = sorted(list(set_fhe & set_plain))
    logger.info(f"{len(common)} common training IDs to compare")

    # Extract aligned 1D vectors
    # FHE side:
    if fhe_inf.ndim == 2 and fhe_inf.shape[0]>=1:
        fhe_vec = fhe_inf[0]
    else:
        fhe_vec = fhe_inf
    # Plaintext side:
    if plain_inf.ndim == 2 and plain_inf.shape[0]>=1:
        plain_vec = plain_inf[0]
    else:
        plain_vec = plain_inf

    # Index by common IDs
    idx_fhe   = {str(v):i for i,v in enumerate(tgt_ids)}
    idx_plain = {str(v):i for i,v in enumerate(plain_ids)}
    fhe_vals  = np.array([fhe_vec[idx_fhe[c]]     for c in common])
    plain_vals= np.array([plain_vec[idx_plain[c]] for c in common])

    # Compute metrics
    from scipy.stats import pearsonr
    if np.std(fhe_vals)>1e-9 and np.std(plain_vals)>1e-9:
        corr, pval = pearsonr(fhe_vals, plain_vals)
    else:
        corr, pval = np.nan, np.nan
    max_diff  = np.max(np.abs(fhe_vals-plain_vals))
    mean_diff = np.mean(np.abs(fhe_vals-plain_vals))

    logger.info("---- FINAL VALIDATION ----")
    logger.info(f"Pearson Correlation = {corr:.8f} (p={pval:.3e})")
    logger.info(f"Max abs diff = {max_diff:.3e}")
    logger.info(f"Mean abs diff = {mean_diff:.3e}")

    print("\n✅ BERT FHE-vs-Plain comparison complete.")

if __name__ == "__main__":
    main()
