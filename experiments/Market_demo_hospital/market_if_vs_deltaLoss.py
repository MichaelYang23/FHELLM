#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, time, math, argparse, shutil
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.optim import SGD

# SciPy is optional
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# project-local utils
from utils_hscrc import (
    set_seed,
    build_file_map,
    list_index_tuples_for_hospital,
    HSCRCFeatureEncoder,
    make_loader_from_tuples,
    construct_mlp_tabular,
)

from logix import LogIX, LogIXScheduler
from logix.utils import DataIDGenerator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def disable_inplace_activations(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.SELU, nn.SiLU)):
            if hasattr(m, "inplace") and m.inplace:
                m.inplace = False

def flatten_grads(model: nn.Module) -> torch.Tensor:
    parts = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            parts.append(torch.zeros_like(p, device=DEVICE).view(-1))
        else:
            parts.append(p.grad.detach().view(-1))
    if not parts:
        return torch.zeros(0, device=DEVICE)
    return torch.cat(parts, dim=0)

@torch.no_grad()
def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    num = torch.dot(a, b).item()
    denom = (a.norm() * b.norm()).item()
    if denom < eps:
        return 0.0
    return num / denom

@torch.no_grad()
def eval_loss(model: nn.Module, loader) -> float:
    model.eval().to(DEVICE)
    ce_sum = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        total_loss += ce_sum(logits, yb).item()
        total += yb.size(0)
    if total == 0:
        return float("nan")
    return total_loss / total

def adapt_once(model, train_loader, mode="head", epochs=1, lr=1e-3, weight_decay=0.0, clip_grad=1.0):
    """Tiny fine-tune on seller_K samples."""
    if mode not in ("head","full","none"):
        raise ValueError("adapt_mode must be 'head'|'full'|'none'")
    if mode == "none":
        return model
    model = model.to(DEVICE)
    model.train()

    if mode == "head":
        for p in model.parameters():
            p.requires_grad = False
        last_lin = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                last_lin = m
        if last_lin is not None:
            for p in last_lin.parameters():
                p.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable params for adaptation.")
    opt = SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    ce = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = ce(out, yb)
            loss.backward()
            if clip_grad and clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=clip_grad)
            opt.step()
    model.eval()
    return model


def _extract_path_from_record(rec):
    """Normalize build_file_map() outputs to a plain path string."""
    if isinstance(rec, str):
        return rec
    if isinstance(rec, dict):
        for k in ("fp","path","file","npz"):
            v = rec.get(k, None)
            if isinstance(v, str):
                return v
    return None

def normalize_file_map(file_map_raw: Dict[str, object]) -> Dict[str, str]:
    fm = {}
    bad = []
    for hosp, rec in file_map_raw.items():
        p = _extract_path_from_record(rec)
        if isinstance(p, str):
            fm[hosp] = p
        else:
            bad.append((hosp, rec))
    if bad:
        hosp0, rec0 = bad[0]
        raise TypeError(
            f"build_file_map() returned non-path record for hosp={hosp0}. "
            f"Type={type(rec0)} keys={list(rec0.keys()) if isinstance(rec0, dict) else None}."
        )
    return fm

def list_first_k_tuples(file_map: Dict[str,str], hosp: str, K: int) -> List[Tuple[str,int]]:
    tuples = list_index_tuples_for_hospital(file_map, hosp)
    return tuples[: min(K, len(tuples))]


def compute_mean_test_grad(model, eval_tuples, file_map, encoder, batch=128) -> torch.Tensor:
    ce_sum = nn.CrossEntropyLoss(reduction="sum")
    model.eval().to(DEVICE)
    loader = make_loader_from_tuples(file_map, eval_tuples, encoder, batch_size=batch, shuffle=False)
    g_accum = None
    n_seen = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        for i in range(xb.size(0)):
            x = xb[i:i+1]; y = yb[i:i+1]
            model.zero_grad(set_to_none=True)
            out = model(x)
            loss = ce_sum(out, y)
            loss.backward()
            gi = flatten_grads(model)
            if g_accum is None:
                g_accum = gi.clone()
            else:
                g_accum.add_(gi)
            n_seen += 1
    if g_accum is None or n_seen == 0:
        raise RuntimeError("Empty eval set for test-grad.")
    g_accum.div_(float(n_seen))
    return g_accum.detach()

@torch.no_grad()
def compute_mean_test_feature(eval_tuples, file_map, encoder, batch=128) -> torch.Tensor:
    loader = make_loader_from_tuples(file_map, eval_tuples, encoder, batch_size=batch, shuffle=False)
    x_accum = None
    n_seen = 0
    for xb, _ in loader:
        xb = xb.to(DEVICE)
        if x_accum is None:
            x_accum = xb.sum(dim=0)
        else:
            x_accum.add_(xb.sum(dim=0))
        n_seen += xb.size(0)
    if x_accum is None or n_seen == 0:
        raise RuntimeError("Empty eval set for test-feature.")
    x_accum.div_(float(n_seen))
    return x_accum.detach().view(-1)


def compute_if_logix_for_seller(
    model: nn.Module,
    *,
    project_stub: str,
    buyer_train_tuples: List[Tuple[str,int]],
    buyer_eval_tuples: List[Tuple[str,int]],
    seller_tuples: List[Tuple[str,int]],
    file_map: Dict[str,str],
    encoder: HSCRCFeatureEncoder,
    seller_batch: int = 32,
    log_batch: int = 4,
    test_M: int = 256,
    damping: float = 1e-5,
    hessian: str = "kfac",
    logix_config: Optional[str] = "./config.yaml",
    seed: int = 0,
    keep_logs: bool = False,
) -> Optional[np.ndarray]:
    set_seed(seed)
    disable_inplace_activations(model)
    model.eval().to(DEVICE)

    # seller loader (fixed batch, drop_last to keep shapes consistent)
    if len(seller_tuples) < seller_batch:
        # undersized seller; skip
        return None

    seller_full = (len(seller_tuples) // seller_batch) * seller_batch
    seller_tuples = seller_tuples[:seller_full]
    if seller_full == 0:
        return None

    seller_loader = make_loader_from_tuples(
        file_map=file_map, tuples=seller_tuples, encoder=encoder,
        batch_size=seller_batch, shuffle=False, num_workers=0, pin_memory=False, drop_last=True
    )

    logix = LogIX(project=project_stub, config=logix_config)
    id_gen = DataIDGenerator()
    logix.watch(model)

    warm_loader = make_loader_from_tuples(
        file_map=file_map, tuples=buyer_train_tuples, encoder=encoder,
        batch_size=seller_batch, shuffle=False, num_workers=0, pin_memory=False, drop_last=True
    )
    scheduler = LogIXScheduler(logix, lora="none", hessian=hessian, save="none")
    for _ in scheduler:
        for xb, yb in warm_loader:
            with logix(data_id=id_gen(xb)):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                model.zero_grad(set_to_none=True)
                out = model(xb)
                loss = torch.nn.functional.cross_entropy(out, yb, reduction="sum")
                loss.backward()
    logix.finalize()

    scheduler2 = LogIXScheduler(logix, lora="none", hessian="none", save="grad")
    for _ in scheduler2:
        for xb, yb in seller_loader:
            with logix(data_id=id_gen(xb)):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                model.zero_grad(set_to_none=True)
                out = model(xb)
                loss = torch.nn.functional.cross_entropy(out, yb, reduction="sum")
                loss.backward()
    logix.finalize()

    log_loader = logix.build_log_dataloader(batch_size=log_batch, num_workers=0, flatten=False)

    test_batch = seller_batch
    avail = len(buyer_eval_tuples)
    if avail == 0:
        proj_dir = os.path.join("logix_logs", project_stub)
        if not keep_logs:
            shutil.rmtree(proj_dir, ignore_errors=True)
        return None

    M = min(test_M, avail)
    M = (M // test_batch) * test_batch

    eval_tuples_for_test = None
    if M >= test_batch:
        eval_tuples_for_test = buyer_eval_tuples[:M]
    else:
        rng_local = np.random.default_rng(abs(hash((seed, project_stub))) % (2**32))
        take = rng_local.integers(0, avail, size=test_batch)
        eval_tuples_for_test = [buyer_eval_tuples[i] for i in take]
        M = test_batch  

    test_loader = make_loader_from_tuples(
        file_map=file_map, tuples=eval_tuples_for_test, encoder=encoder,
        batch_size=test_batch, shuffle=False, num_workers=0, pin_memory=False, drop_last=True
    )

    logix.setup({"grad": ["log"]})
    logix.eval()
    xb, yb = next(iter(test_loader))
    with logix(data_id=id_gen(xb)):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        model.zero_grad(set_to_none=True)
        out = model(xb)
        loss = torch.nn.functional.cross_entropy(out, yb, reduction="sum")
        loss.backward()
    test_log = logix.get_log()

    result = logix.influence.compute_influence_all(test_log, log_loader, damping=damping)
    key = "influence" if "influence" in result else ("total" if "total" in result else None)
    proj_dir = os.path.join("logix_logs", project_stub)

    if key is None:
        if not keep_logs:
            shutil.rmtree(proj_dir, ignore_errors=True)
        return None

    vec = result[key].detach().cpu()
    if vec.dim() == 2: 
        vec = vec.mean(dim=0)
    arr = vec.contiguous().view(-1).numpy()

    if not keep_logs:
        shutil.rmtree(proj_dir, ignore_errors=True)

    return arr


def main():
    ap = argparse.ArgumentParser("IF (K-FAC) & CosSim vs ΔLoss (market demo)")
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--logs_dir", type=str, required=True)


    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--N_train", type=int, default=2000)    
    ap.add_argument("--N_eval", type=int, default=2000)   
    ap.add_argument("--N_sellers", type=int, default=10)
    ap.add_argument("--seller_K", type=int, default=2000)

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--batch_train", type=int, default=256)
    ap.add_argument("--batch_eval", type=int, default=512)

    ap.add_argument("--adapt_mode", type=str, default="head", choices=["head","full","none"])
    ap.add_argument("--adapt_epochs", type=int, default=1)
    ap.add_argument("--adapt_lr", type=float, default=1e-3)
    ap.add_argument("--adapt_wd", type=float, default=0.0)
    ap.add_argument("--clip_grad", type=float, default=1.0)

    ap.add_argument("--seller_batch", type=int, default=32)
    ap.add_argument("--test_M", type=int, default=256)
    ap.add_argument("--log_batch", type=int, default=4)
    ap.add_argument("--damping", type=float, default=1e-5)
    ap.add_argument("--hessian", type=str, default="kfac")
    ap.add_argument("--logix_config", type=str, default="./config.yaml")

    ap.add_argument("--keep_logix", action="store_true",
                    help="Keep per-(rep,seller) LogIX project folders (default: delete after IF).")

    ap.add_argument("--results_tag", type=str, default="",
                    help="Append a tag to output CSV names, e.g., '1031' -> market_results_1031.csv")

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.logs_dir, exist_ok=True)
    set_seed(args.seed)

    file_map_raw = build_file_map(args.root)
    file_map = normalize_file_map(file_map_raw)

    all_hosps = sorted(list(file_map.keys()))
    if len(all_hosps) < (2 + args.N_sellers):
        raise RuntimeError("Not enough hospitals for the chosen replicate design.")

    enc = HSCRCFeatureEncoder()

    tag = (f"_{args.results_tag}" if (args.results_tag or args.results_tag == "0") else "")
    per_row_csv = os.path.join(args.logs_dir, f"market_results{tag}.csv")
    corr_csv    = os.path.join(args.logs_dir, f"market_correlations{tag}.csv")
    need_hdr1 = not os.path.exists(per_row_csv)
    need_hdr2 = not os.path.exists(corr_csv)

    f_res = open(per_row_csv, "a", newline="")
    w_res = csv.DictWriter(f_res, fieldnames=[
        "rep","buyer_train_hosp","buyer_eval_hosp","seller","K",
        "if_mean","if_sum","if_std","if_min","if_max",
        "cos_mean","data_cos_mean","loss0","loss1","delta_loss",
        "seller_batch","test_M","log_batch","damping","hessian",
        "adapt_mode","adapt_epochs","adapt_lr","adapt_wd","clip_grad","seed"
    ])
    if need_hdr1: w_res.writeheader()

    f_cor = open(corr_csv, "a", newline="")
    w_cor = csv.DictWriter(f_cor, fieldnames=[
        "rep",
        "pearson_if_vs_dloss","pearson_if_p",
        "pearson_cos_vs_dloss","pearson_cos_p",
        "pearson_data_cos_vs_dloss","pearson_data_cos_p",
        "spearman_if_vs_dloss","spearman_if_p",
        "spearman_cos_vs_dloss","spearman_cos_p",
        "spearman_data_cos_vs_dloss","spearman_data_cos_p",
        "spearman_if_ci_lo","spearman_if_ci_hi",
        "spearman_cos_ci_lo","spearman_cos_ci_hi",
        "spearman_data_cos_ci_lo","spearman_data_cos_ci_hi",
        "N_pairs"
    ])
    if need_hdr2: w_cor.writeheader()

    rng = np.random.default_rng(args.seed)

    for rep in range(args.reps):
        choices = rng.choice(all_hosps, size=(2 + args.N_sellers), replace=False)
        buyer_train_hosp = str(choices[0])
        buyer_eval_hosp  = str(choices[1])
        seller_hosps     = [str(x) for x in choices[2:]]

        buyer_pretrain = list_first_k_tuples(file_map, buyer_train_hosp, args.N_train)
        buyer_eval     = list_first_k_tuples(file_map, buyer_eval_hosp,  args.N_eval)

        model = construct_mlp_tabular(input_dim=enc.input_dim, num_classes=2, seed=args.seed).to(DEVICE)
        disable_inplace_activations(model)

        ce = nn.CrossEntropyLoss()
        opt = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        train_loader = make_loader_from_tuples(file_map, buyer_pretrain, enc,
                                               batch_size=args.batch_train, shuffle=True)
        eval_loader  = make_loader_from_tuples(file_map, buyer_eval,     enc,
                                               batch_size=args.batch_eval,  shuffle=False)

        for _ in range(args.epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(set_to_none=True)
                out = model(xb)
                loss = ce(out, yb)
                loss.backward()
                opt.step()

        model.eval()
        loss0 = eval_loss(model, eval_loader)

        g_test = compute_mean_test_grad(model, buyer_eval, file_map, enc, batch=args.batch_eval)
        x_test = compute_mean_test_feature(buyer_eval, file_map, enc, batch=args.batch_eval)

        base_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        per_seller_rows = []
        for sid in seller_hosps:
            seller_tuples = list_first_k_tuples(file_map, sid, args.seller_K)
            if len(seller_tuples) < args.seller_K:
                continue

            if_vec = compute_if_logix_for_seller(
                model=model,
                project_stub=f"market_rep{rep}_{sid}",
                buyer_train_tuples=buyer_pretrain,
                buyer_eval_tuples=buyer_eval,
                seller_tuples=seller_tuples,
                file_map=file_map,
                encoder=enc,
                seller_batch=args.seller_batch,
                log_batch=args.log_batch,
                test_M=args.test_M,
                damping=args.damping,
                hessian=args.hessian,
                logix_config=args.logix_config,
                seed=args.seed,
                keep_logs=args.keep_logix,
            )

            if if_vec is None:
                continue

            if_mean = float(np.mean(if_vec))
            if_sum  = float(np.sum(if_vec))
            if_std  = float(np.std(if_vec))
            if_min  = float(np.min(if_vec))
            if_max  = float(np.max(if_vec))

            cs_list = []
            data_cs_list = []
            seller_loader_id = make_loader_from_tuples(
                file_map=file_map, tuples=seller_tuples, encoder=enc,
                batch_size=args.seller_batch, shuffle=False, num_workers=0, pin_memory=False
            )
            ce_sum = nn.CrossEntropyLoss(reduction="sum")
            for xb, yb in seller_loader_id:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                for i in range(xb.size(0)):
                    x = xb[i:i+1]; y = yb[i:i+1]
                    data_cs_list.append(cosine_sim(x_test, x.view(-1)))
                    model.zero_grad(set_to_none=True)
                    out = model(x)
                    loss = ce_sum(out, y)
                    loss.backward()
                    gi = flatten_grads(model)
                    cs_list.append(cosine_sim(g_test, gi))
            cos_mean = float(np.mean(cs_list)) if len(cs_list) > 0 else float("nan")
            data_cos_mean = float(np.mean(data_cs_list)) if len(data_cs_list) > 0 else float("nan")

            adapted = construct_mlp_tabular(input_dim=enc.input_dim, num_classes=2, seed=args.seed).to(DEVICE)
            adapted.load_state_dict(base_state, strict=True)
            disable_inplace_activations(adapted)

            seller_train_loader = make_loader_from_tuples(
                file_map=file_map, tuples=seller_tuples, encoder=enc,
                batch_size=args.batch_train, shuffle=True, num_workers=0, pin_memory=False
            )
            adapted = adapt_once(
                adapted, seller_train_loader,
                mode=args.adapt_mode, epochs=args.adapt_epochs,
                lr=args.adapt_lr, weight_decay=args.adapt_wd, clip_grad=args.clip_grad
            )
            loss1 = eval_loss(adapted, eval_loader)
            dloss = float(loss1 - loss0) if (loss0 == loss0 and loss1 == loss1) else float("nan")

            row = {
                "rep": int(rep),
                "buyer_train_hosp": buyer_train_hosp,
                "buyer_eval_hosp":  buyer_eval_hosp,
                "seller": sid,
                "K": int(args.seller_K),
                "if_mean": if_mean,
                "if_sum":  if_sum,
                "if_std":  if_std,
                "if_min":  if_min,
                "if_max":  if_max,
                "cos_mean": cos_mean,
                "data_cos_mean": data_cos_mean,
                "loss0": float(loss0),
                "loss1": float(loss1),
                "delta_loss": dloss,
                "seller_batch": int(args.seller_batch),
                "test_M": int(args.test_M),
                "log_batch": int(args.log_batch),
                "damping": float(args.damping),
                "hessian": args.hessian,
                "adapt_mode": args.adapt_mode,
                "adapt_epochs": int(args.adapt_epochs),
                "adapt_lr": float(args.adapt_lr),
                "adapt_wd": float(args.adapt_wd),
                "clip_grad": float(args.clip_grad),
                "seed": int(args.seed),
            }
            w_res.writerow(row); f_res.flush()
            per_seller_rows.append(row)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if len(per_seller_rows) >= 3:
            df = pd.DataFrame(per_seller_rows)
            x = df["delta_loss"].values
            y_if  = df["if_mean"].values
            y_cos = df["cos_mean"].values
            y_data_cos = df["data_cos_mean"].values

            def _pearson_with_p(a, b):
                if SCIPY_AVAILABLE:
                    r, p = stats.pearsonr(a, b)
                    return float(r), float(p)
                r = float(np.corrcoef(a, b)[0,1])
                n = len(a)
                if np.isfinite(r) and n > 3:
                    t = r * math.sqrt((n-2)/(1-r*r + 1e-12))
                    from math import erf, sqrt
                    p = 2.0*(1.0 - 0.5*(1+erf(abs(t)/sqrt(2))))
                else:
                    p = float("nan")
                return r, p

            pear_if_r,  pear_if_p  = _pearson_with_p(x, y_if)
            pear_cs_r,  pear_cs_p  = _pearson_with_p(x, y_cos)
            pear_dcs_r, pear_dcs_p = _pearson_with_p(x, y_data_cos)

            def _spearman_with_p(a, b):
                if SCIPY_AVAILABLE:
                    rho, p = stats.spearmanr(a, b)
                    return float(rho), float(p)
                ra = pd.Series(a).rank(method="average").values
                rb = pd.Series(b).rank(method="average").values
                rho = float(np.corrcoef(ra, rb)[0,1])
                # simple permutation p-value
                B = 1000
                cnt = 0
                for _ in range(B):
                    perm = np.random.permutation(rb)
                    rho_perm = float(np.corrcoef(ra, perm)[0,1])
                    if abs(rho_perm) >= abs(rho):
                        cnt += 1
                p = (cnt+1)/(B+1)
                return rho, p

            spe_if_r,  spe_if_p  = _spearman_with_p(x, y_if)
            spe_cs_r,  spe_cs_p  = _spearman_with_p(x, y_cos)
            spe_dcs_r, spe_dcs_p = _spearman_with_p(x, y_data_cos)

            def _spearman_bootstrap_ci(a, b, B=1000, alpha=0.05):
                rng_local = np.random.default_rng(12345)
                n = len(a)
                vals = []
                for _ in range(B):
                    idx = rng_local.integers(0, n, size=n)
                    aa = a[idx]; bb = b[idx]
                    if SCIPY_AVAILABLE:
                        r, _ = stats.spearmanr(aa, bb)
                    else:
                        ra = pd.Series(aa).rank(method="average").values
                        rb = pd.Series(bb).rank(method="average").values
                        r = float(np.corrcoef(ra, rb)[0,1])
                    vals.append(r)
                lo = float(np.percentile(vals, 2.5))
                hi = float(np.percentile(vals, 97.5))
                return lo, hi

            if len(df) >= 4:
                ci_if_lo,  ci_if_hi  = _spearman_bootstrap_ci(x, y_if)
                ci_cs_lo,  ci_cs_hi  = _spearman_bootstrap_ci(x, y_cos)
                ci_dcs_lo, ci_dcs_hi = _spearman_bootstrap_ci(x, y_data_cos)
            else:
                ci_if_lo = ci_if_hi = float("nan")
                ci_cs_lo = ci_cs_hi = float("nan")
                ci_dcs_lo = ci_dcs_hi = float("nan")

            w_cor.writerow({
                "rep": rep,
                "pearson_if_vs_dloss": pear_if_r,  "pearson_if_p": pear_if_p,
                "pearson_cos_vs_dloss": pear_cs_r, "pearson_cos_p": pear_cs_p,
                "pearson_data_cos_vs_dloss": pear_dcs_r, "pearson_data_cos_p": pear_dcs_p,
                "spearman_if_vs_dloss": spe_if_r,  "spearman_if_p": spe_if_p,
                "spearman_cos_vs_dloss": spe_cs_r, "spearman_cos_p": spe_cs_p,
                "spearman_data_cos_vs_dloss": spe_dcs_r, "spearman_data_cos_p": spe_dcs_p,
                "spearman_if_ci_lo": ci_if_lo,  "spearman_if_ci_hi": ci_if_hi,
                "spearman_cos_ci_lo": ci_cs_lo, "spearman_cos_ci_hi": ci_cs_hi,
                "spearman_data_cos_ci_lo": ci_dcs_lo, "spearman_data_cos_ci_hi": ci_dcs_hi,
                "N_pairs": int(len(df)),
            })
            f_cor.flush()

        print(f"[rep {rep}] finished. Sellers used: {len(per_seller_rows)}")

    f_res.close(); f_cor.close()

    dt = time.time() - t0
    print(f"\nAll done. Results:")
    print(f"  Rows → {per_row_csv}")
    print(f"  Corr → {corr_csv}")
    print(f"Total time: {dt/60:.2f} min.")
    return


if __name__ == "__main__":
    main()
