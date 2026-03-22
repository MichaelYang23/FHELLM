#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import argparse
import numpy as np
import pandas as pd

from typing import Dict, List, Tuple
try:
    from scipy import stats
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


def fisher_z(r: float, eps: float = 1e-7) -> float:
    r = np.clip(r, -1 + eps, 1 - eps)
    return 0.5 * np.log((1 + r) / (1 - r))

def inv_fisher_z(z: float) -> float:
    return (np.exp(2*z) - 1) / (np.exp(2*z) + 1)

def pearsonr_safe(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 2:
        return np.nan, np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    if SCIPY_OK:
        r, p = stats.pearsonr(x, y)
        return float(r), float(p)
    # fallback
    x0 = (x - x.mean()) / (x.std() + 1e-12)
    y0 = (y - y.mean()) / (y.std() + 1e-12)
    r = float(np.mean(x0 * y0))
    return r, np.nan

def spearmanr_safe(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 2:
        return np.nan, np.nan
    if SCIPY_OK:
        rho, p = stats.spearmanr(x, y)
        # scipy may return scalar or object; ensure float
        return float(rho), float(p)
    # fallback: rank then pearson
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    return pearsonr_safe(xr, yr)

def bootstrap_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=n)

def quantile_ci(samples: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    lo = float(np.quantile(samples, alpha/2))
    hi = float(np.quantile(samples, 1 - alpha/2))
    return lo, hi


def pooled_signed_with_ci(r_list: List[float], n_list: List[int], *,
                          B: int, weighted: bool, seed: int) -> Tuple[float, float, float]:
    """Return pooled r and 95% CI via bootstrap over reps."""
    r_arr = np.asarray(r_list, dtype=float)
    n_arr = np.asarray(n_list, dtype=float)
    ok = np.isfinite(r_arr)
    r_arr = r_arr[ok]
    n_arr = n_arr[ok]
    if r_arr.size == 0:
        return np.nan, np.nan, np.nan

    # observed pooled
    z = np.array([fisher_z(r) for r in r_arr])
    if weighted:
        w = n_arr / (n_arr.sum() + 1e-12)
        z_obs = float(np.sum(w * z))
    else:
        z_obs = float(np.mean(z))
    r_obs = inv_fisher_z(z_obs)

    # bootstrap
    rng = np.random.default_rng(seed)
    boots = []
    m = len(z)
    for _ in range(B):
        idx = rng.integers(0, m, size=m)
        z_b = z[idx]
        if weighted:
            w_b = n_arr[idx]
            w_b = w_b / (w_b.sum() + 1e-12)
            zb = float(np.sum(w_b * z_b))
        else:
            zb = float(np.mean(z_b))
        boots.append(inv_fisher_z(zb))
    lo, hi = quantile_ci(np.array(boots))
    return r_obs, lo, hi

def paired_delta_signed_fisherz_with_ci(r_if: List[float], r_cos: List[float], n: List[int], *,
                                        B: int, weighted: bool, seed: int) -> Tuple[float, float, float, float]:
    r_if = np.asarray(r_if, dtype=float)
    r_cos = np.asarray(r_cos, dtype=float)
    n = np.asarray(n, dtype=float)

    ok = np.isfinite(r_if) & np.isfinite(r_cos)
    r_if, r_cos, n = r_if[ok], r_cos[ok], n[ok]
    if r_if.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    z_if = np.array([fisher_z(r) for r in r_if])
    z_cos = np.array([fisher_z(r) for r in r_cos])

    if weighted:
        w = n / (n.sum() + 1e-12)
        z_if_obs = float(np.sum(w * z_if))
        z_cos_obs = float(np.sum(w * z_cos))
    else:
        z_if_obs = float(np.mean(z_if))
        z_cos_obs = float(np.mean(z_cos))
    d_obs = z_if_obs - z_cos_obs

    # bootstrap paired on reps
    rng = np.random.default_rng(seed)
    m = len(z_if)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, m, size=m)
        if weighted:
            w_b = n[idx]
            w_b = w_b / (w_b.sum() + 1e-12)
            zb_if = float(np.sum(w_b * z_if[idx]))
            zb_cos = float(np.sum(w_b * z_cos[idx]))
        else:
            zb_if = float(np.mean(z_if[idx]))
            zb_cos = float(np.mean(z_cos[idx]))
        boots.append(zb_if - zb_cos)
    boots = np.array(boots)
    lo, hi = quantile_ci(boots)
    # two-sided p by bootstrap
    p = 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))
    return float(d_obs), float(lo), float(hi), float(p)

def paired_delta_abs_with_ci(r_if: List[float], r_cos: List[float], n: List[int], *,
                             B: int, weighted: bool, seed: int) -> Tuple[float, float, float, float]:
    r_if = np.asarray(r_if, dtype=float)
    r_cos = np.asarray(r_cos, dtype=float)
    n = np.asarray(n, dtype=float)
    ok = np.isfinite(r_if) & np.isfinite(r_cos)
    r_if, r_cos, n = r_if[ok], r_cos[ok], n[ok]
    if r_if.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    abs_if = np.abs(r_if)
    abs_cos = np.abs(r_cos)

    # observed weighted mean of |r|
    if weighted:
        w = n / (n.sum() + 1e-12)
        obs = float(np.sum(w * (abs_if - abs_cos)))
    else:
        obs = float(np.mean(abs_if - abs_cos))

    # bootstrap over reps
    rng = np.random.default_rng(seed)
    m = len(abs_if)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, m, size=m)
        if weighted:
            w_b = n[idx]
            w_b = w_b / (w_b.sum() + 1e-12)
            boots.append(float(np.sum(w_b * (abs_if[idx] - abs_cos[idx]))))
        else:
            boots.append(float(np.mean(abs_if[idx] - abs_cos[idx])))
    boots = np.array(boots)
    lo, hi = quantile_ci(boots)
    p = 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))
    return obs, float(lo), float(hi), float(p)


def main():
    ap = argparse.ArgumentParser("Analyze per-rep correlations (IF vs Grad-Cos vs Data-Cos vs Random) and pool")
    ap.add_argument("--results_csv", type=str, required=True,
                    help="market_results.csv produced by market_if_vs_deltaLoss.py")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight_by_n", action="store_true", default=False,
                    help="Weight pooled estimates by number of sellers in each rep")
    ap.add_argument("--compare_abs", action="store_true", default=False,
                    help="Also do paired bootstrap on |r| (IF vs CosSim)")
    ap.add_argument("--unify_if_sign", action="store_true", default=True,
                    help="Flip IF by -1 so that larger means more beneficial")
    ap.add_argument("--unify_cos_sign", action="store_true", default=False,
                    help="Flip Cos by -1 if needed")
    ap.add_argument("--unify_data_cos_sign", action="store_true", default=False,
                    help="Flip data cosine by -1 if needed")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.results_csv)

    # Required columns
    for col in ["rep", "seller", "if_mean", "cos_mean", "delta_loss"]:
        if col not in df.columns:
            raise RuntimeError(f"Missing column '{col}' in {args.results_csv}")
    if "data_cos_mean" not in df.columns:
        df["data_cos_mean"] = np.nan

    # Scores & benefit (unified direction)
    score_if = df["if_mean"].astype(float).copy()
    score_cos = df["cos_mean"].astype(float).copy()
    score_data_cos = df["data_cos_mean"].astype(float).copy()
    benefit = (-df["delta_loss"].astype(float)).copy()  # more positive = better

    if args.unify_if_sign:
        score_if = -score_if
    if args.unify_cos_sign:
        score_cos = -score_cos
    if args.unify_data_cos_sign:
        score_data_cos = -score_data_cos

    df["_score_if"] = score_if
    df["_score_cos"] = score_cos
    df["_score_data_cos"] = score_data_cos
    df["_benefit"]  = benefit

    rows = []
    rng_global = np.random.default_rng(args.seed)
    for rep, g in df.groupby("rep", sort=True):
        x_if  = g["_score_if"].to_numpy()
        x_cos = g["_score_cos"].to_numpy()
        x_data_cos = g["_score_data_cos"].to_numpy()
        y_ben = g["_benefit"].to_numpy()

        rng_rep = np.random.default_rng(args.seed + int(rep) * 10007)
        x_rand = rng_rep.normal(size=len(g))

        pr_if,  pp_if  = pearsonr_safe(x_if,  y_ben)
        pr_cos, pp_cos = pearsonr_safe(x_cos, y_ben)
        pr_dcs, pp_dcs = pearsonr_safe(x_data_cos, y_ben)
        pr_rnd, pp_rnd = pearsonr_safe(x_rand, y_ben)

        sr_if,  sp_if  = spearmanr_safe(x_if,  y_ben)
        sr_cos, sp_cos = spearmanr_safe(x_cos, y_ben)
        sr_dcs, sp_dcs = spearmanr_safe(x_data_cos, y_ben)
        sr_rnd, sp_rnd = spearmanr_safe(x_rand, y_ben)

        rows.append({
            "rep": int(rep),
            "n_sellers": int(len(g)),
            "pearson_if": pr_if, "pearson_cos": pr_cos, "pearson_data_cos": pr_dcs, "pearson_rand": pr_rnd,
            "spearman_if": sr_if, "spearman_cos": sr_cos, "spearman_data_cos": sr_dcs, "spearman_rand": sr_rnd,
            "pearson_if_p": pp_if, "pearson_cos_p": pp_cos, "pearson_data_cos_p": pp_dcs, "pearson_rand_p": pp_rnd,
            "spearman_if_p": sp_if, "spearman_cos_p": sp_cos, "spearman_data_cos_p": sp_dcs, "spearman_rand_p": sp_rnd,
        })

    per_rep = pd.DataFrame(rows).sort_values("rep").reset_index(drop=True)
    per_rep_csv = os.path.join(args.outdir, "per_rep_corrs.csv")
    per_rep.to_csv(per_rep_csv, index=False)

    r_if  = per_rep["pearson_if"].to_list()
    r_cos = per_rep["pearson_cos"].to_list()
    r_dcs = per_rep["pearson_data_cos"].to_list()
    r_rnd = per_rep["pearson_rand"].to_list()
    nrep  = per_rep["n_sellers"].to_list()

    pooled_p_if,  pooled_p_if_lo,  pooled_p_if_hi  = pooled_signed_with_ci(r_if,  nrep, B=args.bootstrap, weighted=args.weight_by_n, seed=args.seed+11)
    pooled_p_cos, pooled_p_cos_lo, pooled_p_cos_hi = pooled_signed_with_ci(r_cos, nrep, B=args.bootstrap, weighted=args.weight_by_n, seed=args.seed+12)
    pooled_p_dcs, pooled_p_dcs_lo, pooled_p_dcs_hi = pooled_signed_with_ci(r_dcs, nrep, B=args.bootstrap, weighted=args.weight_by_n, seed=args.seed+13)
    pooled_p_rnd, pooled_p_rnd_lo, pooled_p_rnd_hi = pooled_signed_with_ci(r_rnd, nrep, B=args.bootstrap, weighted=args.weight_by_n, seed=args.seed+14)

    d_z_obs, d_z_lo, d_z_hi, p_signed = paired_delta_signed_fisherz_with_ci(r_if, r_cos, nrep, B=args.bootstrap,
                                                                            weighted=args.weight_by_n, seed=args.seed+21)
    d_z_if_dcs, d_z_if_dcs_lo, d_z_if_dcs_hi, p_if_dcs = paired_delta_signed_fisherz_with_ci(r_if, r_dcs, nrep, B=args.bootstrap,
                                                                                               weighted=args.weight_by_n, seed=args.seed+22)
    d_z_dcs_cos, d_z_dcs_cos_lo, d_z_dcs_cos_hi, p_dcs_cos = paired_delta_signed_fisherz_with_ci(r_dcs, r_cos, nrep, B=args.bootstrap,
                                                                                                    weighted=args.weight_by_n, seed=args.seed+23)

    s_if  = per_rep["spearman_if"].to_list()
    s_cos = per_rep["spearman_cos"].to_list()
    s_dcs = per_rep["spearman_data_cos"].to_list()
    s_rnd = per_rep["spearman_rand"].to_list()

    pooled_s_if,  pooled_s_if_lo,  pooled_s_if_hi  = pooled_signed_with_ci(s_if,  nrep, B=args.bootstrap, weighted=args.weight_by_n, seed=args.seed+31)
    pooled_s_cos, pooled_s_cos_lo, pooled_s_cos_hi = pooled_signed_with_ci(s_cos, nrep, B=args.bootstrap, weighted=args.weight_by_n, seed=args.seed+32)
    pooled_s_dcs, pooled_s_dcs_lo, pooled_s_dcs_hi = pooled_signed_with_ci(s_dcs, nrep, B=args.bootstrap, weighted=args.weight_by_n, seed=args.seed+33)
    pooled_s_rnd, pooled_s_rnd_lo, pooled_s_rnd_hi = pooled_signed_with_ci(s_rnd, nrep, B=args.bootstrap, weighted=args.weight_by_n, seed=args.seed+34)

    d_s_obs, d_s_lo, d_s_hi, p_spearman = paired_delta_signed_fisherz_with_ci(s_if, s_cos, nrep, B=args.bootstrap,
                                                                              weighted=args.weight_by_n, seed=args.seed+41)
    d_s_if_dcs, d_s_if_dcs_lo, d_s_if_dcs_hi, p_s_if_dcs = paired_delta_signed_fisherz_with_ci(s_if, s_dcs, nrep, B=args.bootstrap,
                                                                                                 weighted=args.weight_by_n, seed=args.seed+42)
    d_s_dcs_cos, d_s_dcs_cos_lo, d_s_dcs_cos_hi, p_s_dcs_cos = paired_delta_signed_fisherz_with_ci(s_dcs, s_cos, nrep, B=args.bootstrap,
                                                                                                      weighted=args.weight_by_n, seed=args.seed+43)

    mean_abs_pearson_if  = float(np.nanmean(np.abs(per_rep["pearson_if"].to_numpy())))
    mean_abs_pearson_cos = float(np.nanmean(np.abs(per_rep["pearson_cos"].to_numpy())))
    mean_abs_pearson_dcs = float(np.nanmean(np.abs(per_rep["pearson_data_cos"].to_numpy())))
    mean_abs_pearson_rnd = float(np.nanmean(np.abs(per_rep["pearson_rand"].to_numpy())))

    mean_abs_spearman_if  = float(np.nanmean(np.abs(per_rep["spearman_if"].to_numpy())))
    mean_abs_spearman_cos = float(np.nanmean(np.abs(per_rep["spearman_cos"].to_numpy())))
    mean_abs_spearman_dcs = float(np.nanmean(np.abs(per_rep["spearman_data_cos"].to_numpy())))
    mean_abs_spearman_rnd = float(np.nanmean(np.abs(per_rep["spearman_rand"].to_numpy())))

    if args.compare_abs:
        d_abs_p_obs, d_abs_p_lo, d_abs_p_hi, p_abs_p = paired_delta_abs_with_ci(
            per_rep["pearson_if"].to_list(), per_rep["pearson_cos"].to_list(), nrep,
            B=args.bootstrap, weighted=args.weight_by_n, seed=args.seed+51
        )
        d_abs_p_if_dcs, d_abs_p_if_dcs_lo, d_abs_p_if_dcs_hi, p_abs_p_if_dcs = paired_delta_abs_with_ci(
            per_rep["pearson_if"].to_list(), per_rep["pearson_data_cos"].to_list(), nrep,
            B=args.bootstrap, weighted=args.weight_by_n, seed=args.seed+53
        )
        d_abs_s_obs, d_abs_s_lo, d_abs_s_hi, p_abs_s = paired_delta_abs_with_ci(
            per_rep["spearman_if"].to_list(), per_rep["spearman_cos"].to_list(), nrep,
            B=args.bootstrap, weighted=args.weight_by_n, seed=args.seed+52
        )
        d_abs_s_if_dcs, d_abs_s_if_dcs_lo, d_abs_s_if_dcs_hi, p_abs_s_if_dcs = paired_delta_abs_with_ci(
            per_rep["spearman_if"].to_list(), per_rep["spearman_data_cos"].to_list(), nrep,
            B=args.bootstrap, weighted=args.weight_by_n, seed=args.seed+54
        )
    else:
        d_abs_p_obs = d_abs_p_lo = d_abs_p_hi = p_abs_p = np.nan
        d_abs_p_if_dcs = d_abs_p_if_dcs_lo = d_abs_p_if_dcs_hi = p_abs_p_if_dcs = np.nan
        d_abs_s_obs = d_abs_s_lo = d_abs_s_hi = p_abs_s = np.nan
        d_abs_s_if_dcs = d_abs_s_if_dcs_lo = d_abs_s_if_dcs_hi = p_abs_s_if_dcs = np.nan

    # save summary
    summary = {
        "pooled_pearson_if": pooled_p_if,
        "pooled_pearson_if_lo": pooled_p_if_lo,
        "pooled_pearson_if_hi": pooled_p_if_hi,
        "pooled_pearson_cos": pooled_p_cos,
        "pooled_pearson_cos_lo": pooled_p_cos_lo,
        "pooled_pearson_cos_hi": pooled_p_cos_hi,
        "pooled_pearson_data_cos": pooled_p_dcs,
        "pooled_pearson_data_cos_lo": pooled_p_dcs_lo,
        "pooled_pearson_data_cos_hi": pooled_p_dcs_hi,
        "pooled_pearson_rand": pooled_p_rnd,
        "pooled_pearson_rand_lo": pooled_p_rnd_lo,
        "pooled_pearson_rand_hi": pooled_p_rnd_hi,

        "pooled_spearman_if": pooled_s_if,
        "pooled_spearman_if_lo": pooled_s_if_lo,
        "pooled_spearman_if_hi": pooled_s_if_hi,
        "pooled_spearman_cos": pooled_s_cos,
        "pooled_spearman_cos_lo": pooled_s_cos_lo,
        "pooled_spearman_cos_hi": pooled_s_cos_hi,
        "pooled_spearman_data_cos": pooled_s_dcs,
        "pooled_spearman_data_cos_lo": pooled_s_dcs_lo,
        "pooled_spearman_data_cos_hi": pooled_s_dcs_hi,
        "pooled_spearman_rand": pooled_s_rnd,
        "pooled_spearman_rand_lo": pooled_s_rnd_lo,
        "pooled_spearman_rand_hi": pooled_s_rnd_hi,

        "delta_fisherz_pearson": d_z_obs,
        "delta_fisherz_pearson_lo": d_z_lo,
        "delta_fisherz_pearson_hi": d_z_hi,
        "p_pearson": p_signed,
        "delta_fisherz_pearson_if_minus_data_cos": d_z_if_dcs,
        "delta_fisherz_pearson_if_minus_data_cos_lo": d_z_if_dcs_lo,
        "delta_fisherz_pearson_if_minus_data_cos_hi": d_z_if_dcs_hi,
        "p_pearson_if_minus_data_cos": p_if_dcs,
        "delta_fisherz_pearson_data_cos_minus_cos": d_z_dcs_cos,
        "delta_fisherz_pearson_data_cos_minus_cos_lo": d_z_dcs_cos_lo,
        "delta_fisherz_pearson_data_cos_minus_cos_hi": d_z_dcs_cos_hi,
        "p_pearson_data_cos_minus_cos": p_dcs_cos,

        "delta_fisherz_spearman": d_s_obs,
        "delta_fisherz_spearman_lo": d_s_lo,
        "delta_fisherz_spearman_hi": d_s_hi,
        "p_spearman": p_spearman,
        "delta_fisherz_spearman_if_minus_data_cos": d_s_if_dcs,
        "delta_fisherz_spearman_if_minus_data_cos_lo": d_s_if_dcs_lo,
        "delta_fisherz_spearman_if_minus_data_cos_hi": d_s_if_dcs_hi,
        "p_spearman_if_minus_data_cos": p_s_if_dcs,
        "delta_fisherz_spearman_data_cos_minus_cos": d_s_dcs_cos,
        "delta_fisherz_spearman_data_cos_minus_cos_lo": d_s_dcs_cos_lo,
        "delta_fisherz_spearman_data_cos_minus_cos_hi": d_s_dcs_cos_hi,
        "p_spearman_data_cos_minus_cos": p_s_dcs_cos,

        "mean_abs_pearson_if": mean_abs_pearson_if,
        "mean_abs_pearson_cos": mean_abs_pearson_cos,
        "mean_abs_pearson_data_cos": mean_abs_pearson_dcs,
        "mean_abs_pearson_rand": mean_abs_pearson_rnd,
        "mean_abs_spearman_if": mean_abs_spearman_if,
        "mean_abs_spearman_cos": mean_abs_spearman_cos,
        "mean_abs_spearman_data_cos": mean_abs_spearman_dcs,
        "mean_abs_spearman_rand": mean_abs_spearman_rnd,

        "delta_abs_pearson_if_minus_cos": d_abs_p_obs,
        "delta_abs_pearson_if_minus_cos_lo": d_abs_p_lo,
        "delta_abs_pearson_if_minus_cos_hi": d_abs_p_hi,
        "p_abs_pearson": p_abs_p,
        "delta_abs_pearson_if_minus_data_cos": d_abs_p_if_dcs,
        "delta_abs_pearson_if_minus_data_cos_lo": d_abs_p_if_dcs_lo,
        "delta_abs_pearson_if_minus_data_cos_hi": d_abs_p_if_dcs_hi,
        "p_abs_pearson_if_minus_data_cos": p_abs_p_if_dcs,

        "delta_abs_spearman_if_minus_cos": d_abs_s_obs,
        "delta_abs_spearman_if_minus_cos_lo": d_abs_s_lo,
        "delta_abs_spearman_if_minus_cos_hi": d_abs_s_hi,
        "p_abs_spearman": p_abs_s,
        "delta_abs_spearman_if_minus_data_cos": d_abs_s_if_dcs,
        "delta_abs_spearman_if_minus_data_cos_lo": d_abs_s_if_dcs_lo,
        "delta_abs_spearman_if_minus_data_cos_hi": d_abs_s_if_dcs_hi,
        "p_abs_spearman_if_minus_data_cos": p_abs_s_if_dcs,

        "n_reps": int(per_rep.shape[0]),
        "weighted_total_n": int(per_rep["n_sellers"].sum()),
        "weighted": bool(args.weight_by_n),
        "bootstrap": int(args.bootstrap),
        "unify_if_sign": bool(args.unify_if_sign),
        "unify_cos_sign": bool(args.unify_cos_sign),
        "unify_data_cos_sign": bool(args.unify_data_cos_sign),
    }
    summary_csv = os.path.join(args.outdir, "summary_overall.csv")
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    def fmt_ci(v, lo, hi, sgn=False):
        if np.isnan(v): return "nan [nan, nan]"
        if sgn:
            return f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"

    txt = []
    txt.append("=== OVERALL (pooled, 95% CI) ===")
    txt.append(f"Pearson  IF:   {fmt_ci(pooled_p_if,  pooled_p_if_lo,  pooled_p_if_hi,  True)}")
    txt.append(f"Pearson  Cos:  {fmt_ci(pooled_p_cos, pooled_p_cos_lo, pooled_p_cos_hi, True)}")
    txt.append(f"Pearson  DataCos:  {fmt_ci(pooled_p_dcs, pooled_p_dcs_lo, pooled_p_dcs_hi, True)}")
    txt.append(f"Pearson  Rand: {fmt_ci(pooled_p_rnd, pooled_p_rnd_lo, pooled_p_rnd_hi, True)}")
    txt.append(f"Paired Δz (IF−Cos, Pearson): {d_z_obs:+.3f}  95% CI [{d_z_lo:+.3f}, {d_z_hi:+.3f}]  p={p_signed:.5f}")
    txt.append(f"Paired Δz (IF−DataCos, Pearson): {d_z_if_dcs:+.3f}  95% CI [{d_z_if_dcs_lo:+.3f}, {d_z_if_dcs_hi:+.3f}]  p={p_if_dcs:.5f}")
    txt.append(f"Paired Δz (DataCos−Cos, Pearson): {d_z_dcs_cos:+.3f}  95% CI [{d_z_dcs_cos_lo:+.3f}, {d_z_dcs_cos_hi:+.3f}]  p={p_dcs_cos:.5f}")
    txt.append("")
    txt.append(f"Spearman IF (rank):   {fmt_ci(pooled_s_if,  pooled_s_if_lo,  pooled_s_if_hi,  True)}")
    txt.append(f"Spearman Cos (rank):  {fmt_ci(pooled_s_cos, pooled_s_cos_lo, pooled_s_cos_hi, True)}")
    txt.append(f"Spearman DataCos (rank):  {fmt_ci(pooled_s_dcs, pooled_s_dcs_lo, pooled_s_dcs_hi, True)}")
    txt.append(f"Spearman Rand:        {fmt_ci(pooled_s_rnd, pooled_s_rnd_lo, pooled_s_rnd_hi, True)}")
    txt.append(f"Paired Δz (IF−Cos, Spearman): {d_s_obs:+.3f}  95% CI [{d_s_lo:+.3f}, {d_s_hi:+.3f}]  p={p_spearman:.5f}")
    txt.append(f"Paired Δz (IF−DataCos, Spearman): {d_s_if_dcs:+.3f}  95% CI [{d_s_if_dcs_lo:+.3f}, {d_s_if_dcs_hi:+.3f}]  p={p_s_if_dcs:.5f}")
    txt.append(f"Paired Δz (DataCos−Cos, Spearman): {d_s_dcs_cos:+.3f}  95% CI [{d_s_dcs_cos_lo:+.3f}, {d_s_dcs_cos_hi:+.3f}]  p={p_s_dcs_cos:.5f}")
    txt.append("")
    txt.append("Simple mean of |r| across reps (including Random):")
    txt.append(f"  mean |Pearson|   IF={mean_abs_pearson_if:.3f}, DataCos={mean_abs_pearson_dcs:.3f}, Cos={mean_abs_pearson_cos:.3f},  Rand={mean_abs_pearson_rnd:.3f}")
    txt.append(f"  mean |Spearman|  IF={mean_abs_spearman_if:.3f}, DataCos={mean_abs_spearman_dcs:.3f}, Cos={mean_abs_spearman_cos:.3f}, Rand={mean_abs_spearman_rnd:.3f}")
    if args.compare_abs:
        txt.append(f"Paired Δ mean |Pearson| (IF−Cos):  {d_abs_p_obs:+.3f}  95% CI [{d_abs_p_lo:+.3f}, {d_abs_p_hi:+.3f}]  p={p_abs_p:.5f}")
        txt.append(f"Paired Δ mean |Pearson| (IF−DataCos):  {d_abs_p_if_dcs:+.3f}  95% CI [{d_abs_p_if_dcs_lo:+.3f}, {d_abs_p_if_dcs_hi:+.3f}]  p={p_abs_p_if_dcs:.5f}")
        txt.append(f"Paired Δ mean |Spearman| (IF−Cos): {d_abs_s_obs:+.3f}  95% CI [{d_abs_s_lo:+.3f}, {d_abs_s_hi:+.3f}]  p={p_abs_s:.5f}")
        txt.append(f"Paired Δ mean |Spearman| (IF−DataCos): {d_abs_s_if_dcs:+.3f}  95% CI [{d_abs_s_if_dcs_lo:+.3f}, {d_abs_s_if_dcs_hi:+.3f}]  p={p_abs_s_if_dcs:.5f}")
    txt.append("")
    txt.append(f"n_reps={per_rep.shape[0]}  total_sellers={int(per_rep['n_sellers'].sum())}  weighted={args.weight_by_n}  B={args.bootstrap}")
    txt_path = os.path.join(args.outdir, "summary.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(txt))

    print("\n".join(txt))
    print(f"\nSaved per-rep CSV → {per_rep_csv}")
    print(f"Saved summary CSV  → {summary_csv}")
    print(f"Saved text summary → {txt_path}")

if __name__ == "__main__":
    main()
