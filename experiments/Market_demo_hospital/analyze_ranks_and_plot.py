#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# ---------- helpers: Fisher z pooling & bootstrap ----------

def fisher_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)

def fisher_inv(z):
    return np.tanh(z)

def pooled_fisher_z(rs, ns):
    rs = np.asarray(rs, float); ns = np.asarray(ns, int)
    m = np.isfinite(rs) & np.isfinite(ns) & (ns >= 4)
    if m.sum() == 0: return np.nan
    z = fisher_z(rs[m]); w = (ns[m] - 3).astype(float)
    zbar = np.sum(w * z) / np.sum(w)
    return float(fisher_inv(zbar))

def bootstrap_ci_over_reps(rs, ns, B=5000, seed=0):
    rng = np.random.default_rng(seed)
    rs = np.asarray(rs, float); ns = np.asarray(ns, int)
    idx = np.arange(len(rs))
    samples = []
    for _ in range(B):
        take = rng.choice(idx, size=len(idx), replace=True)
        samples.append(pooled_fisher_z(rs[take], ns[take]))
    lo, hi = np.nanpercentile(samples, [2.5, 97.5])
    return float(lo), float(hi)

def paired_bootstrap_delta(rs_a, rs_b, ns, B=5000, seed=0):
    rng = np.random.default_rng(seed)
    rs_a = np.asarray(rs_a, float); rs_b = np.asarray(rs_b, float); ns = np.asarray(ns, int)
    idx = np.arange(len(rs_a))
    diffs = []
    for _ in range(B):
        take = rng.choice(idx, size=len(idx), replace=True)
        rA = pooled_fisher_z(rs_a[take], ns[take])
        rB = pooled_fisher_z(rs_b[take], ns[take])
        diffs.append(fisher_z(rA) - fisher_z(rB))
    diffs = np.asarray(diffs)
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs >= 0).mean(), (diffs <= 0).mean())
    # point estimate on full set:
    delta_hat = fisher_z(pooled_fisher_z(rs_a, ns)) - fisher_z(pooled_fisher_z(rs_b, ns))
    return float(delta_hat), float(lo), float(hi), float(p)


def safe_corr(x, y, fn):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3: return np.nan, np.nan, 0
    r, p = fn(x[m], y[m])
    return float(r), float(p), int(m.sum())

def quantile_bins(all_vals, nbins):
    qs = np.linspace(0, 1, nbins + 1)
    edges = np.quantile(all_vals, qs)
    edges = np.unique(edges)
    if len(edges) < 3: 
        edges = np.linspace(np.min(all_vals), np.max(all_vals), nbins + 1)
    return edges

def bin_corrs_for_rep(df_rep, edges, rng, metric="pearson"):
    benefit = df_rep["benefit"].values
    x_if = df_rep["if_help"].values
    x_cos = df_rep["cos_mean"].values
    x_rand = rng.normal(size=len(benefit))

    bins = np.digitize(benefit, edges[1:-1], right=False)  # 0..(B-1)

    B = len(edges) - 1
    r_if = np.full(B, np.nan); r_cos = np.full(B, np.nan); r_rnd = np.full(B, np.nan)
    n_if = np.zeros(B, dtype=int)

    for b in range(B):
        m = bins == b
        if m.sum() < 3:
            n_if[b] = int(m.sum()); continue
        if metric == "pearson":
            r_if[b], _, _ = safe_corr(x_if[m],  benefit[m], pearsonr)
            r_cos[b], _, _ = safe_corr(x_cos[m], benefit[m], pearsonr)
            r_rnd[b], _, _ = safe_corr(x_rand[m], benefit[m], pearsonr)
        else:
            r_if[b], _, _ = safe_corr(x_if[m],  benefit[m], spearmanr)
            r_cos[b], _, _ = safe_corr(x_cos[m], benefit[m], spearmanr)
            r_rnd[b], _, _ = safe_corr(x_rand[m], benefit[m], spearmanr)
        n_if[b] = int(m.sum())

    return dict(r_if=np.abs(r_if), r_cos=np.abs(r_cos), r_rnd=np.abs(r_rnd), n=n_if)


def line_with_ci(ax, x, mean, lo, hi, label):
    ax.plot(x, mean, marker="o", label=label)
    ax.fill_between(x, lo, hi, alpha=0.2)

def plot_per_rep_lines(df_rep, edges, out_path_prefix, rep, rng):
    centers = 0.5*(edges[:-1] + edges[1:])
    # Pearson |r|
    res_p = bin_corrs_for_rep(df_rep, edges, rng, metric="pearson")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(centers, res_p["r_if"], marker="o", label="IF (|r|)")
    ax.plot(centers, res_p["r_cos"], marker="o", label="CosSim (|r|)")
    for x, n in zip(centers, res_p["n"]):
        ax.annotate(str(int(n)), (x, 0.02), fontsize=8, ha="center")
    ax.set_xlabel("Benefit = -ΔLoss")
    ax.set_ylabel("|Pearson r|")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"rep {rep}: |Pearson| vs Benefit (quantile bins)")
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(f"{out_path_prefix}_pearson_abs.png", dpi=200); plt.close(fig)

    # Spearman |rho|
    res_s = bin_corrs_for_rep(df_rep, edges, rng, metric="spearman")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(centers, res_s["r_if"], marker="o", label="IF (|ρ|)")
    ax.plot(centers, res_s["r_cos"], marker="o", label="CosSim (|ρ|)")
    for x, n in zip(centers, res_s["n"]):
        ax.annotate(str(int(n)), (x, 0.02), fontsize=8, ha="center")
    ax.set_xlabel("Benefit = -ΔLoss")
    ax.set_ylabel("|Spearman ρ|")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"rep {rep}: |Spearman| vs Benefit (quantile bins)")
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(f"{out_path_prefix}_spearman_abs.png", dpi=200); plt.close(fig)

def plot_aggregate_lines(per_rep_bin, edges, out_path, metric_name, weight_by_n, B=5000, seed=0):
    """
    per_rep_bin: list of dicts from bin_corrs_for_rep(metric=...)
    Bootstrap across reps to get mean ±95% CI per bin.
    """
    rng = np.random.default_rng(seed)
    centers = 0.5*(edges[:-1] + edges[1:])
    R = len(per_rep_bin); Bn = len(centers)

    arr_if  = np.stack([p["r_if"] for p in per_rep_bin])   # (R,B)
    arr_cos = np.stack([p["r_cos"] for p in per_rep_bin])
    Ns      = np.stack([p["n"]    for p in per_rep_bin]).astype(float)

    def boot_mean(arr):
        idx = np.arange(R)
        samples_if, samples_cos = [], []
        for _ in range(B):
            take = rng.choice(idx, size=len(idx), replace=True)
            a_if  = arr_if[take]   # (R,B)
            a_cos = arr_cos[take]
            n     = Ns[take]
            if weight_by_n:
                w = np.where(np.isfinite(a_if), n, 0.0)
                m_if  = np.nansum(w * a_if,  axis=0) / np.clip(np.nansum(w, axis=0), 1e-9, None)
                w = np.where(np.isfinite(a_cos), n, 0.0)
                m_cos = np.nansum(w * a_cos, axis=0) / np.clip(np.nansum(w, axis=0), 1e-9, None)
            else:
                m_if  = np.nanmean(a_if,  axis=0)
                m_cos = np.nanmean(a_cos, axis=0)
            samples_if.append(m_if); samples_cos.append(m_cos)
        s_if  = np.stack(samples_if)  
        s_cos = np.stack(samples_cos)
        mean_if  = np.nanmean(s_if,  axis=0); lo_if  = np.nanpercentile(s_if,  2.5, axis=0); hi_if  = np.nanpercentile(s_if, 97.5, axis=0)
        mean_cos = np.nanmean(s_cos, axis=0); lo_cos = np.nanpercentile(s_cos, 2.5, axis=0); hi_cos = np.nanpercentile(s_cos,97.5, axis=0)
        return mean_if, lo_if, hi_if, mean_cos, lo_cos, hi_cos

    mean_if, lo_if, hi_if, mean_cos, lo_cos, hi_cos = boot_mean(arr_if)

    totalN = np.nansum(Ns, axis=0)

    fig, ax = plt.subplots(figsize=(9,6))
    line_with_ci(ax, centers, mean_if,  lo_if,  hi_if,  "IF (|r|)")
    line_with_ci(ax, centers, mean_cos, lo_cos, hi_cos, "CosSim (|r|)")
    for x, n in zip(centers, totalN):
        ax.annotate(str(int(n)), (x, 0.02), fontsize=9, ha="center")
    ax.set_xlabel("Benefit = -ΔLoss")
    ax.set_ylabel(f"|{metric_name}|")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Aggregate across reps  (shaded = 95% CI; numbers = per-bin N)")
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--nbins", type=int, default=8)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--weight_by_n", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    per_rep_dir = os.path.join(args.outdir, "per_rep_corr_bins"); os.makedirs(per_rep_dir, exist_ok=True)

    df = pd.read_csv(args.results_csv)
    if not {"rep","seller","delta_loss","if_mean","cos_mean"}.issubset(df.columns):
        raise ValueError("results_csv must contain columns: rep, seller, delta_loss, if_mean, cos_mean")

    # unify directions
    df["benefit"]  = -df["delta_loss"]
    df["if_help"]  = -df["if_mean"]   # larger = more helpful
    df["cos_mean"] = df["cos_mean"]   # larger = more helpful

    per_rows, rank_rows = [], []
    rng = np.random.default_rng(args.seed)

    edges = quantile_bins(df["benefit"].values, args.nbins)

    # for aggregate line CIs
    per_rep_bins_pearson, per_rep_bins_spearman = [], []

    for rep, g in df.groupby("rep"):
        g = g.copy().reset_index(drop=True)

        # random baseline for this rep
        x_rand = rng.normal(size=len(g))

        # Pearson (values)
        r_if,  p_if,  n_if  = safe_corr(g["if_help"].values,  g["benefit"].values, pearsonr)
        r_cos, p_cos, n_cos = safe_corr(g["cos_mean"].values, g["benefit"].values, pearsonr)
        r_rnd, p_rnd, n_rnd = safe_corr(x_rand,               g["benefit"].values, pearsonr)

        # ranks (rank 1 = best)
        g["rank_benefit"] = (-g["benefit"]).rank(method="average", ascending=False)
        g["rank_if"]      = (-g["if_help"]).rank(method="average", ascending=False)
        g["rank_cos"]     = (-g["cos_mean"]).rank(method="average", ascending=False)
        g["rank_rand"]    = pd.Series(x_rand).rank(method="average", ascending=False)

        s_if,  sp_if,  _ = safe_corr(g["rank_if"].values,   g["rank_benefit"].values, spearmanr)
        s_cos, sp_cos, _ = safe_corr(g["rank_cos"].values,  g["rank_benefit"].values, spearmanr)
        s_rnd, sp_rnd, _ = safe_corr(g["rank_rand"].values, g["rank_benefit"].values, spearmanr)

        per_rows.append(dict(
            rep=int(rep), n_sellers=int(len(g)),
            pearson_if=r_if,  pearson_if_p=p_if,
            pearson_cos=r_cos, pearson_cos_p=p_cos,
            pearson_rand=r_rnd, pearson_rand_p=p_rnd,
            spearman_if=s_if,  spearman_if_p=sp_if,
            spearman_cos=s_cos, spearman_cos_p=sp_cos,
            spearman_rand=s_rnd, spearman_rand_p=sp_rnd
        ))

        # save ranks (optional CSV for debugging / inspection)
        for _, r in g.iterrows():
            rank_rows.append(dict(
                rep=int(rep), seller=r["seller"],
                benefit=float(r["benefit"]),
                rank_benefit=float(r["rank_benefit"]),
                if_help=float(r["if_help"]), rank_if=float(r["rank_if"]),
                cos_mean=float(r["cos_mean"]), rank_cos=float(r["rank_cos"])
            ))

        # per-rep per-bin plots (two figures per rep)
        out_prefix = os.path.join(per_rep_dir, f"rep_{int(rep)}")
        plot_per_rep_lines(g, edges, out_prefix, rep=int(rep), rng=rng)

        # store for aggregate CIs
        per_rep_bins_pearson.append(bin_corrs_for_rep(g, edges, rng, metric="pearson"))
        per_rep_bins_spearman.append(bin_corrs_for_rep(g, edges, rng, metric="spearman"))

    per_rep = pd.DataFrame(per_rows).sort_values("rep").reset_index(drop=True)
    per_rep.to_csv(os.path.join(args.outdir, "per_rep_corrs.csv"), index=False)
    pd.DataFrame(rank_rows).to_csv(os.path.join(args.outdir, "per_rep_ranks.csv"), index=False)



    ns = per_rep["n_sellers"].values

    # pooled Pearson
    pooled_if  = pooled_fisher_z(per_rep["pearson_if"].values,  ns)
    pooled_cos = pooled_fisher_z(per_rep["pearson_cos"].values, ns)
    pooled_rnd = pooled_fisher_z(per_rep["pearson_rand"].values, ns)
    lo_if,  hi_if  = bootstrap_ci_over_reps(per_rep["pearson_if"].values,  ns, B=args.bootstrap, seed=args.seed)
    lo_cos, hi_cos = bootstrap_ci_over_reps(per_rep["pearson_cos"].values, ns, B=args.bootstrap, seed=args.seed+1)
    lo_rnd, hi_rnd = bootstrap_ci_over_reps(per_rep["pearson_rand"].values, ns, B=args.bootstrap, seed=args.seed+2)
    dP, dP_lo, dP_hi, pP = paired_bootstrap_delta(per_rep["pearson_if"].values,
                                                  per_rep["pearson_cos"].values, ns,
                                                  B=args.bootstrap, seed=args.seed+3)

    # pooled Spearman (rank corr)
    pooled_s_if  = pooled_fisher_z(per_rep["spearman_if"].values,  ns)
    pooled_s_cos = pooled_fisher_z(per_rep["spearman_cos"].values, ns)
    pooled_s_rnd = pooled_fisher_z(per_rep["spearman_rand"].values, ns)
    slo_if,  shi_if  = bootstrap_ci_over_reps(per_rep["spearman_if"].values,  ns, B=args.bootstrap, seed=args.seed+4)
    slo_cos, shi_cos = bootstrap_ci_over_reps(per_rep["spearman_cos"].values, ns, B=args.bootstrap, seed=args.seed+5)
    slo_rnd, shi_rnd = bootstrap_ci_over_reps(per_rep["spearman_rand"].values, ns, B=args.bootstrap, seed=args.seed+6)
    dS, dS_lo, dS_hi, pS = paired_bootstrap_delta(per_rep["spearman_if"].values,
                                                  per_rep["spearman_cos"].values, ns,
                                                  B=args.bootstrap, seed=args.seed+7)

    # simple means of absolute correlations (intuitive)
    mean_abs_pearson_if  = np.nanmean(np.abs(per_rep["pearson_if"].values))
    mean_abs_pearson_cos = np.nanmean(np.abs(per_rep["pearson_cos"].values))
    mean_abs_spearman_if  = np.nanmean(np.abs(per_rep["spearman_if"].values))
    mean_abs_spearman_cos = np.nanmean(np.abs(per_rep["spearman_cos"].values))

    # write summary CSV + TXT
    summary = dict(
        pooled_pearson_if=pooled_if,  pooled_pearson_if_lo=lo_if,  pooled_pearson_if_hi=hi_if,
        pooled_pearson_cos=pooled_cos, pooled_pearson_cos_lo=lo_cos, pooled_pearson_cos_hi=hi_cos,
        pooled_pearson_rand=pooled_rnd, pooled_pearson_rand_lo=lo_rnd, pooled_pearson_rand_hi=hi_rnd,
        pooled_spearman_if=pooled_s_if,  pooled_spearman_if_lo=slo_if,  pooled_spearman_if_hi=shi_if,
        pooled_spearman_cos=pooled_s_cos, pooled_spearman_cos_lo=slo_cos, pooled_spearman_cos_hi=shi_cos,
        pooled_spearman_rand=pooled_s_rnd, pooled_spearman_rand_lo=slo_rnd, pooled_spearman_rand_hi=shi_rnd,
        delta_fisherz_pearson=dP, delta_fisherz_pearson_lo=dP_lo, delta_fisherz_pearson_hi=dP_hi, p_pearson=pP,
        delta_fisherz_spearman=dS, delta_fisherz_spearman_lo=dS_lo, delta_fisherz_spearman_hi=dS_hi, p_spearman=pS,
        mean_abs_pearson_if=mean_abs_pearson_if, mean_abs_pearson_cos=mean_abs_pearson_cos,
        mean_abs_spearman_if=mean_abs_spearman_if, mean_abs_spearman_cos=mean_abs_spearman_cos,
        n_reps=int(per_rep.shape[0]), weighted_total_n=int(ns.sum())
    )
    pd.DataFrame([summary]).to_csv(os.path.join(args.outdir, "summary_overall.csv"), index=False)

    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write("=== OVERALL (pooled, 95% CI) ===\n")
        f.write(f"Pearson  IF:  {pooled_if:+.3f} [{lo_if:+.3f}, {hi_if:+.3f}]\n")
        f.write(f"Pearson  Cos: {pooled_cos:+.3f} [{lo_cos:+.3f}, {hi_cos:+.3f}]\n")
        f.write(f"Pearson  Rand:{pooled_rnd:+.3f} [{lo_rnd:+.3f}, {hi_rnd:+.3f}]\n")
        f.write(f"Paired Δz (IF−Cos, Pearson): {dP:+.3f}  95% CI [{dP_lo:+.3f}, {dP_hi:+.3f}]  p={pP:.3g}\n")
        f.write(f"Spearman IF (rank):  {pooled_s_if:+.3f} [{slo_if:+.3f}, {shi_if:+.3f}]\n")
        f.write(f"Spearman Cos (rank): {pooled_s_cos:+.3f} [{slo_cos:+.3f}, {shi_cos:+.3f}]\n")
        f.write(f"Spearman Rand:       {pooled_s_rnd:+.3f} [{slo_rnd:+.3f}, {shi_rnd:+.3f}]\n")
        f.write(f"Paired Δz (IF−Cos, Spearman): {dS:+.3f}  95% CI [{dS_lo:+.3f}, {dS_hi:+.3f}]  p={pS:.3g}\n")
        f.write("\nSimple mean of |r| across reps:\n")
        f.write(f"  mean |Pearson|  IF={mean_abs_pearson_if:.3f}, Cos={mean_abs_pearson_cos:.3f}\n")
        f.write(f"  mean |Spearman| IF={mean_abs_spearman_if:.3f}, Cos={mean_abs_spearman_cos:.3f}\n")

    # ---- aggregate per-bin lines with 95% CI across reps
    plot_aggregate_lines(per_rep_bins_pearson,  edges,
                         os.path.join(args.outdir, "agg_corr_bins_pearson_abs.png"),
                         metric_name="Pearson r", weight_by_n=args.weight_by_n,
                         B=args.bootstrap, seed=args.seed+10)
    plot_aggregate_lines(per_rep_bins_spearman, edges,
                         os.path.join(args.outdir, "agg_corr_bins_spearman_abs.png"),
                         metric_name="Spearman ρ", weight_by_n=args.weight_by_n,
                         B=args.bootstrap, seed=args.seed+11)

    print("Done. Wrote per-rep stats, pooled numbers, and all figures to:", args.outdir)

if __name__ == "__main__":
    main()
