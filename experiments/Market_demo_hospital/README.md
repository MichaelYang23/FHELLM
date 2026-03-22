# Healthcare Data Market (HSCRC)

This experiment simulates a data market using Maryland HSCRC inpatient records. A buyer hospital trains a binary classifier for revisit risk prediction and uses TIP to evaluate whether acquiring data from seller hospitals would improve its model. We compare three valuation tiers: **FHE-IF** (second-order, Hessian-weighted), **Gradient-Cosine** (first-order), and **Data-Cosine** (model-agnostic), against ground-truth utility obtained from controlled adaptation.

**Paper reference:** Section 4.3.1 (Healthcare Data Market).

## Setup

- **Dataset:** HSCRC inpatient discharge records (1.29M rows across 54 hospitals). Each record is encoded into a 5,273-dimensional feature vector via `HSCRCFeatureEncoder`.
- **Buyer:** One hospital with N_train=2,000 and N_eval=2,000 labeled encounters. Trains an MLP to predict `LABEL_UP` (revisit risk).
- **Sellers:** Up to 10 peer hospitals, each contributing 2,000 encounters.
- **Replicates:** 50 independent market simulations with random hospital assignments.

## Pipeline

### 1. Run simulation

```bash
python market_if_vs_deltaLoss.py \
  --root <path_to_hospital_parquet_files> \
  --logs_dir ./logs_corr \
  --results_tag run1 \
  --reps 50 \
  --N_train 2000 --N_eval 2000 \
  --N_sellers 10 --seller_K 2000 \
  --epochs 5 --lr 1e-2 --momentum 0.9 \
  --adapt_mode head --adapt_epochs 1 --adapt_lr 1e-3 \
  --seller_batch 32 --test_M 256 --log_batch 4 \
  --damping 1e-5 --hessian kfac \
  --seed 0
```

For each replicate: trains buyer baseline, computes IF and cosine scores per seller, adapts on each seller's data, and records delta-loss. Outputs: `market_results_<tag>.csv`, `market_correlations_<tag>.csv`.

### 2. Analyze correlations

```bash
python analyze_ranks_and_plot.py \
  --results_csv ./logs_corr/market_results_run1.csv \
  --outdir ./logs_corr/summary_corrs \
  --nbins 8 --bootstrap 10000 --weight_by_n \
  --seed 42
```

Computes Fisher-z pooled correlations with bootstrap CIs and paired improvement tests. Outputs: summary CSVs and per-rep/aggregated correlation plots.

## Key Results (50 reps, 378 buyer-seller pairs)

| Metric | FHE-IF | Grad-Cosine | Data-Cosine | Random |
| --- | --- | --- | --- | --- |
| \|Pearson\| | 0.959 | 0.913 | 0.285 | 0.354 |
| \|Spearman\| | 0.900 | 0.855 | 0.311 | 0.352 |

FHE-IF consistently outperforms gradient-cosine similarity (Pearson lift +0.048, p < 0.001).

## Files

| File | Purpose |
| --- | --- |
| `market_if_vs_deltaLoss.py` | Core simulation: IF + cosine scoring + adaptation |
| `analyze_ranks_and_plot.py` | Correlation analysis and plotting |
| `analyze_and_plot_corrs.py` | Additional correlation summaries |
| `utils.py` | MLP construction, data utilities |
| `utils_hscrc.py` | HSCRC feature encoder (5,273-dim sparse vector) |
