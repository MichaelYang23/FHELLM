# BERT / SST-2 -- Transformer Scalability

This experiment tests the Trustworthy Influence Protocol (TIP) on a modern Transformer architecture. BERT-base is fine-tuned on SST-2 (binary sentiment classification), and influence scores are computed both in plaintext and under CKKS encryption. The experiment evaluates the efficacy of KFAC preconditioners in handling complex attention layers.

**Paper reference:** Section 4.2.2 (BERT on SST-2).

## Setup

- **Model:** BERT-base-cased, fine-tuned for sequence classification (2 labels).
- **Dataset:** SST-2 from the GLUE benchmark (movie review sentiment).
- **LoGra adapters** (rank 8) attached to all attention and feed-forward blocks, yielding 4,676-dimensional projected gradients.
- **FHE scheme:** CKKS via Pyfhel (n=32768, scale=2^40).
- **Hessian:** KFAC approximation with damping 1e-5.

## Pipeline

### Step 1: Fine-tune BERT

```bash
python train.py --data_name sst2
```

Fine-tunes BERT-base-cased for 3 epochs with AdamW (lr=2e-5). Checkpoints saved to `files/checkpoints/0/sst2_epoch_*.pt`.

### Step 2: Extract training gradients

```bash
python extract_log.py \
  --project sst2 \
  --config_path ./config_lorarank_4_n_2**15_scale_2**40.yaml \
  --data_name sst2 \
  --lora random \
  --hessian kfac \
  --save grad
```

Logs per-sample gradients and KFAC covariance statistics for the full training set via LogIX.

### Step 3a: Plaintext influence scores

```bash
python compute_influence.py \
  --project sst2 \
  --config_path ./config_lorarank_4_n_2**15_scale_2**40.yaml \
  --data_name sst2
```

Computes influence scores using the saved training logs. Saves results to `if_logix.pt`.

### Step 3b: FHE influence scores

```bash
python compute_influence_fhe.py \
  --project sst2 \
  --config ./config_lorarank_4_n_2**15_scale_2**40.yaml \
  --data_name sst2 \
  --checkpoint files/checkpoints/0/sst2_epoch_3.pt \
  --hessian kfac \
  --lora
```

Runs the same computation under CKKS encryption. Reports Pearson correlation vs. plaintext and saves FHE results.

### Step 4: Post-hoc validation (optional)

```bash
python analyze_fhe_results.py \
  --project sst2 \
  --config ./config_lorarank_4_n_2**15_scale_2**40.yaml \
  --data_name sst2 \
  --checkpoint files/checkpoints/0/sst2_epoch_3.pt \
  --fhe_results <path_to_fhe_results.pt> \
  --hessian kfac \
  --lora
```

Standalone script to compare pre-computed FHE results against a fresh plaintext computation. Reports Pearson correlation, max and mean absolute differences.

## Expected Results

Pearson correlation between FHE and plaintext scores: ~0.97. Mean absolute error: ~1.84e-5.

## Files

| File | Purpose |
| --- | --- |
| `train.py` | Fine-tune BERT-base on SST-2 |
| `extract_log.py` | Log training gradients and KFAC statistics via LogIX |
| `compute_influence.py` | Plaintext influence function computation |
| `compute_influence_fhe.py` | FHE-encrypted influence function computation |
| `analyze_fhe_results.py` | Post-hoc FHE vs. plaintext comparison |
| `qualitative_analysis.py` | Extract and display top-k influential training samples |
| `extract_single_comparison.py` | Single-sample influence comparison utility |
| `utils.py` | Model construction, tokenizer, GLUE data loaders |
| `config_lorarank_4_n_2**15_scale_2**40.yaml` | Config with CKKS scale 2^40 (recommended) |
| `config_lorarank_4_n_2**15_scale_2**30.yaml` | Config with CKKS scale 2^30 (lower precision) |
