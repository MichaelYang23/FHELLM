# GPT-2 / WikiText-2 -- LLM Scalability

This experiment assesses TIP's scalability to autoregressive language models. GPT-2 is used for next-token prediction on WikiText-2, and influence scores are computed under CKKS encryption. A surgical LoGra placement strategy (MLP-only, rank 4) compresses per-example gradients to a 384-dimensional vector, demonstrating that per-sample FHE cost depends on the projected gradient dimension rather than the underlying model size.

**Paper reference:** Section 4.2.3 (GPT-2 on WikiText-2).

## Setup

- **Model:** GPT-2 (124M parameters), causal language modeling.
- **Dataset:** WikiText-2 (2M+ tokens from Wikipedia articles), grouped into 512-token blocks.
- **LoGra adapters** (rank 4) applied only to feed-forward sublayers (`--mlp_only`), yielding a compact 384-dimensional projected gradient.
- **FHE scheme:** CKKS via Pyfhel (n=32768, scale=2^40).

## Pipeline

### Step 1: Extract training gradients

No separate training step is needed; we use the pre-trained GPT-2 weights from HuggingFace.

```bash
python extract_log.py \
  --config_path ./config_extract_log.yaml \
  --model_name gpt2 \
  --data_path wikitext \
  --data_name wikitext-2-v1 \
  --lora random \
  --hessian raw \
  --save grad \
  --mlp_only
```

Logs per-sample gradients and (optionally) Hessian statistics for the training split. Results are stored in the LogIX project directory.

### Step 2a: Plaintext influence scores

```bash
python compute_influence.py \
  --config_path ./config_compute_influence.yaml \
  --model_name gpt2 \
  --data_path wikitext \
  --data_name wikitext-2-v1 \
  --lora random \
  --hessian raw \
  --mlp_only
```

Computes influence scores using saved training logs and saves the result matrix to `./save/scores.pt`.

### Step 2b: FHE influence scores

```bash
python compute_influence_fhe_gpt2.py \
  --project <project_name> \
  --config ./config_compute_influence.yaml \
  --model_name gpt2 \
  --data_path wikitext \
  --data_name wikitext-2-v1 \
  --hessian raw \
  --lora random \
  --mlp_only
```

Runs the same computation under CKKS encryption. Validates against plaintext results (Pearson correlation, absolute error) unless `--skip_validation` is set. Saves FHE results to `if_fhe_gpt2_<project>.pt`.

### Step 3: Qualitative analysis (optional)

```bash
python qualitative_analysis.py \
  --score_path ./save/scores.pt \
  --model_name gpt2 \
  --data_path wikitext \
  --data_name wikitext-2-v1
```

Displays the top-k most influential training passages for each test query in human-readable form. Optionally compares two score files via `--score_path2`.

## Expected Results

Pearson correlation between FHE and plaintext scores: ~1.00. Mean absolute error: ~1.12e-5. Per-sample FHE time: ~0.15 s.

## Files

| File | Purpose |
| --- | --- |
| `extract_log.py` | Log training gradients via LogIX |
| `compute_influence.py` | Plaintext influence function computation |
| `compute_influence_fhe_gpt2.py` | FHE-encrypted influence function computation |
| `qualitative_analysis.py` | Display top-k influential training passages per test query |
| `utils.py` | Model loading (with Conv1D-to-Linear conversion for GPT-2), tokenizer, dataset utilities |
| `config_extract_log.yaml` | Config for gradient extraction (LoRA rank 8, damping 1e-5) |
| `config_compute_influence.yaml` | Config for influence computation with FHE parameters (CKKS n=32768, scale=2^40) |
