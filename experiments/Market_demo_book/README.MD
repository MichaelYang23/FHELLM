# Book Data Market (OpenELM + MuSR)

This experiment simulates a data market for generative AI training. An OpenELM-270M language model (the buyer) uses TIP to rank 1,000 books from BookCorpusOpen by their marginal utility for a downstream reasoning task (MuSR team allocation), without accessing the books' raw content. The results reveal extreme heterogeneity in data utility, challenging flat-rate compensation schemes.

**Paper reference:** Section 4.3.2 (Book Data Market Simulation).

## Setup

- **Buyer model:** `apple/OpenELM-270M` (FP16/BF16 via Accelerate).
- **Sellers:** Individual books from `lucadiliello/bookcorpusopen`, tokenized into 512-token blocks. Each block inherits the seller's book ID.
- **Evaluation task:** MuSR `team_allocation.json` -- multi-step reasoning about role assignment under constraints.
- **Influence function:** `IF(z_i, z_test) = -g_test^T H^{-1} g_i`, computed via LogIX with LoGra gradient compression.

## Pipeline

### Step 1: Log seller gradients

```bash
python step1_extract_log_books.py \
  --project openelm_if_books_raw_v1 \
  --model_name apple/OpenELM-270M \
  --use_injector \
  --block_size 512 \
  --batch_size 1 \
  --lora random \
  --hessian raw \
  --save grad \
  --name_filter attn ffn \
  --max_files 1000
```

Runs each seller's tokenized chunks through OpenELM and logs per-example gradients via LogIX. Uses `BookSellerDataset` (from `data_id_injector.py`) to assign deterministic `seller:<title>#chunk:<idx>` IDs.

### Step 2: Compute influence vs. MuSR

```bash
python step2_compute_influence_books_musr.py \
  --project openelm_if_books_raw_v1 \
  --musr_json ./MuSR_datasets/team_allocation.json \
  --max_items 8 \
  --model_name apple/OpenELM-270M \
  --name_filter attn ffn \
  --damping 1e-2 \
  --batch_size_src 64 \
  --out_dir ./results_books_1000sellers_musr
```

Builds test gradients from MuSR prompts, runs `compute_influence_all` against all seller logs, and aggregates per-chunk scores into per-book scores.

**Outputs:**
- `per_chunk_scores.pt/.tsv` -- per-chunk IF scores.
- `per_book_scores.tsv` -- aggregated per-book ranking (more negative = more helpful).

## Key Findings

The distribution of influence scores is highly skewed:
- **High utility (large negative IF):** Texts with explicit entity mapping and structured reasoning (e.g., theatrical scripts, translation pairs) align well with the MuSR constraint-satisfaction task.
- **Harmful (large positive IF):** Texts with archaic phrasing, humor-based logic, or misaligned technical jargon increase the buyer's evaluation loss.

This heterogeneity demonstrates that a meritocratic, utility-proportional licensing model can outperform flat-rate compensation.

## Files

| File | Purpose |
| --- | --- |
| `step1_extract_log_books.py` | Log seller gradients with LogIX |
| `step2_compute_influence_books_musr.py` | Compute and aggregate IF scores vs. MuSR |
| `data_id_injector.py` | Wraps BookCorpusOpen with deterministic seller IDs |
| `utils.py` | Tokenizer/model helpers, `BookSellerDataset` |
| `plot_tip_scores.py` | Visualize score distribution |
