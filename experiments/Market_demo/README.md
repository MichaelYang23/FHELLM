# Market Demo (MNIST) -- Approximation Consistency

This experiment demonstrates a privacy-preserving data marketplace on MNIST. A buyer model (trained on digits {1,2}) uses FHE-encrypted influence scores to rank sellers offering different digit data, without seeing the sellers' raw gradients. The demo validates that **FHE-IF rankings match the observed adaptation gains** (delta-accuracy / delta-loss).

**Paper reference:** Section 4.1 (Approximation Consistency).

## Scenario

- **Buyer** trains a 4-class MLP on digits {1,2} and wants to improve on digit 3.
- **Sellers:** SellerA ({1}), SellerB ({2}), SellerC ({3}).
- **Expected outcome:** SellerC dominates with the highest FHE-IF score and the largest observed delta-loss reduction.

## Pipeline

### 1. Partition data and train buyer

```bash
python 1_partition_and_train.py \
  --buyer_size 6000 --buyer_p1 0.30 --buyer_p2 0.70 \
  --seller_size 4000 \
  --eval_size 2000 --eval_digit 3 \
  --epochs 10 --seed 0
```

Creates disjoint partitions and trains a 4-class MLP. Outputs: `partitions/*.npy`, `checkpoints/buyer/*.pt`.

### 2. Log sellers + FHE scoring (recommended: single-run)

```bash
python combined_log_and_fhe.py \
  --project mnist_market_4class \
  --buyer_ckpt ./checkpoints/buyer/mlp4_seed0_epoch10.pt \
  --partitions ./partitions \
  --root ./data \
  --outdir ./logs \
  --eval_digit 3 \
  --use_lora --hessian kfac \
  --batch_eval 256 --batch_log 64 \
  --greedy_k 500 --select_mode max \
  --seed 0
```

Logs seller gradients with KFAC covariance, builds the plaintext mean test-gradient, and runs **one FHE pass** to produce IF scores. Outputs: `logs/scores_eval3_fhe.npy`, `logs/manifest.json`, `logs/purchase_plan.json`.

Alternatively, use the two-stage path: `2_log_sellers_and_build_testgrad.py` then `3_score_and_select_fast.py`.

### 3. Verify consistency

```bash
python 4_report_consistency.py \
  --buyer_ckpt ./checkpoints/buyer/mlp4_seed0_epoch10.pt \
  --partitions ./partitions \
  --logs_dir ./logs \
  --root ./data \
  --eval_digit 3 \
  --adapt_mode head --epochs 6 --lr 5e-3 --batch 256 \
  --seed 0
```

Performs one-shot adaptation per seller and reports per-seller IF aggregates alongside observed delta-accuracy / delta-loss. Output: `logs/if_vs_acc_report_single_eval.json`.

## Sign Convention

In this repo, **positive IF = predicted helpful** (flipped from textbook convention for readability).

## Files

| File | Purpose |
| --- | --- |
| `1_partition_and_train.py` | Build buyer/seller/eval partitions, train buyer MLP |
| `2_log_sellers_and_build_testgrad.py` | Log seller gradients + KFAC, build mean test-grad (stage 1) |
| `3_score_and_select_fast.py` | FHE pass over seller logs (stage 2) |
| `combined_log_and_fhe.py` | Single-run alternative combining stages 1 + 2 |
| `4_report_consistency.py` | Compare IF aggregates vs. observed adaptation gains |
| `run_replicates.py` | Run multiple replicates for statistical robustness |
| `utils.py` | Model, MNIST filtering/remap, dataloaders, seeding |
