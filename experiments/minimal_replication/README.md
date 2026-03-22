# Minimal Replication: CKKS FHE + Influence Function (IF)

This folder is a compact end-to-end demo of a privacy-aware data market:

1. Train a buyer model on MNIST digits `{1,2}`.
2. Define buyer utility as test loss on digit `{3}`.
3. Treat digits `{1,2,3}` as three sellers.
4. For each seller:
   - compute plaintext IF (`IF_plain`),
   - verify a subset with CKKS (`IF_fhe_mean(subset)`),
   - fine-tune on seller data and measure real utility (`delta_loss`).
5. Check whether IF ranking matches true utility ranking.

## Main script

- `minimal_FHE-IF_calculation.py`

## Dependencies

```bash
pip install torch torchvision pyfhel numpy
```

## Run

```bash
bash run.sh
```

or:

```bash
python minimal_FHE-IF_calculation.py \
  --data_root ./data \
  --buyer_digits 1,2 \
  --seller_digits 1,2,3 \
  --test_digit 3 \
  --buyer_train_per_digit 200 \
  --seller_samples 32 \
  --eval_samples 128 \
  --eval_grad_samples 16 \
  --damping 1e-2
```

## How to read the output

- `IF_plain_mean`: average plaintext influence score for one seller.
- `IF_fhe_mean(subset)`: CKKS-based IF on a verification subset.
- `abs_diff`: absolute gap between plaintext and FHE subset means.
- `delta_loss = loss_after - loss_before`: negative means the seller helps.
- `benefit = -delta_loss`: larger is better.
- `helpful_IF = -IF_plain_mean`: IF-based utility proxy (larger is better).

In a typical successful run:
- seller `3` has the largest `helpful_IF` and the largest positive `benefit`,
- Pearson/Spearman between `helpful_IF` and `benefit` are high (often near `1.0` in this tiny setup),
- `abs_diff` is very small, showing FHE verification is numerically consistent on the checked subset.

## Notes

- This is a minimal sanity-check pipeline, not the full large-scale experiment.
- For full experiments, see:
  - `experiments/mnist`
  - `experiments/bert`
  - `experiments/language_modeling`
  - `experiments/Market_demo*`
