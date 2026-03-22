# MNIST / MLP -- Fidelity Verification

This experiment validates the numerical fidelity of CKKS-encrypted influence scores against their plaintext counterparts using a simple MLP on MNIST. Because the model is small enough to compute exact influence scores without approximation, this setup serves as a controlled baseline for verifying the cryptographic layer.

**Paper reference:** Section 4.1 (Approximation Consistency) and Section 4.2.1 (MLP with MNIST).

## Setup

- **Model:** 3-layer MLP (784 -> 512 -> 256 -> 10, no bias, ReLU activations).
- **Dataset:** MNIST, subsampled to 6,000 training examples.
- **LoGra adapters** compress the full gradient (~500k params) to an 8,292-dimensional vector for encryption.
- **FHE scheme:** CKKS via Pyfhel.

## Pipeline

### Step 1: Train the MLP

```bash
python train.py
```

Trains 10 models (seeds 0-9) for 10 epochs each with SGD (lr=0.1, momentum=0.9). Checkpoints are saved to `checkpoints/`. Expected test accuracy: ~95.5%.

### Step 2a: Plaintext influence scores

```bash
python compute_influences.py \
  --data mnist \
  --lora random \
  --hessian raw \
  --save grad
```

Logs training gradients via LogIX, computes influence scores for specified test samples, and saves results to `if_logix.pt`.

### Step 2b: FHE influence scores

```bash
python compute_influences_fhe.py \
  --config config.yaml \
  --checkpoint checkpoints/mnist_0_epoch_9.pt \
  --hessian raw \
  --lora random
```

Performs the same influence computation under CKKS encryption. Saves encrypted-then-decrypted scores to `if_fhe_mnist.pt` and reports Pearson correlation and max absolute error vs. plaintext.

### Step 3: Compare

```bash
python compare.py
```

Loads both result files and prints Pearson correlation. Expected result: correlation ~1.0, mean absolute error ~2.16e-5.

## Files

| File | Purpose |
| --- | --- |
| `train.py` | Train MLP on MNIST (and Fashion-MNIST) |
| `compute_influences.py` | Plaintext influence function computation via LogIX |
| `compute_influences_fhe.py` | FHE-encrypted influence function computation |
| `compare.py` | Pearson correlation between plaintext and FHE scores |
| `utils.py` | Model construction, data loaders, seed utilities |
| `config.yaml` | LogIX / FHE configuration (LoRA rank, CKKS parameters) |
