# Sell Data to AI Algorithms Without Revealing It

This repository contains the code for replicating the experiments in:

> Michael Yang, Ruijiang Gao, Zhiqiang (Eric) Zheng
>
> **[Sell Data to AI Algorithms Without Revealing It: Secure Data Valuation and Sharing via Homomorphic Encryption](https://arxiv.org/abs/2512.06033)**

We introduce the **Trustworthy Influence Protocol (TIP)**, a privacy-preserving framework that enables data buyers to quantify the utility of external data *without ever seeing the raw assets*. TIP integrates Fully Homomorphic Encryption (CKKS) with gradient-based influence functions, built on top of the [LogIX](https://github.com/logix-project/logix) library for scalable gradient logging and Hessian approximation, allowing precise, blind scoring of data against a buyer's specific AI model.

## Quick Start: Minimal Replication

The fastest way to verify the TIP is through our minimal replication:

```bash
pip install torch torchvision pyfhel numpy
cd experiments/minimal_replication
bash run.sh
```

This self-contained script trains a small MLP on MNIST digits {1, 2}, treats digits {1, 2, 3} as three data sellers, computes plaintext influence scores, verifies them under CKKS encryption, and confirms that the encrypted rankings match the actual utility (measured by fine-tuning). Expected output: a comparison table showing near-perfect agreement between plaintext IF, encrypted IF, and ground-truth benefit.

## Repository Structure

```
logix/                          Core library (influence functions + FHE engine)
experiments/
  minimal_replication/          Self-contained FHE-IF proof-of-concept (start here)
  mnist/                        MLP on MNIST (Sec. 4.1)
  bert/                         BERT on SST-2 (Sec. 4.2)
  language_modeling/            GPT-2 on WikiText-2 (Sec. 4.3)
  Market_demo/                  Causal validity: 3-seller MNIST market (Sec. 5)
  Market_demo_hospital/         Healthcare data market simulation (Sec. 5)
  Market_demo_book/             Book data market simulation (Sec. 5)
```

## Experiments

We evaluate TIP's fidelity and scalability across three tasks of increasing complexity:

| Experiment | Model | Dataset | Purpose |
|---|---|---|---|
| **MNIST / MLP** | 3-layer MLP | MNIST | Controlled baseline; verifies cryptographic precision without approximation |
| **BERT / SST-2** | BERT-base | SST-2 (GLUE) | Tests compatibility with Transformer attention layers and KFAC preconditioners |
| **GPT-2 / WikiText** | GPT-2 | WikiText-2 | Assesses scalability to autoregressive LLMs and high-dimensional parameter spaces |

In all experiments, lightweight LoGra (Low-rank Gradient) adapters compress influence-relevant gradients into a homomorphically tractable representation. Only these compressed gradients are encrypted, shared, or processed under homomorphic operations. The per-sample cost depends primarily on the projected gradient dimension rather than the underlying model size.

We also include three data market simulations that demonstrate end-to-end secure data valuation:

| Simulation | Setup | Key Finding |
|---|---|---|
| **Causal Validity** | 3 MNIST sellers (digits 1, 2, 3) | Encrypted IF rankings match ground-truth utility |
| **Healthcare Market** | 54 Maryland hospitals, readmission prediction | IF significantly outperforms cosine similarity for seller ranking |
| **Book Market** | 1,000 BookCorpus sellers, OpenELM buyer | Reveals concentrated value distribution; most books do not help |

## Dependencies

Core:
- Python 3.9+
- PyTorch
- NumPy, SciPy, tqdm
- [Pyfhel](https://github.com/ibarrond/Pyfhel) (CKKS homomorphic encryption)

Additional (for BERT / GPT-2 / market simulations):
- Transformers, Datasets, Accelerate (Hugging Face)


## License

Apache 2.0
