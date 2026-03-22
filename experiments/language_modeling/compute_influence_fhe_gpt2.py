#!/usr/bin/env python3

import argparse
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm

import logix
from logix.analysis import InfluenceFunction, InfluenceFunctionFHE
from logix.utils import DataIDGenerator, get_logger, merge_logs

from examples.language_modeling.utils import get_model, get_tokenizer, get_loader

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True


def main():
    parser = argparse.ArgumentParser("GPT-2 FHE Influence Analysis")
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="LogIX project name (must match extract_log.py)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to LogIX config YAML (must include an `fhe:` section)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="HuggingFace model identifier, e.g. 'gpt2' or 'gpt2-xl'",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="HF cache directory (same as extract_log.py)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="wikitext",
        help="HF dataset path (e.g. 'wikitext')",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="wikitext-2-v1",
        help="HF dataset name (e.g. 'wikitext-2-v1')",
    )
    parser.add_argument(
        "--hessian",
        type=str,
        default="raw",
        choices=["auto", "kfac", "raw"],
        help="Hessian approximation (same as extract_log.py)",
    )
    parser.add_argument(
        "--lora",
        type=str,
        default="random",
        help="LoRA init method (same as extract_log.py; e.g. 'random' or 'none')",
    )
    parser.add_argument(
        "--mlp_only",
        action="store_true",
        help="Whether to insert LoRA only into MLP parts (same as extract_log.py)",
    )
    parser.add_argument(
        "--log_loader_batch_size",
        type=int,
        default=64,
        help="Batch size for loading saved training gradients (flatten=True)",
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="If set, skip plaintext-vs-FHE comparison at the end",
    )
    parser.add_argument("--split", type=str, default="train", help="Which split to extract logs from: train / validation / test")
    
    args = parser.parse_args()

    logger = get_logger()
    accelerator = Accelerator()

    # ---------------------------
    # 1. Load GPT-2 model + tokenizer
    # ---------------------------
    logger.info(f"Loading model {args.model_name} (Conv1D → Linear conversion if needed)")
    model = get_model(model_name=args.model_name, cache_dir=args.cache_dir)
    # When we call get_tokenizer for GPT-2, we want a pad token so that DataLoader collates properly
    tokenizer = get_tokenizer(
        model_name=args.model_name, 
        cache_dir=args.cache_dir, 
        add_padding_token=True
    )
    model.eval()

    # ---------------------------
    # 2. Initialize LogIX with FHE config
    # ---------------------------
    logger.info(f"Initializing LogIX project={args.project}, config={args.config}")
    run = logix.init(project=args.project, config=args.config)
    if run is None:
        # If init() returned None, we likely already initialized earlier in this process.
        logger.warning("logix.init() returned None; reusing existing instance")
        run = logix._LOGIX_INSTANCE
        run.config.load_config(args.config)

    # Ensure that `fhe:` exists in config
    if not hasattr(run.config, "fhe") or run.config.fhe is None:
        raise ValueError("Missing `fhe:` section in the provided config.yaml")
    logger.info(
        f"FHE parameters: n={run.config.fhe.n}, scale={run.config.fhe.scale}, qi_sizes={run.config.fhe.qi_sizes}"
    )

    # ---------------------------
    # 3. Load pre‐extracted logs back into LogIX state
    # ---------------------------
    logger.info("Loading existing logs into LogIX state …")
    run.initialize_from_log()
    logger.info(f"State loaded from {run.log_dir}")

    # ---------------------------
    # 4. Re‐insert LoRA (if any) and re‐watch the model
    # ---------------------------
    run.model = model

    # If the saved config used PCA and we don't have PCA data, force random
    if (
        args.lora != "none"
        and hasattr(run.config, "lora")
        and run.config.lora.init == "pca"
    ):
        logger.warning(
            "Config specified lora.init='pca', but PCA data is missing → forcing 'random'"
        )
        run.config.lora.init = "random"
        if hasattr(run, "lora_handler") and run.lora_handler is not None:
            run.lora_handler.init_strategy = "random"


    if args.mlp_only:
        # When extracting, also use mlp_only -> only record .mlp.c_fc/.mlp.c_proj
        run.watch(model, name_filter=["mlp"])
    else:
        # When extracting, name_filter=["att","mlp"]
        run.watch(model, name_filter=["att", "mlp"])

    
    if args.lora != "none":
        logger.info("Adding LoRA modules …")
        run.add_lora()
    else:
        logger.info("No LoRA modules inserted.")

    # Wrap the model with Accelerator
    model = accelerator.prepare(model)


    # ---------------------------
    # 5. Build DataLoader for saved training‐gradient logs (flatten=True)
    # ---------------------------
    logger.info("Building DataLoader for saved training gradients (flatten=True) …")
    log_loader = run.build_log_dataloader(
        batch_size=args.log_loader_batch_size, flatten=True
    )
    num_train = len(log_loader.dataset)
    logger.info(f"Loaded {num_train} training samples from the log")

    # ---------------------------
    # 6. Build validation/test DataLoader (Wikitext validation split, batch_size=1)
    # ---------------------------
    logger.info(f"Loading validation split of {args.data_path}/{args.data_name} …")
    test_loader = get_loader(
        model_name=args.model_name,
        tokenizer=tokenizer,
        batch_size=1,
        data_path=args.data_path,
        data_name=args.data_name,
        # split="validation",
        split=args.split,
        cache_dir=args.cache_dir,
    )
    test_loader = accelerator.prepare(test_loader)
    logger.info(f"Prepared test loader with {len(test_loader.dataset)} samples")

    # ---------------------------
    # 7. Instantiate FHE and Plaintext influence calculators
    # ---------------------------
    logger.info("Instantiating InfluenceFunctionFHE and InfluenceFunction …")
    influence_fhe = InfluenceFunctionFHE(state=run.state, fhe_config=run.config.fhe)
    influence_plain = InfluenceFunction(state=run.state)

    # ---------------------------
    # 8. Compute test gradients (one sample at a time, batch_size=1)
    # ---------------------------
    logger.info("Capturing test gradients …")
    run.eval()                              # Set LogIX in eval mode
    run.setup({"grad": ["log"]})            # Capture gradients in the log

    id_gen = DataIDGenerator(mode="hash")
    test_logs = []

    for batch in tqdm(test_loader, desc="Test Gradients"):
        # Each batch is a dict with keys: input_ids, attention_mask, labels (shifted)
        data_id = tokenizer.batch_decode(
            batch["input_ids"], skip_special_tokens=True
        )
        # Pop labels before passing batch to model
        targets = batch.pop("labels")

        with run(data_id=data_id, mask=batch["attention_mask"]):
            model.zero_grad()
            outputs = model(**batch)  # Now batch only has input_ids & attention_mask
            # GPT-2 returns [batch, seq_len, vocab_size]
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
                ignore_index=-100,
            )
            accelerator.backward(loss)

        # Save a deep copy of the log for this test sample
        test_logs.append(copy.deepcopy(run.get_log()))

    if not test_logs:
        raise RuntimeError("No test gradients were collected!")

    # Merge multiple test-sample logs into a single “merged_test_log”
    merged_test_log = merge_logs(test_logs)
    logger.info("Test gradient computation complete.")

    # ---------------------------
    # 9. Run FHE‐based influence computation
    # ---------------------------
    logger.info("Running FHE‐based influence computation …")
    start_time_fhe = time.time()
    fhe_res = influence_fhe.compute_influence_all_fhe(
        test_log=merged_test_log,
        log_loader=log_loader,
        decrypt_results=True,
        hessian=args.hessian,
        damping=run.config.influence.damping,
    )
    logger.info(f"FHE IF finished in {(time.time() - start_time_fhe):.1f}s")

    # ---------------------------
    # 10. Save decrypted FHE results (if decrypted) and optionally validate vs plaintext
    # ---------------------------
    if isinstance(fhe_res.get("influence"), torch.Tensor):
        # We have a decrypted Tensor of shape (num_test, num_successful_train)
        if_scores = fhe_res["influence"].cpu().numpy()
        torch.save(
            {
                "src_ids": fhe_res["src_ids"],   # list of test sample IDs
                "tgt_ids": fhe_res["tgt_ids"],   # list of training IDs that succeeded
                "influence": if_scores,
            },
            f"if_fhe_gpt2_{args.project}.pt",
        )
        logger.info("Saved decrypted FHE influence scores")

        # If not skipping validation, compare to plaintext influence
        if not args.skip_validation:
            logger.info("Validating against plaintext influence …")
            # Compute plaintext influence (exact same inputs & hyperparams)
            plain_res = influence_plain.compute_influence_all(
                src_log=merged_test_log,
                loader=log_loader,
                hessian=args.hessian,
                damping=run.config.influence.damping,
            )

            # If plaintext returned a decrypted Tensor, align and compare
            if isinstance(plain_res.get("influence"), torch.Tensor):
                # Convert both to numpy
                fhe_scores = fhe_res["influence"].numpy()
                plain_scores = plain_res["influence"].numpy()
                fhe_tgt_ids = fhe_res["tgt_ids"]
                plain_tgt_ids = plain_res["tgt_ids"]

                # Find common training IDs for which both FHE and plaintext succeeded
                # Cast IDs to str for consistent set operations
                try:
                    set_fhe = set(map(str, fhe_tgt_ids))
                    set_plain = set(map(str, plain_tgt_ids))
                    common_ids = sorted(list(set_fhe & set_plain))
                except Exception as e:
                    logger.error(
                        f"Error while computing common_tgt_ids: {e}. Skipping comparison."
                    )
                    common_ids = []

                if not common_ids:
                    logger.error(
                        "No common training sample IDs between successful FHE and plaintext results → cannot compare."
                    )
                else:
                    logger.info(
                        f"Comparing on {len(common_ids)} common training samples."
                    )
                    # Map str(ID) → index in each result’s “tgt_ids” list
                    fhe_id2idx = {str(i): idx for idx, i in enumerate(fhe_tgt_ids)}
                    plain_id2idx = {str(i): idx for idx, i in enumerate(plain_tgt_ids)}

                    # Extract 1D score arrays for the *first* test sample (index 0).
                    # The arrays may be 2D (shape [num_test, num_train]) or 1D (if num_test==1).
                    if fhe_scores.ndim == 2 and fhe_scores.shape[0] >= 1:
                        fhe_scores_1d = fhe_scores[0]
                    elif fhe_scores.ndim == 1:
                        fhe_scores_1d = fhe_scores
                    else:
                        logger.error(
                            f"Unexpected FHE scores shape: {fhe_scores.shape}. Skipping comparison."
                        )
                        common_ids = []

                    if plain_scores.ndim == 2 and plain_scores.shape[0] >= 1:
                        plain_scores_1d = plain_scores[0]
                    elif plain_scores.ndim == 1:
                        plain_scores_1d = plain_scores
                    else:
                        logger.error(
                            f"Unexpected plaintext scores shape: {plain_scores.shape}. Skipping comparison."
                        )
                        common_ids = []

                    if common_ids:
                        fhe_vals = []
                        plain_vals = []
                        for cid in common_ids:
                            try:
                                i_fhe = fhe_id2idx[cid]
                                i_plain = plain_id2idx[cid]
                                fhe_vals.append(fhe_scores_1d[i_fhe])
                                plain_vals.append(plain_scores_1d[i_plain])
                            except KeyError as e:
                                logger.error(
                                    f"KeyError aligning ID {cid}: {e}. Skipping that ID."
                                )
                            except IndexError as e:
                                logger.error(
                                    f"IndexError aligning ID {cid}: {e}. Skipping that ID."
                                )

                        if fhe_vals and plain_vals:
                            from scipy.stats import pearsonr

                            fhe_arr = np.array(fhe_vals)
                            plain_arr = np.array(plain_vals)
                            if np.std(fhe_arr) > 1e-9 and np.std(plain_arr) > 1e-9:
                                corr, pval = pearsonr(fhe_arr, plain_arr)
                            else:
                                corr, pval = np.nan, np.nan
                                logger.warning(
                                    "Cannot compute Pearson correlation (constant arrays)."
                                )
                            max_abs_diff = np.max(np.abs(fhe_arr - plain_arr))
                            mean_abs_diff = np.mean(np.abs(fhe_arr - plain_arr))

                            logger.info("-" * 30 + " VALIDATION RESULTS " + "-" * 30)
                            test_id0 = fhe_res["src_ids"][0] if fhe_res["src_ids"] else "Unknown"
                            logger.info(f"Test sample ID: {test_id0}")
                            logger.info(f"  #Common training IDs: {len(common_ids)}")
                            logger.info(f"  Pearson Correlation: {corr:.8f} (p-value {pval:.3e})")
                            logger.info(f"  Max absolute diff: {max_abs_diff:.6e}")
                            logger.info(f"  Mean absolute diff: {mean_abs_diff:.6e}")
                            logger.info("-" * (60 + len(" VALIDATION RESULTS ")))
                        else:
                            logger.error("No aligned scores to compare after filtering.")
            else:
                logger.warning("Plaintext did not return a Tensor of influences → skipping comparison.")

    else:
        logger.warning(
            "InfluenceFunctionFHE returned encrypted results (did not decrypt). Unable to save or validate."
        )

    print(f"\nDone! GPT-2 FHE Influence Analysis for project={args.project} complete.")


if __name__ == "__main__":
    main()
