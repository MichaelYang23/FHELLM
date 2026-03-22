# File: examples/bert/compute_influences_fhe_bert.py
# Updated version incorporating fixes from previous discussions

import argparse
import copy
import os
import time # To measure FHE computation time

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm

# LogIX and BERT example imports
import logix
# Import BOTH FHE and Plaintext IF classes
from logix.analysis import InfluenceFunction, InfluenceFunctionFHE
from logix.utils import DataIDGenerator, get_logger, merge_logs
# Reuse BERT utils for model/data loading
from examples.bert.utils import construct_model, get_loaders

# Pyfhel (optional, can be useful for type hints or direct manipulation if needed)
# from Pyfhel import PyCtxt

# Enable TF32 if available (potentially speeds up some PyTorch operations)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

def main():
    parser = argparse.ArgumentParser("BERT FHE Influence Analysis")
    # --- Arguments ---
    parser.add_argument("--project", type=str, required=True,
                        help="LogIX project name (MUST match the project name used in extract_log.py)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to LogIX config YAML (MUST include the 'fhe:' section)")
    parser.add_argument("--data_name", type=str, default="sst2",
                        help="GLUE task name used for training/extraction (e.g., sst2, qnli, rte)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the fine-tuned BERT model checkpoint (output of train.py)")
    parser.add_argument("--hessian", type=str, default="kfac",
                        help="Hessian approximation used ('kfac', 'raw') - determines preconditioning method")
    parser.add_argument("--lora", action="store_true",
                        help="Flag indicating LoRA was used during log extraction (MUST be set if LoRA was used)")
    parser.add_argument("--damping", type=float, default=None,
                        help="Override damping factor for Hessian preconditioning (uses value from config if None)")
    parser.add_argument("--skip_validation", action="store_true",
                        help="Skip the validation step comparing FHE results to plaintext results")
    parser.add_argument("--test_batch_size", type=int, default=1,
                        help="Number of test samples to process for influence analysis")
    parser.add_argument("--log_loader_batch_size", type=int, default=64,
                        help="Batch size for loading training gradients during influence computation")
    parser.add_argument("--data_id_mode", type=str, default="hash", choices=["hash", "decode"],
                        help="Method for generating data IDs ('hash' recommended for uniqueness)")


    args = parser.parse_args()
    logger = get_logger()

    # Use Accelerator for potential multi-GPU setup compatibility
    accelerator = Accelerator()

    # --- 1. Load Model ---
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    # Use the utility function from the BERT example
    # Ensure weights_only=True if loading untrusted checkpoints for security
    model, tokenizer = construct_model(args.data_name, ckpt_path=args.checkpoint)
    model.eval() # Set model to evaluation mode

    # --- 2. Setup LogIX with FHE Config ---
    logger.info(f"Initializing LogIX project '{args.project}' with config '{args.config}'")
    # Initialize LogIX. It reads the --config file initially.
    run = logix.init(project=args.project, config=args.config)
    if run is None:
         # This can happen if logix.init() was called previously in the same process
         logger.warning("logix.init() returned None, likely already initialized globally. Getting instance.")
         # Attempt to get the existing global instance
         if logix._LOGIX_INSTANCE is None:
              raise RuntimeError("LogIX global instance is None after init returned None.")
         run = logix._LOGIX_INSTANCE
         # Manually ensure the config object is updated if needed (though ideally run init only once)
         run.config.load_config(args.config)

    if not hasattr(run.config, 'fhe') or run.config.fhe is None:
        raise ValueError(f"FHE configuration ('fhe:') not found in {args.config}.")
    logger.info(f"Initial FHE parameters loaded: n={run.config.fhe.n}, scale={run.config.fhe.scale}, qi_sizes={run.config.fhe.qi_sizes}")

    # --- 3. Initialize LogIX State, Handle Config Override, Add LoRA ---
    logger.info("Initializing LogIX state from previously saved logs...")
    run.initialize_from_log() # Loads state AND reloads config from the log directory
    logger.info(f"LogIX state initialized. Config potentially reloaded from log dir '{run.log_dir}'.")
    # Now run.config reflects the config saved during extract_log

    # Manually assign model to run instance *before* potential LoRA operations
    run.model = model

    # --- Force 'random' init strategy AFTER potential config reload from log dir ---
    # This prevents the TypeError if the saved config had 'pca' but data is missing
    if args.lora and hasattr(run.config, 'lora') and run.config.lora.init == 'pca':
        logger.warning(f"Config specifies lora.init='pca', but forcing 'random' to avoid potential errors if PCA data is missing.")
        run.config.lora.init = 'random'
        # If the lora_handler exists from a previous step (e.g. within init), update it
        if hasattr(run, 'lora_handler') and run.lora_handler is not None:
             run.lora_handler.init_strategy = 'random'

    # --- Watch model AFTER initializing state and setting model reference ---
    # The order depends on whether LoRA is used.
    if args.lora:
        logger.info(f"Adding LoRA layers (init strategy: {run.config.lora.init})...")
        run.add_lora() # Modifies the 'model' object inplace. Uses the (potentially forced random) config.
        logger.info("Watching LoRA-adapted model layers...")
        run.watch(model, type_filter=[torch.nn.Linear]) # Watch the now LoRA-adapted model
        logger.info("LoRA layers added and model watched.")
    else:
         logger.info("LoRA not used. Watching original model layers...")
         run.watch(model, type_filter=[torch.nn.Linear]) # Watch the original model

    # Prepare model with accelerator AFTER potential LoRA modification and watching
    model = accelerator.prepare(model)

    # --- 4. Build Log Loader (for plaintext training gradients) ---
    logger.info("Building DataLoader for logged training gradients...")
    # Gradients loaded are based on the structure saved during extract_log (LoRA or original)
    # Flatten gradients if influence config specifies it (usually needed for FHE)

    should_flatten = True # Defaulting to True for FHE
    if hasattr(run.config, 'influence') and hasattr(run.config.influence, 'flatten'):
        should_flatten = run.config.influence.flatten
        if not should_flatten:
             logger.warning("FHE generally requires flattened gradients. Ensure influence.flatten is true in config.")


    log_loader = run.build_log_dataloader(
        batch_size=args.log_loader_batch_size,
        flatten=True,
    )

    try:
        num_train_samples = len(log_loader.dataset)
        logger.info(f"Log DataLoader built. Found {num_train_samples} training samples.")
        if num_train_samples == 0:
             logger.warning("Log DataLoader is empty! Did extract_log.py run correctly?")
    except Exception as e:
         logger.error(f"Could not determine dataset length from log_loader: {e}")

    # --- 5. Load Test Data ---
    logger.info(f"Loading test data for {args.data_name}...")
    test_indices = list(range(args.test_batch_size))
    try:
         # Ensure get_loaders returns a tuple/list where the last element is the validation loader
         test_loader = get_loaders(data_name=args.data_name, eval_batch_size=args.test_batch_size, valid_indices=test_indices)[-1]
    except IndexError:
         logger.error(f"Failed to get validation loader from get_loaders for {args.data_name}. Check utils.py.")
         raise
    test_loader = accelerator.prepare(test_loader)
    logger.info(f"Test DataLoader built for {len(test_indices)} samples.")

    # --- 6. Instantiate FHE and Plaintext Calculators ---
    logger.info("Instantiating FHE and Plaintext influence calculators...")
    # Instantiate directly, passing the current state and FHE config
    influence_fhe_calculator = InfluenceFunctionFHE(state=run.state, fhe_config=run.config.fhe)
    # Instantiate plaintext version for validation
    influence_plain_calculator = InfluenceFunction(state=run.state)

    # --- 7. Compute Test Gradient(s) ---
    logger.info("Computing test gradient(s)...")
    run.eval() # Set LogIX to evaluation mode
    run.setup({"grad": ["log"]}) # Configure logger to capture gradients
    # Use specified ID mode ('hash' recommended)
    id_gen = DataIDGenerator(mode=args.data_id_mode)
    test_logs = [] # Store logs for each test batch/sample

    for batch in tqdm(test_loader, desc="Computing Test Gradients"):
        try:
             # Generate unique ID for the batch/sample
            batch_data_ids = id_gen(batch["input_ids"])
            labels = batch.pop("labels").view(-1)
            # Remove 'idx' if it exists in the batch dictionary
            if "idx" in batch: _ = batch.pop("idx")

            # Use LogIX context manager to capture gradients
            with run(data_id=batch_data_ids, mask=batch.get("attention_mask")):
                model.zero_grad()
                # Pass relevant inputs from the batch to the model
                outputs = model(**batch)
                # Process logits (use the fix for direct tensor output)
                logits = outputs.view(-1, outputs.shape[-1])
                loss = F.cross_entropy(logits, labels, reduction="sum", ignore_index=-100)
                # Backward pass to compute gradients
                accelerator.backward(loss)

            # Retrieve and store the computed gradient log
            test_logs.append(copy.deepcopy(run.get_log()))
        except Exception as e:
            logger.error(f"Error during test gradient computation batch: {e}")
            # Optionally continue to next batch or re-raise
            # continue
            raise e

    if not test_logs:
        raise RuntimeError("Failed to compute any test gradients.")

    # Merge logs if multiple test samples were processed
    merged_test_log = merge_logs(test_logs) if len(test_logs) > 1 else test_logs[0]
    num_test_samples_computed = len(merged_test_log[0])
    logger.info(f"Test gradient computation complete for {num_test_samples_computed} samples.")

    # --- 8. Compute FHE Influences ---
    logger.info("Starting FHE influence computation...")
    start_time_fhe = time.time()
    # Determine damping value
    damping_value = args.damping
    if damping_value is None and hasattr(run.config, 'influence'):
        damping_value = run.config.influence.damping
    if damping_value is None: # Fallback
         damping_value = 1e-5
         logger.warning(f"Damping not specified, using default: {damping_value}")

    # Call the FHE computation function on the calculator instance
    fhe_results = influence_fhe_calculator.compute_influence_all_fhe(
        test_log=merged_test_log,
        log_loader=log_loader,
        decrypt_results=True, # Request decryption
        hessian=args.hessian,
        damping=damping_value
    )
    end_time_fhe = time.time()
    logger.info(f"FHE influence computation finished. Time taken: {end_time_fhe - start_time_fhe:.2f} seconds.")

    # --- 9. Process and Save/Analyze FHE Results ---
    # Check if FHE computation returned results and if they are decrypted tensors
    if fhe_results and isinstance(fhe_results.get("influence"), torch.Tensor):
        if_scores_fhe = fhe_results["influence"].numpy() # Shape: (num_test, num_successful_train)
        successful_train_ids = fhe_results.get("tgt_ids", [])
        # num_successful_fhe = if_scores_fhe.shape[1] if if_scores_fhe.ndim == 2 else 0
        num_successful_fhe = len(successful_train_ids)
        
        logger.info(f"Decrypted FHE Influence Scores obtained (shape): {if_scores_fhe.shape}")
        logger.info(f"Scores correspond to {num_successful_fhe} training samples where FHE ops succeeded.")

        if num_successful_fhe == 0:
             logger.error("No valid FHE scores were computed or decrypted.")
        else:
            # Save the FHE scores and corresponding IDs
            fhe_save_path = f"if_fhe_{args.data_name}_{args.project}_lorarank_4_n_2**15_scale_2**40.pt"
            torch.save({
                 'src_ids': fhe_results['src_ids'],
                 'tgt_ids': successful_train_ids,
                 'influence': if_scores_fhe
                 }, fhe_save_path)
            logger.info(f"FHE influence results saved to {fhe_save_path}")

            # # Print top K influential indices for the first test sample
            # if if_scores_fhe.shape[0] > 0:
            #     first_test_scores_fhe = torch.from_numpy(if_scores_fhe[0])
            #     k = min(10, num_successful_fhe) # Ensure k is valid
            #     if k > 0:
            #          vals, relative_indices = torch.topk(first_test_scores_fhe, k=k)
            #          # Map relative indices back to original training IDs
            #          top_original_ids = [successful_train_ids[i] for i in relative_indices.numpy()]
            #          test_sample_id = fhe_results['src_ids'][0] if fhe_results['src_ids'] else "Unknown"
            #          logger.info(f"Top {k} FHE influential training data IDs for test sample '{test_sample_id}': {top_original_ids}")
            #          logger.info(f"  Corresponding FHE scores: {[f'{v:.4e}' for v in vals.numpy()]}")

            # Print top K influential indices for the first test sample
            if if_scores_fhe.size > 0:
                # handle both 1D and 2D outputs
                if if_scores_fhe.ndim == 1:
                    scores_vec = if_scores_fhe
                else:
                    scores_vec = if_scores_fhe[0]
                first_test_scores_fhe = torch.from_numpy(scores_vec)
                k = min(10, len(scores_vec))
                if k > 0:
                    vals, rel_idx = torch.topk(first_test_scores_fhe, k=k)
                    top_ids = [successful_train_ids[i] for i in rel_idx.numpy()]
                    test_id = fhe_results['src_ids'][0] if fhe_results.get('src_ids') else "Unknown"
                    logger.info(f"Top {k} FHE influential training IDs for test sample '{test_id}': {top_ids}")
                    logger.info(f"  Corresponding FHE scores: {[f'{v:.4e}' for v in vals.numpy()]}")
                else:
                     logger.info("No valid FHE scores to rank for the first test sample.")

    elif not fhe_results.get('decrypt_results', True): # Check if decryption was skipped
        logger.warning("FHE results are still encrypted (list of PyCtxt objects). Cannot process further.")
        if args.skip_validation:
             logger.info("Validation skipped as requested.")
             print(f"\nBERT FHE Influence Analysis for project '{args.project}' complete (results remain encrypted).")
             return # Exit early
    else: # Decryption requested but failed or returned empty
         logger.error("FHE influence computation or decryption failed. Cannot process results.")


# --- 10. Validation Step (Compare with Plaintext) ---
    if not args.skip_validation:
        # Proceed only if FHE results were successfully decrypted into a tensor
        if not (fhe_results and isinstance(fhe_results.get("influence"), torch.Tensor) and fhe_results["influence"].numel() > 0):
            logger.error("Cannot perform validation because FHE results are missing, encrypted, or empty.")
        else:
            logger.info("Starting Plaintext influence computation for validation...")
            start_time_plain = time.time()

            # Compute plaintext influences using the exact same inputs and parameters
            # Assumes 'influence_plain_calculator' was instantiated earlier like:
            # influence_plain_calculator = InfluenceFunction(state=run.state)
            plain_results = influence_plain_calculator.compute_influence_all(
                src_log=merged_test_log,   # Same test log used for FHE
                loader=log_loader,         # Same training gradient loader
                damping=damping_value,     # Same damping value
                hessian=args.hessian       # Same Hessian method
            )
            end_time_plain = time.time()
            logger.info(f"Plaintext influence computation finished. Time taken: {end_time_plain - start_time_plain:.2f} seconds.")

            # Compare results
            if isinstance(plain_results.get("influence"), torch.Tensor):
                # Get the numpy arrays for comparison
                if_scores_fhe = fhe_results["influence"].numpy()
                if_scores_plain = plain_results["influence"].numpy()
                logger.info(f"Plaintext Influence Scores obtained (shape): {if_scores_plain.shape}")
                logger.info(f"FHE Influence Scores obtained (shape): {if_scores_fhe.shape}")


                # --- Careful Comparison Handling ---
                fhe_tgt_ids = fhe_results.get('tgt_ids', [])
                plain_tgt_ids = plain_results.get('tgt_ids', [])

                # Find common target IDs for which *both* computations succeeded
                # Ensure IDs are strings for reliable set operations
                try:
                    common_tgt_ids_set = set(map(str, fhe_tgt_ids)) & set(map(str, plain_tgt_ids))
                    common_tgt_ids = sorted(list(common_tgt_ids_set))
                except TypeError:
                     logger.error("Target IDs cannot be consistently converted to strings. Cannot find common IDs.")
                     common_tgt_ids = [] # Skip comparison

                if not common_tgt_ids:
                    logger.error("No common training sample IDs between successful FHE and plaintext results. Cannot compare.")
                else:
                    logger.info(f"Comparing results for {len(common_tgt_ids)} commonly computed training samples.")

                    # Create mappings from ID string to index for efficient lookup
                    fhe_id_to_idx = {str(id_val): i for i, id_val in enumerate(fhe_tgt_ids)}
                    plain_id_to_idx = {str(id_val): i for i, id_val in enumerate(plain_tgt_ids)}

                    # --- FIX for IndexError ---
                    # Determine the correct array slice for the first test sample (index 0)
                    # Handle cases where the result might be 1D (if test_batch_size=1) or 2D
                    if if_scores_fhe.ndim == 2 and if_scores_fhe.shape[0] >= 1:
                         fhe_scores_1d = if_scores_fhe[0] # Get the first row (scores for test sample 0)
                    elif if_scores_fhe.ndim == 1:
                         fhe_scores_1d = if_scores_fhe # It's already 1D
                    else:
                         logger.error(f"Unexpected shape for FHE scores: {if_scores_fhe.shape}. Cannot perform validation.")
                         common_tgt_ids = [] # Skip comparison loop

                    if if_scores_plain.ndim == 2 and if_scores_plain.shape[0] >= 1:
                         plain_scores_1d = if_scores_plain[0]
                    elif if_scores_plain.ndim == 1:
                         plain_scores_1d = if_scores_plain
                    else:
                         logger.error(f"Unexpected shape for Plaintext scores: {if_scores_plain.shape}. Cannot perform validation.")
                         common_tgt_ids = [] # Skip comparison loop

                    # Proceed only if we have valid 1D arrays to index
                    if common_tgt_ids:
                        fhe_common_scores_list = []
                        plain_common_scores_list = []
                        comparison_successful = True
                        for id_val in common_tgt_ids:
                            try:
                                # Index into the 1D score arrays using the map
                                fhe_idx = fhe_id_to_idx[id_val]
                                plain_idx = plain_id_to_idx[id_val]
                                fhe_common_scores_list.append(fhe_scores_1d[fhe_idx])
                                plain_common_scores_list.append(plain_scores_1d[plain_idx])
                            except KeyError as e:
                                logger.error(f"KeyError during score alignment for ID '{id_val}': {e}. Inconsistent IDs between runs?")
                                comparison_successful = False
                                break
                            except IndexError as e:
                                logger.error(f"IndexError during score alignment for ID '{id_val}': {e}. Check shapes/indices.")
                                logger.error(f" FHE 1D shape: {fhe_scores_1d.shape}, Plain 1D shape: {plain_scores_1d.shape}, Trying index FHE:{fhe_idx}, Plain:{plain_idx}")
                                comparison_successful = False
                                break

                        # Convert lists to numpy arrays only if alignment was successful
                        if comparison_successful:
                            fhe_common_scores = np.array(fhe_common_scores_list)
                            plain_common_scores = np.array(plain_common_scores_list)
                            # --- END FIX ---

                            # Perform comparison
                            from scipy.stats import pearsonr
                            # Ensure arrays are not constant before calculating correlation
                            if np.std(fhe_common_scores) > 1e-9 and np.std(plain_common_scores) > 1e-9:
                                correlation, p_value = pearsonr(fhe_common_scores, plain_common_scores)
                            else:
                                correlation, p_value = np.nan, np.nan # Avoid error for constant arrays
                                logger.warning("Cannot calculate Pearson correlation due to constant score values.")

                            max_abs_diff = np.max(np.abs(fhe_common_scores - plain_common_scores))
                            mean_abs_diff = np.mean(np.abs(fhe_common_scores - plain_common_scores))

                            logger.info("-" * 30 + " VALIDATION RESULTS " + "-" * 30)
                            test_sample_id_val = fhe_results['src_ids'][0] if fhe_results['src_ids'] else "Unknown"
                            logger.info(f" Comparing results for test sample: {test_sample_id_val}")
                            logger.info(f"  Number of commonly computed samples: {len(common_tgt_ids)}")
                            logger.info(f"  Pearson Correlation: {correlation:.8f} (p-value: {p_value:.3e})")
                            logger.info(f"  Max Absolute Difference: {max_abs_diff:.6e}")
                            logger.info(f"  Mean Absolute Difference: {mean_abs_diff:.6e}")
                            logger.info("-" * (60 + len(" VALIDATION RESULTS ")))
                        else:
                            logger.error("Comparison skipped due to errors during score alignment.")
            else:
                logger.error("Plaintext influence computation failed or did not return a tensor.")
    else:
        logger.info("Validation skipped as requested.")

    print(f"\nBERT FHE Influence Analysis for project '{args.project}' complete.")


if __name__ == "__main__":
    main()