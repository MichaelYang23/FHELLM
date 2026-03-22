# Proposed content for: logix/analysis/influence_function_fhe.py

# Copyright 2023-present the LogIX team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional, Tuple, Union, Set
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Pyfhel import Pyfhel, PyCtxt # Assuming Pyfhel types might be needed

# LogIX imports
from logix.config import FHEConfig # Assuming FHEConfig is defined in config.py
from logix.state import LogIXState
from logix.utils import get_logger, flatten_log # Import necessary utils

# Import FHE utility functions (assuming they are defined here)
from .influence_function_utils_fhe import (
    setup_fhe_context,
    encrypt_vector,
    decrypt_scalar,
    homomorphic_dot_product,
)

# Import original preconditioning functions
from .influence_function_utils import (
    precondition_kfac,
    precondition_raw,
    merge_influence_results, # May need adaptation for FHE results
)


class InfluenceFunctionFHE:
    """
    Computes influence functions using Fully Homomorphic Encryption (FHE)
    to protect gradient privacy.
    """
    def __init__(self, state: LogIXState, fhe_config: FHEConfig):
        """
        Initializes the FHE Influence Function calculator.

        Args:
            state (LogIXState): The LogIX state object containing necessary
                                 information like covariance matrices.
            fhe_config (FHEConfig): Configuration object containing parameters
                                    for the FHE scheme (CKKS).
        """
        # import pdb; pdb.set_trace()
        self._state = state
        self.fhe_config = fhe_config
        self.HE: Pyfhel = setup_fhe_context(fhe_config) # Setup Pyfhel context

        # Check if secret key is available for potential decryption
        # Pyfhel might not have a direct check, manage key availability externally if needed.
        # self.has_secret_key = self.HE.secret_key is not None # Example check
        
        
        # # === ADD DEBUG PRINT HERE ===
        # logger = get_logger()
        # try:
        #     context_params = self.HE.getContextParams()
        #     num_slots = self.HE.get_nSlots()
        #     logger.info(f"DEBUG FHE Context: n={context_params.n}, slots={num_slots}")
        # except Exception as e:
        #     logger.error(f"DEBUG FHE Context: Failed to get context info - {e}")
        # # === END DEBUG PRINT ===



        # Storage for results if needed (similar to original IF)
        self.influence_scores_fhe = {}
        get_logger().info("InfluenceFunctionFHE initialized.")

    @torch.no_grad()
    def precondition(
        self,
        src_log: Tuple[List[str], Dict[str, Dict[str, torch.Tensor]]],
        damping: Optional[float] = None,
        hessian: Optional[str] = "auto",
    ) -> Tuple[List[str], Dict[str, Dict[str, torch.Tensor]]]:
        """
        Precondition gradients using the Hessian (KFAC or Raw).
        This method computes H^{-1} g_test in **plaintext**.

        Args:
            src_log (Tuple[List[str], Dict[str, Dict[str, torch.Tensor]]]):
                Log containing plaintext gradients (typically the test gradient).
                Format: (list_of_data_ids, nested_dict_of_gradients)
            damping (Optional[float], optional): Damping parameter for preconditioning.
            hessian (Optional[str], optional): Type of Hessian approximation ('kfac', 'raw', 'auto').

        Returns:
            Tuple[List[str], Dict[str, Dict[str, torch.Tensor]]]:
                The original data IDs and the preconditioned gradients in plaintext.
        """
        # This logic is largely reused from the original InfluenceFunction
        # It operates entirely in plaintext using the stored Hessian approximations.

        # === ADD 'none' HANDLING ===
        if hessian == 'none':
            get_logger().info("Hessian type is 'none', skipping preconditioning.")
            return src_log # Return original gradient
        # === END ADDITION ===

        assert hessian in ["auto", "kfac", "raw"], f"Invalid hessian type: {hessian}"

        src_ids, src_dict = src_log
        cov_state = self._state.get_covariance_state()

        # Check if necessary covariance states are available
        if not cov_state or len(set(src_dict.keys()) - set(cov_state.keys())) != 0:
            get_logger().warning(
                "Covariance state missing for some modules or not computed."
                " No Hessian preconditioning will be applied."
            )
            return src_log # Return original gradients if Hessian info is missing

        # Determine which preconditioning function to use
        precondition_fn = precondition_kfac
        # Heuristic: if 'grad' covariance exists, assume 'raw' (full gradient covariance) is preferred/available
        if hessian == "raw" or (
            hessian == "auto" and "grad" in cov_state[list(src_dict.keys())[0]]
        ):
            get_logger().info("Using 'raw' Hessian preconditioning.")
            precondition_fn = precondition_raw
        else:
             get_logger().info("Using 'kfac' Hessian preconditioning.")


        # Compute preconditioned gradient v = H^{-1} g_test
        preconditioned_grad_dict = precondition_fn(
            src=src_dict, state=self._state, damping=damping
        )

        return (src_ids, preconditioned_grad_dict)

    def compute_influence_all_fhe(
        self,
        test_log: Tuple[List[str], Dict[str, Dict[str, torch.Tensor]]],
        log_loader: DataLoader,
        decrypt_results: bool = True, # Option to control decryption
        hessian: Optional[str] = "auto",
        damping: Optional[float] = None,
        # Add other relevant parameters like mode if needed later
    ) -> Dict:
        """
        Computes FHE-based influence scores for test samples against all training data.

        Args:
            test_log (Tuple): Plaintext test gradient log (data_ids, grad_dict).
                              Usually contains one or a few test samples.
            log_loader (DataLoader): DataLoader yielding batches of plaintext
                                     training gradients (data_ids, grad_dict).
            decrypt_results (bool): If True and secret key is available, decrypt
                                    the influence scores before returning.
                                    Otherwise, return encrypted scores (PyCtxt objects).
            hessian (str): Hessian approximation type ('kfac', 'raw', 'auto').
            damping (float): Damping factor for Hessian preconditioning.

        Returns:
            Dict: A dictionary containing 'src_ids' (test sample IDs),
                  'tgt_ids' (training sample IDs), and 'influence'
                  (a list of decrypted floats or PyCtxt objects).
        """
        # 1. Precondition the test gradient(s) in plaintext
        #    v_plain_dict format: {module_name: {'grad': tensor}}
        # import pdb; pdb.set_trace()


        test_ids, v_plain_dict = self.precondition(
            src_log=test_log, 
            damping=damping, 
            hessian=hessian
        )

        # Assuming we process one test sample at a time for now.
        # Handle batching of test samples later if needed.
        if len(test_ids) > 1:
            get_logger().warning("Processing multiple test samples at once in FHE mode is not fully implemented yet. Using the first test sample.")
            # Select the first sample's data
            test_ids = [test_ids[0]]
            v_plain_dict = {m: {'grad': v['grad'][0:1]} for m, v in v_plain_dict.items()}


        # 2. Flatten and Encrypt the preconditioned test gradient v
        #    Flatten the dictionary into a single vector before encryption
        v_plain_flat_tensor = flatten_log(v_plain_dict, path=self._state.get_state("model_module")["path"])
        # Ensure it's a numpy array for encryption
        v_plain_flat_np = v_plain_flat_tensor.cpu().numpy().astype(np.float64)

        # Check dimensions if needed, ensure compatibility with FHE context
        # num_slots = self.HE.poly_modulus_degree // 2
        # if v_plain_flat_np.shape[1] > num_slots:
        #    raise ValueError(f"Gradient dimension ({v_plain_flat_np.shape[1]}) exceeds available FHE slots ({num_slots})")

        # Encrypt the (potentially batched) vector v
        ctxt_v = encrypt_vector(self.HE, v_plain_flat_np[0]) # Encrypt first sample's vector

        # 3. Iterate through training data loader and compute homomorphic dot products
        logger = get_logger()
        all_tgt_ids = []
        all_influences = [] # Will store PyCtxt or decrypted floats

        get_logger().info("Starting FHE influence computation...")

        for train_ids_batch, train_grads_flat_batch_tensor in tqdm(log_loader, desc="Compute FHE IF"):

            # import pdb; pdb.set_trace()

            # Convert the batched flat tensor to numpy for easier iteration
            # Ensure it's float64 for Pyfhel
            try:
                train_grads_flat_batch_np = train_grads_flat_batch_tensor.cpu().numpy().astype(np.float64)
            except AttributeError:
                logger.error("Log loader did not yield a tensor as the second element. Did you use flatten=True?")
                # Handle error appropriately, maybe break or raise
                raise TypeError("Expected a tensor from log_loader (flatten=True)")

            current_batch_size = len(train_ids_batch)
            if current_batch_size != train_grads_flat_batch_np.shape[0]:
                logger.error(f"Batch size mismatch between IDs ({current_batch_size}) and tensor ({train_grads_flat_batch_np.shape[0]})")
                continue # Skip problematic batch

            # *** CORRECTED: Loop through each FLAT gradient tensor IN the batch ***
            for i in range(current_batch_size):
                # --- a. Get the flattened gradient for the i-th sample ---
                # No need to call flatten_log here, tensor is already flat!
                g_train_flat_np = train_grads_flat_batch_np[i]

                # # === CORRECTED DEBUG PRINT ===
                # logger.info(f"DEBUG FHE Encrypt: Sample index in batch {i}, ID: {train_ids_batch[i]}, "
                #             f"Vector shape: {g_train_flat_np.shape}, dtype: {g_train_flat_np.dtype}")
                # # === END CORRECTION ===

                # --- b. Encrypt the flattened training gradient g_train ---
                ctxt_g = encrypt_vector(self.HE, g_train_flat_np)
                if ctxt_g is None:
                    logger.warning(f"Encryption failed for training sample {i} (ID: {train_ids_batch[i]}). Skipping.")
                    continue # Skip if encryption fails

                # --- c. Compute the homomorphic dot product: v^T g_train ---
                # ctxt_v was computed earlier (assuming only one test sample for now)
                ctxt_infl = homomorphic_dot_product(self.HE, ctxt_v, ctxt_g)
                if ctxt_infl is None:
                    logger.warning(f"Homomorphic dot product failed for training sample {i} (ID: {train_ids_batch[i]}). Skipping.")
                    continue # Skip if dot product fails

                # --- d. Append the result (encrypted or decrypted) ---
                all_influences.append(ctxt_infl)
                # Append the corresponding training ID
                all_tgt_ids.append(train_ids_batch[i])
        # *** END of inner loop through batch samples ***
        # for train_ids_batch, train_grads_batch_dict in tqdm(log_loader, desc="Compute FHE IF"):
        #     # Flatten the batch of training gradients
        #     train_grads_flat_tensor = flatten_log(train_grads_batch_dict, 
        #                                           path=self._state.get_state("model_module")["path"])
        #     train_grads_flat_np = train_grads_flat_tensor.cpu().numpy().astype(np.float64)

        #     # Process each training gradient in the batch
        #     for i, g_train_flat_np in enumerate(train_grads_flat_np):
        #         # Encrypt the training gradient g_train
        #         ctxt_g = encrypt_vector(self.HE, g_train_flat_np)

        #         # Compute the homomorphic dot product: v^T g_train
        #         ctxt_infl = homomorphic_dot_product(self.HE, ctxt_v, ctxt_g)

        #         # Append the result (encrypted or decrypted)
        #         all_influences.append(ctxt_infl)
        #         all_tgt_ids.append(train_ids_batch[i])

        # 4. Decrypt results if requested and possible
        final_influences = []
        if decrypt_results:
            # Add a check here if self.HE actually holds the secret key
            # For now, assume it does if decrypt_results is True
            get_logger().info("Decrypting FHE influence scores...")
            try:
                 for ctxt_infl in tqdm(all_influences, desc="Decrypting"):
                     # DecryptFrac usually returns an array, take the first element for dot product
                     decrypted_array = decrypt_scalar(self.HE, ctxt_infl)
                     final_influences.append(decrypted_array[0] if isinstance(decrypted_array, np.ndarray) else decrypted_array)
            except Exception as e:
                 get_logger().error(f"Decryption failed: {e}. Returning encrypted results.")
                 decrypt_results = False # Fallback to returning ciphertexts

        if not decrypt_results:
             get_logger().warning("Returning encrypted influence scores (PyCtxt objects).")
             final_influences = all_influences # Return list of ciphertexts

        # 5. Format and return results
        #    The structure should match the output of the original compute_influence_all
        #    The 'influence' value is now a list (potentially very long)
        result = {
            "src_ids": test_ids, # List containing the ID(s) of the test sample(s)
            "tgt_ids": all_tgt_ids, # List containing all training sample IDs
            "influence": torch.tensor(final_influences) if decrypt_results else final_influences
            # Convert to tensor if decrypted, otherwise return list of PyCtxt
            # Note: Returning a list of PyCtxt might be memory intensive
        }

        # Potentially store results internally if needed, similar to original IF
        # merge_influence_results(self.influence_scores_fhe, result, axis="src") # Needs adaptation for FHE

        get_logger().info("FHE influence computation finished.")
        return result

    # Add compute_influence_fhe and compute_self_influence_fhe later if needed,
    # potentially reusing parts of compute_influence_all_fhe.
    # Self-influence is v^T v = (H^{-1}g)^T (H^{-1}g), which might be complex in FHE.
    # Or maybe g^T H^{-1} g, which requires encrypting g, applying H^{-1}, then dot product.







    # ----------------------------------------------------------------------
    #  Second-order (MISS-style) group influence – FHE implementation
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def compute_second_order_group_influence_fhe(
        self,
        test_log: Tuple[List[str], Dict[str, Dict[str, torch.Tensor]]],
        log_loader: DataLoader,
        group_indices: Set[str],
        enc_table    : Optional[Dict[str, Tuple[PyCtxt, PyCtxt, PyCtxt]]] = None,
        damping: Optional[float] = None,
        hessian: str = "none",          # "none" = raw g_i,  else "kfac"/"raw"
        bootstrap: bool = False,
        p_keep: float = 0.05,
        seed: Optional[int] = None,
    ) -> PyCtxt:
        """
        Returns a CKKS ciphertext encrypting
            I²(U) = Σ_i s_i  +  ½ Σ_{i,j∈U} s_i · (g_iᵀ g_j')
        where g_j' =  H⁻¹ g_j  if `hessian` ∈ {"kfac","raw"}, else g_j.

        * Works with any `log_loader` built with `flatten=True`.
        * Shows tqdm progress for the O(|U|²) interaction loop.
        * Never crashes on scale/level — every multiply is rescaled once.
        """
        import numpy as np
        logger = get_logger()
        logger.info(
            f"Second-order influence (hessian='{hessian}') for |U|={len(group_indices)}"
        )

        # 1 ── encrypt v = H⁻¹ g_test  (uses self.precondition, may be raw)
        _, v_plain = self.precondition(test_log, damping=damping, hessian=hessian)
        first = {m: {"grad": v["grad"][0:1]} for m, v in v_plain.items()}
        v_np = flatten_log(first, path=self._state.get_state("model_module")["path"])\
                .cpu().numpy()[0].astype(np.float64)
        ctxt_v = encrypt_vector(self.HE, v_np)
        


        # --------------------------------------------------------------
        # 2 ── collect from cached table  or  fallback to log_loader
        # --------------------------------------------------------------
        s_ctxt, g_ctxt = {}, {}

        if enc_table is not None:
            for tid in group_indices:
                if tid in enc_table:
                    c_g, c_gprime, c_s = enc_table[tid]
                    g_ctxt[tid] = (c_g, c_gprime)
                    s_ctxt[tid] = c_s
        else:
            for ids_batch, flat_tensor in tqdm(log_loader, desc="Encrypt grads"):
                flat_np = flat_tensor.cpu().numpy().astype(np.float64)
                for k, tid_raw in enumerate(ids_batch):
                    tid = str(tid_raw)
                    if tid not in group_indices:
                        continue
                    g_np = flat_np[k]

                    if hessian in {"kfac", "raw"}:
                        gprime_np = self._apply_inverse_hessian_to_flat(
                            g_np, hessian=hessian, damping=damping
                        )
                    else:
                        gprime_np = g_np

                    c_g      = encrypt_vector(self.HE, g_np)
                    c_gprime = encrypt_vector(self.HE, gprime_np)
                    c_s      = homomorphic_dot_product(self.HE, ctxt_v, c_g)

                    g_ctxt[tid] = (c_g, c_gprime)
                    s_ctxt[tid] = c_s

        if not s_ctxt:
            raise RuntimeError("No group IDs matched the log loader IDs.")

        # 3 ── initialise accumulators  (scale ~ 2^40)
        zero_ptxt   = np.array([0.0], dtype=np.float64)
        ctxt_first  = self.HE.encryptFrac(zero_ptxt)
        ctxt_second = self.HE.encryptFrac(zero_ptxt)

        for c in s_ctxt.values():
            ctxt_first += c           # same scale, no rescale needed

        # # 4 ── pairwise ½ Σ_{i≤j} s_i (g_iᵀ g_j')
        # tids = list(s_ctxt.keys())
        # total_pairs = len(tids) * (len(tids) + 1) // 2
        # with tqdm(total=total_pairs, desc="Pairwise interactions", unit="pair") as pbar:
        #     for i, tid_i in enumerate(tids):
        #         s_i  = s_ctxt[tid_i]
        #         g_i, _ = g_ctxt[tid_i]
        #         for tid_j in tids[i:]:
        #             _, gprime_j = g_ctxt[tid_j]

        #             dot_ij = homomorphic_dot_product(self.HE, g_i, gprime_j)
        #             # s_i · dot_ij
        #             q_ij = s_i * dot_ij
        #             self.HE.relinearize(q_ij); self.HE.rescale_to_next(q_ij)

        #             q_ij *= 0.5           # mul-plain keeps scale, then rescale
        #             self.HE.rescale_to_next(q_ij)

        #             if tid_i != tid_j:    # duplicate off-diagonal
        #                 q_ij += q_ij      # addition keeps scale/level

        #             ctxt_second += q_ij

        #             if bootstrap and ctxt_second.level() < 2:
        #                 self.HE.bootstrap(ctxt_second)
        #             pbar.update(1)



        # ---------- 4.  Monte-Carlo interaction term -------------------------
        # p_keep    = 0.05                    # keep 5 % of pairs
        # p_keep    = 0.1 # keep 10 % of pairs
        scale_mc  = 1.0 / p_keep

        tids = list(s_ctxt.keys())
        total_pairs = len(tids) * (len(tids) + 1) // 2
        # rng = np.random.default_rng()
        from numpy.random import default_rng
        rng = default_rng(seed) 

        with tqdm(total=total_pairs, desc=f"Pairs (MC p={p_keep})", unit="pair") as pbar:
            for i, tid_i in enumerate(tids):
                s_i, (g_i, _) = s_ctxt[tid_i], g_ctxt[tid_i]
                for tid_j in tids[i:]:
                    _, gprime_j = g_ctxt[tid_j]

                    if rng.random() > p_keep:     # skip most pairs
                        pbar.update(1)
                        continue

                    # -- compute q_ij exactly as before (NO scaling) ------------
                    dot_ij = homomorphic_dot_product(self.HE, g_i, gprime_j)
                    q_ij   = s_i * dot_ij
                    self.HE.relinearize(q_ij); self.HE.rescale_to_next(q_ij)

                    q_ij *= 0.5
                    self.HE.rescale_to_next(q_ij)

                    if tid_i != tid_j:
                        q_ij += q_ij              # duplicate off-diagonal

                    ctxt_second += q_ij           # same-scale add

                    if bootstrap and ctxt_second.level() < 2:
                        self.HE.bootstrap(ctxt_second)

                    pbar.update(1)

        # -------------------------------------------------------------------

        def _dbg_ckks_status(tag: str, ctxt: PyCtxt, he: Pyfhel) -> None:
            """
            Print scale and remaining-modulus information for a ciphertext.

            Works with both old (he.context.qi_sizes) and new (he.qi_sizes) APIs.
            Fails gracefully if neither is present.
            """
            sc  = ctxt.scale
            lvl = ctxt.mod_level                 # property

            # obtain qi_sizes list safely
            if hasattr(he, "qi_sizes"):
                qi = he.qi_sizes                 # Pyfhel ≥ 3.4
            elif hasattr(he, "context") and hasattr(he.context, "qi_sizes"):
                qi = he.context.qi_sizes         # older versions
            else:
                qi = None

            if qi is not None:
                maxlvl = len(qi) - 1
                rem    = maxlvl - lvl
                msg = f"scale={sc:.3e}, level={lvl}/{maxlvl} (remaining primes: {rem})"
            else:
                msg = f"scale={sc:.3e}, level={lvl} (qi_sizes unavailable)"

            get_logger().info(f"[CKKS-DBG] {tag}: {msg}")




        
        # ---- apply Monte-Carlo weight once (safe) ---------------------------
        _dbg_ckks_status("before MC scaling", ctxt_second, self.HE)
        ctxt_second *= scale_mc            # small number of rescalings left
        self.HE.rescale_to_next(ctxt_second)
        _dbg_ckks_status("after  MC scaling", ctxt_second, self.HE)





        logger.info("Second-order group influence finished.")
        return ctxt_first + ctxt_second


    # ----------------------------------------------------------------------
    # helper :  H⁻¹ · g   for ONE *flattened* gradient vector
    # ----------------------------------------------------------------------
    def _apply_inverse_hessian_to_flat(
        self,
        g_flat_np: np.ndarray,
        hessian: str = "kfac",
        damping: Optional[float] = None,
    ) -> np.ndarray:
        """
        Plain-text preconditioning helper.
        * If shapes cannot be inferred, returns the original `g_flat_np`
        (only ONE warning the first time).
        """
        logger = get_logger()

        if hessian == "none":
            return g_flat_np.copy()

        # ── cache / derive layer shapes once ─────────────────────────────
        if not hasattr(self, "_cached_shapes"):
            shapes = (self._state.get_state("model_module") or {}).get("shapes")
            if shapes is None:
                cov = self._state.get_covariance_state()
                if cov:
                    shapes = {k: tuple(v["grad"].shape[1:]) for k, v in cov.items()}
            self._cached_shapes = shapes if shapes else None
            if self._cached_shapes is None:
                logger.warning(
                    "[MISS-helper] Could not determine layer shapes; "
                    "second-order uses raw gradients (no H⁻¹)."
                )

        if self._cached_shapes is None:
            return g_flat_np.copy()

        # ── un-flatten ───────────────────────────────────────────────────
        ptr, per_layer = 0, {}
        for name, shp in self._cached_shapes.items():
            size = int(np.prod(shp))
            per_layer[name] = g_flat_np[ptr : ptr + size].reshape(shp)
            ptr += size
        if ptr != g_flat_np.size:
            logger.error("[MISS-helper] Flat vector length mismatch; using raw g.")
            return g_flat_np.copy()

        # ── precondition layer-wise ──────────────────────────────────────
        torch_dict = {k: {"grad": torch.from_numpy(v[None, ...])}
                    for k, v in per_layer.items()}
        try:
            if hessian in {"kfac", "auto"}:
                pre = precondition_kfac(src=torch_dict, state=self._state, damping=damping)
            else:  # "raw"
                pre = precondition_raw(src=torch_dict, state=self._state, damping=damping)
        except Exception as e:
            logger.error(f"[MISS-helper] Preconditioning failed ({e}); using raw g.")
            return g_flat_np.copy()

        # ── re-flatten ───────────────────────────────────────────────────
        flat = [pre[name]["grad"].reshape(-1).numpy() for name in self._cached_shapes]
        return np.concatenate(flat, axis=0).astype(np.float64)


    # ------------------------------------------------------------------
    # helper: pre-encrypt every (g_i, g_i', s_i)  ->  dict[id] = tuple
    # ------------------------------------------------------------------
    @torch.no_grad()
    def build_encrypted_table(
        self,
        log_loader      : DataLoader,
        test_log        : Tuple,          # for v & s_i
        hessian         : str = "none",
        damping         : Optional[float] = None,
    ) -> Dict[str, Tuple[PyCtxt, PyCtxt, PyCtxt]]:
        """
        Returns {train_id: (ctxt_g, ctxt_gprime, ctxt_s)} for *all* samples
        in `log_loader`.  Heavy part (~minutes) but run once.
        """
        logger = get_logger()
        logger.info("Building encrypted gradient table …")

        # --- encrypt v -------------------------------------------------
        _, v_plain = self.precondition(test_log, damping=damping, hessian=hessian)
        v_np = flatten_log(
            {m: {"grad": v["grad"][0:1]} for m, v in v_plain.items()},
            path=self._state.get_state("model_module")["path"]
        ).cpu().numpy()[0].astype(np.float64)
        ctxt_v = encrypt_vector(self.HE, v_np)

        enc_table = {}
        for ids_batch, flat_tensor in tqdm(log_loader, desc="Encrypt grads"):
            flat_np = flat_tensor.cpu().numpy().astype(np.float64)

            for k, tid_raw in enumerate(ids_batch):
                tid = str(tid_raw)
                g_np = flat_np[k]

                # H^{-1}·g_j  (optional)
                if hessian in {"kfac", "raw"}:
                    gprime_np = self._apply_inverse_hessian_to_flat(
                        g_np, hessian=hessian, damping=damping
                    )
                else:
                    gprime_np = g_np

                c_g      = encrypt_vector(self.HE, g_np)
                c_gprime = encrypt_vector(self.HE, gprime_np)
                c_s      = homomorphic_dot_product(self.HE, ctxt_v, c_g)

                enc_table[tid] = (c_g, c_gprime, c_s)

        logger.info(f"Encrypted {len(enc_table)} training samples.")
        return enc_table
