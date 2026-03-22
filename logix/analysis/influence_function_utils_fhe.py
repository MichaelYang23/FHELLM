# Proposed content for: logix/analysis/influence_function_utils_fhe.py

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

import math
import numpy as np
from typing import List, Optional

from Pyfhel import Pyfhel, PyCtxt, PyPtxt # Import necessary Pyfhel classes

from logix.config import FHEConfig # Assuming FHEConfig is defined in config.py
from logix.utils import get_logger


def setup_fhe_context(fhe_config: FHEConfig) -> Pyfhel:
    """
    Sets up the Pyfhel context and generates necessary keys for CKKS.

    Args:
        fhe_config (FHEConfig): Configuration object with FHE parameters.
                                  Requires fields like 'n', 'scale', 'qi_sizes'.

    Returns:
        Pyfhel: An initialized Pyfhel object with context and keys.
    """
    logger = get_logger()
    logger.info("Setting up FHE context (CKKS)...")
    HE = Pyfhel()

    # Prepare parameters for context generation
    # Ensure scale is treated as power of 2 if needed, though Pyfhel handles float scales too.
    # The demo uses 2**30, let's assume fhe_config.scale provides this value directly.
    ckks_params = {
        'scheme': 'CKKS',
        'n': fhe_config.n, # Polynomial modulus degree
        'scale': fhe_config.scale, # Scale for fixed-point encoding
        'qi_sizes': fhe_config.qi_sizes # Bit sizes of coeff_modulus primes
    }

    try:
        HE.contextGen(**ckks_params)
        logger.info(f" Pyfhel context generated with n={ckks_params['n']}, scale={ckks_params['scale']}, qi_sizes={ckks_params['qi_sizes']}")

        HE.keyGen()
        logger.info(" FHE keys generated (public, secret).")

        # Generate relinearization keys (needed after multiplication)
        HE.relinKeyGen()
        logger.info(" FHE relinearization keys generated.")

        # Generate rotation keys (needed for rotate-and-sum in dot product)
        # Generate keys for all powers of 2 up to n/2, covering all possible single rotations needed for summation.
        HE.rotateKeyGen()
        logger.info(" FHE rotation keys generated.")

    except Exception as e:
        logger.error(f"Failed to setup FHE context: {e}")
        raise

    logger.info("FHE context setup complete.")
    return HE


def encrypt_vector(HE: Pyfhel, vector: np.ndarray) -> Optional[PyCtxt]:
    """
    Encrypts a NumPy vector using the CKKS scheme.

    Args:
        HE (Pyfhel): Initialized Pyfhel object.
        vector (np.ndarray): The vector to encrypt (should be float64).

    Returns:
        Optional[PyCtxt]: The resulting ciphertext, or None if encryption fails.
    """
    if not isinstance(vector, np.ndarray):
        vector = np.array(vector)

    if vector.dtype != np.float64:
        vector = vector.astype(np.float64)

    try:
        # Pyfhel's encryptFrac handles encoding and encryption
        ctxt = HE.encryptFrac(vector)
        return ctxt
    except Exception as e:
        get_logger().error(f"Encryption failed: {e}")
        return None


def decrypt_scalar(HE: Pyfhel, ctxt_scalar: PyCtxt) -> Optional[float]:
    """
    Decrypts a PyCtxt expected to contain a scalar value (like a dot product result).

    Args:
        HE (Pyfhel): Initialized Pyfhel object (must hold the secret key).
        ctxt_scalar (PyCtxt): The ciphertext to decrypt.

    Returns:
        Optional[float]: The decrypted scalar value, or None if decryption fails.
                         Returns the value from the first slot.
    """
    try:
        decrypted_array = HE.decryptFrac(ctxt_scalar)
        # The result of dot product sum is usually in the first slot
        if isinstance(decrypted_array, np.ndarray) and decrypted_array.size > 0:
            return decrypted_array[0]
        elif isinstance(decrypted_array, (float, np.float64)): # Handle cases where it might return a scalar directly
             return float(decrypted_array)
        else:
            get_logger().warning(f"Decryption resulted in unexpected type or empty array: {decrypted_array}")
            return None
    except Exception as e:
        # Check if it's due to missing secret key
        if "secret key not set" in str(e).lower():
             get_logger().error("Decryption failed: Secret key not available in this Pyfhel instance.")
        else:
             get_logger().error(f"Decryption failed: {e}")
        return None


def homomorphic_dot_product(HE: Pyfhel, ctxt_v: PyCtxt, ctxt_g: PyCtxt) -> Optional[PyCtxt]:
    """
    Computes the homomorphic dot product of two encrypted vectors v and g.
    Assumes v and g are encrypted vectors of the same logical length k,
    encoded appropriately in the ciphertexts.

    Args:
        HE (Pyfhel): Initialized Pyfhel object with keys.
        ctxt_v (PyCtxt): Ciphertext encrypting vector v.
        ctxt_g (PyCtxt): Ciphertext encrypting vector g.

    Returns:
        Optional[PyCtxt]: Ciphertext encrypting the scalar result s = v^T g,
                          or None if the operation fails.
    """
    logger = get_logger()
    try:
        # 1. Element-wise Multiplication
        # logger.debug(f"DotProd Step 1: Multiplying ctxt_v (scale {ctxt_v.scale()}) and ctxt_g (scale {ctxt_g.scale()})")
        c_prod = ctxt_v * ctxt_g
        # logger.debug(f"           Result c_prod scale: {c_prod.scale()}")


        # 2. Relinearization (reduces ciphertext size from 3 polynomials to 2)
        # logger.debug(f"DotProd Step 2: Relinearizing c_prod (size {c_prod.size()})")
        HE.relinearize(c_prod) # In-place operation
        # logger.debug(f"           Result c_prod size after relin: {c_prod.size()}")

        # 3. Rescaling (reduces scale and consumes one modulus level)
        # logger.debug(f"DotProd Step 3: Rescaling c_prod (mod level {c_prod.mod_level()})")
        HE.rescale_to_next(c_prod) # In-place operation
        # logger.debug(f"           Result c_prod scale after rescale: {c_prod.scale()}, mod level: {c_prod.mod_level()}")


        # 4. Rotate-and-Sum
        # Sum all available slots. This correctly computes the dot product
        # if vectors were shorter than num_slots (padded with zeros)
        # or if vectors used all slots.
        # logger.debug(f"DotProd Step 4: Performing rotate-and-sum...")
        c_sum = c_prod.copy() # Work on a copy if c_prod might be needed later
        num_slots = HE.get_nSlots() # Get the number of available slots
        num_rotations = int(math.log2(num_slots))

        for i in range(num_rotations):
            shift = 1 << i # Rotate by powers of 2: 1, 2, 4, ...
            rotated_sum = HE.rotate(c_sum, k=shift, in_new_ctxt=True) # Use rotate method
            c_sum += rotated_sum
            # logger.debug(f"           Added rotation by {shift}")

        # The final sum is now concentrated in slot 0 (and potentially replicated)
        # logger.debug(f"DotProd Step 5: Finished. Result in c_sum.")
        return c_sum

    except Exception as e:
        logger.error(f"Homomorphic dot product failed: {e}")
        # Consider logging details of the ciphertexts involved if possible
        # logger.error(f"  ctxt_v details: {ctxt_v}")
        # logger.error(f"  ctxt_g details: {ctxt_g}")
        return None