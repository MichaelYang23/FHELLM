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

from typing import Dict, Optional

import torch
from einops import einsum, rearrange, reduce

from logix.state import LogIXState
from logix.statistic.utils import make_2d
from logix.utils import nested_dict, get_logger


def precondition_kfac(
    src: Dict[str, Dict[str, torch.Tensor]],
    state: LogIXState,
    damping: Optional[float] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    preconditioned = nested_dict()
    cov_eigval, cov_eigvec = state.get_covariance_svd_state()
    for module_name in src.keys():
        src_grad = src[module_name]["grad"]
        device = src_grad.device

        module_eigvec = cov_eigvec[module_name]
        fwd_eigvec = module_eigvec["forward"].to(device=device)
        bwd_eigvec = module_eigvec["backward"].to(device=device)

        # Reconstruct the full eigenvalue matrix with the damping factor added
        module_eigval = cov_eigval[module_name]
        if isinstance(module_eigval, torch.Tensor):
            full_eigval = module_eigval.to(device=device)
        else:
            assert "forward" in module_eigval and "backward" in module_eigval
            fwd_eigval = module_eigval["forward"]
            bwd_eigval = module_eigval["backward"]
            full_eigval = torch.outer(bwd_eigval, fwd_eigval).to(device=device)
        if damping is None:
            damping = 0.1 * torch.mean(full_eigval)
        full_eigval += damping

        # Precondition the gradient using eigenvectors and eigenvalues
        rotated_grad = einsum(
            bwd_eigvec.t(),
            src_grad,
            fwd_eigvec,
            "a b, batch b c, c d -> batch a d",
        )
        prec_rotated_grad = rotated_grad / full_eigval
        preconditioned[module_name]["grad"] = einsum(
            bwd_eigvec,
            prec_rotated_grad,
            fwd_eigvec.t(),
            "a b, batch b c, c d -> batch a d",
        )

    return preconditioned


# def precondition_raw(
#     src: Dict[str, Dict[str, torch.Tensor]],
#     state: LogIXState,
#     damping: Optional[float] = None,
# ) -> Dict[str, Dict[str, torch.Tensor]]:
#     preconditioned = nested_dict()
#     cov_inverse = state.get_covariance_inverse_state(damping=damping)
#     for module_name in src.keys():
#         device = src[module_name]["grad"].device
#         grad_cov_inverse = cov_inverse[module_name]["grad"].to(device=device)
#         original_shape = src[module_name]["grad"].shape
#         preconditioned[module_name]["grad"] = (
#             make_2d(src[module_name]["grad"], None, "grad") @ grad_cov_inverse
#         ).reshape(original_shape)

#     return preconditioned

def precondition_raw(
    src: Dict[str, Dict[str, torch.Tensor]],
    state: LogIXState, # Pass the state object to access module info
    damping: Optional[float] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Preconditions gradients using raw gradient covariance inverse.
    MODIFIED: Retrieves module object from state and passes it to make_2d.
    """
    logger = get_logger() # Get logger instance
    preconditioned = nested_dict()
    # Retrieve the inverse covariance state (computed from extract_log phase)
    try:
        # Ensure state provides a way to get the inverse state
        cov_inverse = state.get_covariance_inverse_state(damping=damping)
    except AttributeError:
        logger.error("LogIXState object does not have 'get_covariance_inverse_state' method.")
        raise
    except Exception as e:
         logger.error(f"Failed to get covariance inverse state: {e}")
         raise # Or return src if you want to handle gracefully

    # Retrieve the mapping from module name to module object
    # --- THIS IS THE CRITICAL PART ---
    # We need a reliable way to get the {name: module_obj} map here.
    # Option 1: Assume it's stored in state (preferred)
    # (Requires modification in LogIX.watch to store it)
    # Example: 
    modules_map = state.get_state("logger_modules_map")

    # Option 2: Less ideal, access logger's map if possible (might break encapsulation)
    # modules_map = state._logger_instance_ref.modules_to_name # Hypothetical

    # Option 3 (Workaround): Pass the modules_map directly if needed
    # This requires changing the call site in InfluenceFunctionFHE.precondition
    # Let's assume Option 1 for now, but this needs to be wired correctly.
    modules_map = state.get_state("logger_modules_map")
    if not modules_map:
        logger.error("Module name to object map ('logger_modules_map') not found in LogIXState. Cannot perform correct flattening for precondition_raw.")
        # Fallback: Revert to old behavior with warning, which will likely error again
        # return src # Or raise error
        use_module_for_flattening = False
    else:
        use_module_for_flattening = True


    for module_name in src.keys():
        # Check if covariance inverse exists for this module/log_type ('grad')
        if module_name not in cov_inverse or "grad" not in cov_inverse[module_name]:
             logger.warning(f"Inverse covariance for 'grad' not found for module '{module_name}'. Skipping preconditioning.")
             preconditioned[module_name]["grad"] = src[module_name]["grad"]
             continue

        module = None
        if use_module_for_flattening:
            # Retrieve the actual module object using the name
            module = modules_map.get(module_name)
            if module is None:
                logger.warning(f"Module object for '{module_name}' not found in map. Using default flatten for preconditioning.")

        # --- Use the module object (if found) for consistent flattening ---
        flat_grad = make_2d(src[module_name]["grad"], module, "grad") # Pass module (or None if not found)

        # Retrieve the inverse covariance matrix
        device = src[module_name]["grad"].device
        grad_cov_inverse = cov_inverse[module_name]["grad"].to(device=device)

        # Check shapes before multiplication
        if flat_grad.shape[1] != grad_cov_inverse.shape[0]:
            logger.error(f"FATAL Shape Mismatch in precondition_raw for module '{module_name}'!")
            logger.error(f"  Flattened Test Gradient Shape: {flat_grad.shape}")
            logger.error(f"  Inverse Covariance Shape:    {grad_cov_inverse.shape}")
            logger.error(f"  Was module object used for flattening? {'Yes' if module else 'No (Fallback)'}")
            logger.error("  Ensure 'hessian=raw' computed Covariance with correct flattening during extract_log.")
            # Skip or raise error
            preconditioned[module_name]["grad"] = src[module_name]["grad"]
            continue # Skip this module

        # Perform the preconditioning H_inv @ g
        original_shape = src[module_name]["grad"].shape
        try:
            preconditioned_value = (flat_grad @ grad_cov_inverse).reshape(original_shape)
            preconditioned[module_name]["grad"] = preconditioned_value
        except RuntimeError as e:
            logger.error(f"Matrix multiplication failed for module '{module_name}': {e}")
            logger.error(f"  Shapes were: {flat_grad.shape} @ {grad_cov_inverse.shape}")
            preconditioned[module_name]["grad"] = src[module_name]["grad"] # Pass through on error

    return preconditioned


def cross_dot_product(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    assert src.shape[1:] == tgt.shape[1:]
    src_expanded = rearrange(src, "n ... -> n 1 ...")
    tgt_expanded = rearrange(tgt, "m ... -> 1 m ...")
    dot_product_result = reduce(
        src_expanded * tgt_expanded,
        "n m ... -> n m",
        "sum",
    )

    return dot_product_result


def merge_influence_results(
    result_all: Dict[str, Dict[str, torch.Tensor]],
    result: Dict[str, Dict[str, torch.Tensor]],
    axis: str = "tgt",
) -> None:
    assert axis in ["src", "tgt"], f"Unsupported axis {axis}."

    # If merged result is empty, just copy the result and return
    if not result_all:
        result_all.update(result)
        return

    dim = int(axis == "tgt")
    id_key = f"{axis}_ids"

    result_all[id_key].extend(result[id_key])
    if isinstance(result["influence"], dict):
        for key in result_all["influence"].keys():
            result_all["influence"][key] = torch.cat(
                [result_all["influence"][key], result["influence"][key]], dim=dim
            )
    else:
        assert isinstance(result["influence"], torch.Tensor)
        result_all["influence"] = torch.cat(
            [result_all["influence"], result["influence"]], dim=dim
        )
