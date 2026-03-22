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

import os
from copy import deepcopy
from functools import reduce
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from logix.analysis import (
    INFLUENCE_FHE_IMPORT_ERROR,
    InfluenceFunction,
    InfluenceFunctionFHE,
)
from logix.batch_info import BatchInfo

from logix.config import Config, LoggingConfig, LoRAConfig, FHEConfig, init_config_from_yaml
# from logix.config import Config, LoggingConfig, LoRAConfig, init_config_from_yaml
from logix.logging import HookLogger
from logix.logging.log_loader import LogDataset
from logix.logging.log_loader_utils import collate_nested_dicts
from logix.lora import LoRAHandler
from logix.lora.utils import is_lora
from logix.state import LogIXState
from logix.utils import (
    get_logger,
    get_rank,
    get_repr_dim,
    get_world_size,
    module_check,
    print_tracked_modules,
)

try:
    from Pyfhel import Pyfhel as _PyfhelType
except Exception:
    _PyfhelType = None

class LogIX:
    """
    LogIX is a front-end interface for logging and analyzing neural networks.
    Using (PyTorch) hooks, it tracks, saves, and computes statistics for activations,
    gradients, and other tensors with its logger. Once logging is finished, it provides
    an interface for computing influence scores and other analysis.

    Args:
        project (str):
            The name or identifier of the project. This is used for organizing
            logs and analysis results.
        config (str, optional):
            The path to the YAML configuration file. This file contains settings
            for logging, analysis, and other features. Defaults to an empty string,
            which means default settings are used.
    """

    _SUPPORTED_MODULES = {nn.Linear, nn.Conv1d, nn.Conv2d}

    def __init__(
        self,
        project: str,
        config: Optional[str] = None,
        logging_config: Optional[LoggingConfig] = None,
    ) -> None:
        self.project: str = project

        self.model: Optional[nn.Module] = None

        # Config
        self.config: Config = init_config_from_yaml(
            project=project, logix_config=config
        )
        if logging_config is not None:
            self.set_logging_config(logging_config)
        self.log_dir: str = self.config.log_dir

        # LogIX state
        self.state: LogIXState = LogIXState()
        self.binfo: BatchInfo = BatchInfo()
        self._save: bool = False
        self._save_batch: bool = False

        # Initialize logger
        self.logger: HookLogger = HookLogger(
            config=self.get_logging_config(), state=self.state, binfo=self.binfo
        )

        # Log data
        self.log_dataset: Optional[torch.utils.data.Dataset] = None
        self.log_dataloader: Optional[torch.utils.data.DataLoader] = None

        # Analysis Modules - *** REMOVED direct instantiation ***
        self.influence: InfluenceFunction = InfluenceFunction(state=self.state)
        # Instead, analysis modules will be added via add_analysis

        # Misc
        self.type_filter: Optional[List[nn.Module]] = None
        self.name_filter: Optional[List[str]] = None

    def watch(
        self,
        model: nn.Module,
        type_filter: Optional[List[nn.Module]] = None,
        name_filter: Optional[List[str]] = None,
    ) -> None:
        """
        Sets up modules in the model to be watched based on optional type and name filters.
        Hooks will be added to the watched modules to track their forward activations,
        backward error signals, and gradient, and compute various statistics (e.g. mean,
        variance, covariance) for each of these.

        Args:
            model (nn.Module):
                The neural network model to be watched.
            type_filter (List[nn.Module], optional):
                A list of module types to be watched. If None, all supported module types
                are watched.
            name_filter (List[str], optional):
                A list of module names to be watched. If None, all modules are considered.
        """
        self.model = model
        if get_world_size() > 1 and hasattr(self.model, "module"):
            self.model = self.model.module
        self.type_filter = type_filter or self.type_filter
        self.name_filter = name_filter or self.name_filter

        _is_lora = is_lora(self.model)
        watched_module_names = set() # Keep track of modules explicitly watched


        for name, module in self.model.named_modules():

            # if "logix_lora_" in name or name.endswith("._linear"):
            #     continue 

            # if "logix_lora_" in name and not name.endswith("logix_lora_B"):
            #     continue

            if "logix_lora_" in name and not name.endswith("logix_lora_B"):
                continue


            # *** Check if this module should be watched ***
            should_watch = module_check(
                module=module,
                module_name=name,
                supported_modules=self._SUPPORTED_MODULES,
                type_filter=self.type_filter,
                name_filter=self.name_filter,
                is_lora=_is_lora,
            )

            if should_watch:
                self.logger.add_module(name, module)
                watched_module_names.add(name)
                # *** Ensure ALL parameters within the watched module require grad ***
                for p in module.parameters():
                    p.requires_grad = True
            # *** Modify the freezing logic ***
            # Freeze parameters ONLY IF the module itself isn't watched
            # AND it's a leaf module (has parameters but no children being watched)

            elif name.endswith("logix_lora_B") and hasattr(module, "_linear"):
                inner_name = f"{name}._linear"
                self.logger.add_module(inner_name, module._linear)
                watched_module_names.add(inner_name)
                for p in module._linear.parameters():
                    p.requires_grad = True


            elif not list(module.children()) and name not in watched_module_names:
                 # Check if any parent module was watched (to avoid freezing params of children of watched modules)
                 is_child_of_watched = False
                 for watched_name in watched_module_names:
                     if name.startswith(watched_name + "."):
                         is_child_of_watched = True
                         break
                 # Only freeze if it's a leaf and not a child of an already watched module
                 if not is_child_of_watched:
                     for p in module.parameters():
                         p.requires_grad = False


        # for name, module in self.model.named_modules():
        #     if module_check(
        #         module=module,
        #         module_name=name,
        #         supported_modules=self._SUPPORTED_MODULES,
        #         type_filter=self.type_filter,
        #         name_filter=self.name_filter,
        #         is_lora=_is_lora,
        #     ):
        #         self.logger.add_module(name, module)
        #     elif len(list(module.children())) == 0:
        #         # disable gradient compute for non-tracked modules. This will save a
        #         # significant amount of GPU memory and compute for large models
        #         for p in module.parameters():
        #             p.requires_grad = False

        module_path, repr_dim = get_repr_dim(self.logger.modules_to_name)

        # self.state.register_state("logger_modules_map", synchronize=False, save=True) # Register if not done
        # setattr(self.state, "logger_modules_map", self.logger.modules_to_name)

        self.state.register_state("logger_modules_map", synchronize=False, save=False)
        setattr(self.state, "logger_modules_map", self.logger.modules_to_name)




        # ----  add begin  ----
        if len(repr_dim) == 0:
            get_logger().warning(
                "watch() did not match any module ‑ check type/name filters."
            )
            return                      # Nothing to print / register
        # ----  add end  ----

        print_tracked_modules(reduce(lambda x, y: x + y, repr_dim))
        self.state.set_state("model_module", path=module_path, path_dim=repr_dim)

    def watch_activation(self, tensor_dict: Dict[str, torch.Tensor]) -> None:
        """
        Sets up tensors to be watched. This is useful for logging custom tensors or
        activations that are not directly associated with model modules.

        Args:
            tensor_dict (Dict[str, torch.Tensor]):
                A dictionary where keys are tensor names and values are the tensors
                themselves.
        """
        self.logger.register_all_tensor_hooks(tensor_dict)

    def add_lora(
        self,
        watch: bool = True,
        clear: bool = True,
        type_filter: Optional[List[nn.Module]] = None,
        name_filter: Optional[List[str]] = None,
        lora_path: Optional[str] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        """
        Adds an LoRA variant for gradient compression. Note that the added LoRA module
        is different from the original LoRA module in that it consists of three parts:
        encoder, bottleneck, and decoder. The encoder and decoder are initialized with
        random weights, while the bottleneck is initialized to be zero. In doing so,
        the encoder and decoder are repsectively used to compress forward and backward
        signals, while the bottleneck is used to extract the compressed gradient.
        Mathematically, this corresponds to random Kronecker product compression.

        Args:
            model (Optional[nn.Module]):
                The model to apply LoRA to. If None, the model already set with watch
                is used.
            watch (bool, optional):
                Whether to watch the model after adding LoRA. Defaults to `True`.
            clear (bool, optional):
                Whether to clear the internal states after adding LoRA. Defaults to
                `True`.
            type_filter (Optional[List[nn.Module]], optional):
                A list of module types to be watched.
            name_filter (Optional[List[str]], optional):
                A list of module names to be watched.
            lora_path (Optional[str], optional):
                The path to the LoRA state file. If None, the LoRA state is not loaded.
            lora_config (Optional[LoRAConfig], optional): LoRA configuration.
        """
        if lora_config is not None:
            self.set_lora_config(lora_config)

        if not hasattr(self, "lora_handler"):
            self.lora_handler = LoRAHandler(self.get_lora_config(), self.state)

        self.lora_handler.add_lora(
            model=self.model,
            type_filter=type_filter or self.type_filter,
            name_filter=name_filter or self.name_filter,
        )

        # If lora_path is not none, directly load lora weights from this path
        if lora_path is not None:
            lora_dir = os.path.join(os.path.join(lora_path, "lora"))
            lora_state = torch.load(os.path.join(lora_dir, "lora_state_dict.pt"))
            for name in lora_state:
                assert name in self.model.state_dict(), f"{name} not in model!"
            self.model.load_state_dict(lora_state, strict=False)

        # Save lora state dict
        if get_rank() == 0:
            self.save_lora()

        # Clear state and logger
        if clear or watch:
            msg = "LogIX will clear the previous Hessian, Storage, and Logging "
            msg += "handlers after adding LoRA for gradient compression.\n"
            get_logger().info(msg)
            self.clear()

        # (Re-)watch lora-added model
        if watch:
            self.watch(self.model)

    def log(self, data_id: Iterable[Any], mask: Optional[torch.Tensor] = None) -> None:
        """
        Logs the data. This is an experimental feature for now.

        Args:
            data_id (str):
                A unique identifier associated with the data for the logging session.
            mask (Optional[torch.Tensor], optional):
                An optional attention mask used in Transformer models.
        """
        self.logger.log(data_id=data_id, mask=mask)

    def __call__(
        self,
        data_id: Iterable[Any],
        mask: Optional[torch.Tensor] = None,
        save: bool = False,
    ):
        """
        Args:
            data_id: A unique identifier associated with the data for the logging session.
            mask (torch.Tensor, optional): Mask for the data.

        Returns:
            self: Returns the instance of the LogIX object.
        """
        self.binfo.clear()
        self.binfo.data_id = data_id
        self.binfo.mask = mask

        self._save_batch = save

        return self

    def __enter__(self):
        """
        Sets up the context manager.

        This method is automatically called when the `with` statement is used with an `LogIX` object.
        It sets up the logging environment based on the provided parameters.
        """
        self.logger.clear(hook=True, module=False, buffer=False)
        self.logger.register_all_module_hooks()

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Clears the internal states and removes all the hooks.

        This method is essential for ensuring that there are no lingering hooks that could
        interfere with further operations on the model or with future logging sessions.
        """
        self.logger.update(save=self._save_batch or self._save)

    def start(
        self, data_id: Iterable[Any], mask: Optional[torch.Tensor] = None
    ) -> None:
        """
        This is another programming interface for logging. Instead of using the context manager, we also
        allow users to manually specify `start` and `end` points for logging. In the `start` point,
        users should specify `data_id` and optionally `mask`.

        Args:
            data_id: A unique identifier associated with the data for the logging session.
            mask (torch.Tensor, optional): Mask for the data.
        """
        # Set up batch information
        self.binfo.clear()
        self.binfo.data_id = data_id
        self.binfo.mask = mask

        # Set up logger
        self.logger.clear(hook=True, module=False, buffer=False)
        self.logger.register_all_module_hooks()

    def end(self, save: bool = False) -> None:
        """
        This is another programming interface for logging. Instead of using the context manager, we also
        allow users to manually specify "start" and "end" points for logging.
        """
        self.logger.update(save=save or self._save)

    def build_log_dataset(self, flatten: bool = False) -> torch.utils.data.Dataset:
        """
        Constructs the log dataset from the stored logs. This dataset can then be used
        for analysis or visualization.

        Args:
            flatten (bool, optional): Whether to flatten the nested dictionary structure. Defaults to False.

        Returns:
            LogDataset:
                An instance of LogDataset containing the logged data.
        """
        if self.log_dataset is None:
            self.log_dataset = LogDataset(log_dir=self.log_dir, flatten=flatten)
        return self.log_dataset

    def build_log_dataloader(
        self,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        flatten: bool = False,
    ) -> torch.utils.data.DataLoader:
        """
        Constructs a DataLoader for the log dataset. This is useful for batch processing
        of logged data during analysis. It also follows PyTorch DataLoader conventions.

        Args:
            batch_size (int, optional): The batch size for the DataLoader. Defaults to 16.
            num_workers (int, optional): The number of workers for the DataLoader. Defaults to 0.
            pin_memory (bool, optional): Whether to pin memory for the DataLoader. Defaults to False.
            flatten (bool, optional): Whether to flatten the nested dictionary structure. Defaults to False.

        Return:
            DataLoader:
                A DataLoader instance for the log dataset.
        """
        if self.log_dataloader is None:
            log_dataset = self.build_log_dataset(flatten=flatten)
            collate_fn = None if flatten else collate_nested_dicts
            self.log_dataloader = torch.utils.data.DataLoader(
                log_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle=False,
                collate_fn=collate_fn,
            )
        return self.log_dataloader

    def get_log(
        self, copy: bool = False
    ) -> Tuple[str, Dict[str, Dict[str, torch.Tensor]]]:
        """
        Returns the current log, including data identifiers and logged information.

        Args:
           copy (bool, optional): Whether to return a copy of the log. Defaults to False.

        Returns:
            dict: The current log.
        """
        log = (self.binfo.data_id, self.binfo.log)
        return log if not copy else deepcopy(log)



    # --- *** NEW: add_analysis method *** ---
    def add_analysis(self, analysis_mapping: Dict[str, type]) -> None:
        """
        Adds analysis modules to the LogIX instance.

        Args:
            analysis_mapping (Dict[str, type]): A dictionary where keys are
                the desired attribute names for the analysis modules (e.g., "influence",
                "influence_fhe") and values are the corresponding analysis class types
                (e.g., InfluenceFunction, InfluenceFunctionFHE).
        """
        for name, analysis_class in analysis_mapping.items():
            if hasattr(self, name):
                get_logger().warning(f"Analysis module '{name}' already exists. Overwriting.")

            if analysis_class == InfluenceFunction:
                setattr(self, name, InfluenceFunction(self.state))
                get_logger().info(f"Added plaintext Influence Function analysis as 'run.{name}'")
            elif (
                (InfluenceFunctionFHE is not None and analysis_class == InfluenceFunctionFHE)
                or getattr(analysis_class, "__name__", "") == "InfluenceFunctionFHE"
            ):
                if InfluenceFunctionFHE is None:
                    err_msg = (
                        "InfluenceFunctionFHE is unavailable because optional FHE "
                        "dependencies failed to import."
                    )
                    if INFLUENCE_FHE_IMPORT_ERROR is not None:
                        err_msg += f" Original error: {INFLUENCE_FHE_IMPORT_ERROR}"
                    get_logger().error(err_msg)
                    raise ImportError(err_msg)
                # Check if FHE config exists
                if not hasattr(self.config, 'fhe') or self.config.fhe is None:
                     get_logger().error("FHE configuration not found in LogIX config. Cannot add InfluenceFunctionFHE.")
                     raise ValueError("FHE configuration missing.")
                setattr(self, name, InfluenceFunctionFHE(self.state, self.config.fhe))
                get_logger().info(f"Added FHE Influence Function analysis as 'run.{name}'")
            else:
                # Placeholder for potentially other analysis types
                 get_logger().warning(f"Unsupported analysis class type: {analysis_class}. Skipping '{name}'.")
                # Example: setattr(self, name, analysis_class(self.state, self.config.some_other_config))







    # def compute_influence(
    #     self,
    #     src_log: Tuple[str, Dict[str, Dict[str, torch.Tensor]]],
    #     tgt_log: Tuple[str, Dict[str, Dict[str, torch.Tensor]]],
    #     mode: Optional[str] = "dot",
    #     precondition: Optional[bool] = True,
    #     hessian: Optional[str] = "auto",
    #     influence_groups: Optional[List[str]] = None,
    #     damping: Optional[float] = None,
    # ) -> Dict[str, Union[List[str], torch.Tensor, Dict[str, torch.Tensor]]]:
    #     """
    #     Front-end interface for computing influence scores. It calls the
    #     `compute_influence` method of the `InfluenceFunction` class.

    #     Args:
    #         src_log (Tuple[str, Dict[str, Dict[str, torch.Tensor]]]): Log of source gradients
    #         tgt_log (Tuple[str, Dict[str, Dict[str, torch.Tensor]]]): Log of target gradients
    #         mode (str, optional): Influence function mode. Defaults to "dot".
    #         precondition (bool, optional): Whether to precondition the gradients. Defaults to True.
    #         hessian (str, optional): Hessian computation mode. Defaults to "auto".
    #         influence_groups (List[str], optional): List of influence groups. Defaults to None.
    #         damping (float, optional): Damping factor. Defaults to None.
    #     """
    #     result = self.influence.compute_influence(
    #         src_log=src_log,
    #         tgt_log=tgt_log,
    #         mode=mode,
    #         precondition=precondition,
    #         hessian=hessian,
    #         influence_groups=influence_groups,
    #         damping=damping,
    #     )
    #     return result

    # def compute_influence_all(
    #     self,
    #     src_log: Optional[Tuple[str, Dict[str, Dict[str, torch.Tensor]]]] = None,
    #     loader: Optional[torch.utils.data.DataLoader] = None,
    #     mode: Optional[str] = "dot",
    #     precondition: Optional[bool] = True,
    #     hessian: Optional[str] = "auto",
    #     influence_groups: Optional[List[str]] = None,
    #     damping: Optional[float] = None,
    # ) -> Dict[str, Union[List[str], torch.Tensor, Dict[str, torch.Tensor]]]:
    #     """
    #     Front-end interface for computing influence scores against all train data in the log.
    #     It calls the `compute_influence_all` method of the `InfluenceFunction` class.

    #     Args:
    #         src_log (Tuple[str, Dict[str, Dict[str, torch.Tensor]]]): Log of source gradients
    #         loader (torch.utils.data.DataLoader): DataLoader of train data
    #         mode (str, optional): Influence function mode. Defaults to "dot".
    #         precondition (bool, optional): Whether to precondition the gradients. Defaults to True.
    #         hessian (str, optional): Hessian computation mode. Defaults to "auto".
    #         influence_groups (List[str], optional): List of influence groups. Defaults to None.
    #         damping (float, optional): Damping factor. Defaults to None.
    #     """
    #     src_log = src_log if src_log is not None else self.get_log()
    #     loader = loader if loader is not None else self.build_log_dataloader()

    #     result = self.influence.compute_influence_all(
    #         src_log=src_log,
    #         loader=loader,
    #         mode=mode,
    #         precondition=precondition,
    #         hessian=hessian,
    #         influence_groups=influence_groups,
    #         damping=damping,
    #     )
    #     return result

    # def compute_self_influence(
    #     self,
    #     src_log: Optional[Tuple[str, Dict[str, Dict[str, torch.Tensor]]]] = None,
    #     precondition: Optional[bool] = True,
    #     hessian: Optional[str] = "auto",
    #     influence_groups: Optional[List[str]] = None,
    #     damping: Optional[float] = None,
    # ) -> Dict[str, Union[List[str], torch.Tensor, Dict[str, torch.Tensor]]]:
    #     """
    #     Front-end interface for computing self-influence scores. It calls the
    #     `compute_self_influence` method of the `InfluenceFunction` class.

    #     Args:
    #         src_log (Tuple[str, Dict[str, Dict[str, torch.Tensor]]]): Log of source gradients
    #         precondition (bool, optional): Whether to precondition the gradients. Defaults to True.
    #         heissian (str, optional): Hessian computation mode. Defaults to "auto".
    #         influence_groups (List[str], optional): List of influence groups. Defaults to None.
    #         damping (float, optional): Damping factor. Defaults to None.
    #     """
    #     src_log = src_log if src_log is not None else self.get_log()
    #     result = self.influence.compute_self_influence(
    #         src_log=src_log,
    #         precondition=precondition,
    #         hessian=hessian,
    #         influence_groups=influence_groups,
    #         damping=damping,
    #     )
    #     return result




    def save_lora(self) -> None:
        """
        Save LoRA state to disk.
        """
        state_dict = self.model.state_dict()
        lora_state_dict = {
            name: param for name, param in state_dict.items() if "logix_lora" in name
        }
        if len(lora_state_dict) > 0:
            log_dir = os.path.join(self.log_dir, "lora")
            if not os.path.exists(log_dir) and get_rank() == 0:
                os.makedirs(log_dir)
            torch.save(lora_state_dict, os.path.join(log_dir, "lora_state_dict.pt"))

    def initialize_from_log(
        self, state_path: Optional[str] = None, lora_path: Optional[str] = None
    ) -> None:
        """
        Load all states from disk.

        Args:
            state_path (str, optional): Path to the state file.
            lora_path (str, optional): Path to the LoRA state file.
        """
        assert os.path.exists(self.log_dir), f"{self.log_dir} does not exist!"

        # Load config
        self.config.load_config(os.path.join(self.log_dir, "config.yaml"))

        # Load LoRA state
        lora_dir = os.path.join(lora_path or self.log_dir, "lora")
        if os.path.exists(lora_dir) and self.model is not None:
            if not is_lora(self.model):
                self.add_lora()
            lora_state = torch.load(os.path.join(lora_dir, "lora_state_dict.pt"))
            for name in lora_state:
                assert name in self.model.state_dict(), f"{name} not in model!"
            self.model.load_state_dict(lora_state, strict=False)

        # Load state
        self.state.load_state(state_path or self.log_dir)

    def finalize(
        self,
    ) -> None:
        """
        Finalizes the logging session.
        """
        # Finalizing `state` synchronizes the state across all processes
        self.state.finalize(log_dir=self.log_dir)

        # Finalizing `logger` flushes and writes the remaining log buffer to disk
        self.logger.finalize()

        # Save configurations
        self.config.save_config(log_dir=self.log_dir)

    def setup(self, log_option_kwargs: Dict[str, Any]) -> None:
        """
        Update logging configurations.

        Args:
            log_option_kwargs: Logging configurations.
        """
        self.logger.opt.setup(log_option_kwargs)

    def save(self, enable: bool = True) -> None:
        """
        Turn on saving.
        """
        self._save = enable

    def eval(self) -> None:
        """
        Set the state of LogIX for testing.
        """
        self.save(False)

    def clear(self) -> None:
        """
        Clear everything in LogIX.
        """
        self.state.clear()
        self.logger.clear()

    def set_logging_config(self, logging_config: LoggingConfig) -> None:
        logging_config.log_dir = self.config.log_dir
        self.config.logging = logging_config

    def set_lora_config(self, lora_config: LoRAConfig) -> None:
        self.config.lora = lora_config

    def get_logging_config(self) -> LoggingConfig:
        return self.config.logging

    def get_lora_config(self) -> LoRAConfig:
        return self.config.lora

    # --- *** NEW: Method to get FHE context (optional) *** ---
    def get_fhe_context(self) -> Optional[Any]:
         """
         Returns the Pyfhel HE context if an FHE analysis module exists.
         Requires the analysis module to store the context as self.HE.
         Example usage: run.get_fhe_context() after run.add_analysis({'influence_fhe': InfluenceFunctionFHE})
         """
         # Search for an analysis module that looks like our FHE one
         fhe_module = None
         for attr_name in dir(self):
             attr = getattr(self, attr_name)
             # A bit heuristic: check if it's an instance of InfluenceFunctionFHE
             # This requires InfluenceFunctionFHE to be imported here, which might cause circular dependency
             # Alternative: check for existence of HE attribute.
             if hasattr(attr, "HE"):
                  he_ctx = getattr(attr, "HE", None)
                  if _PyfhelType is None:
                      if he_ctx is not None:
                          fhe_module = attr
                          break
                  elif isinstance(he_ctx, _PyfhelType):
                      fhe_module = attr
                      break

         if fhe_module:
             return fhe_module.HE
         else:
             get_logger().warning("No FHE analysis module with HE context found.")
             return None
