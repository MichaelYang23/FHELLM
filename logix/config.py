# Proposed content for: logix/config.py
# (Showing modifications and additions to the existing file)

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
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional, Union
from logix.utils import get_logger


import torch
import yaml

from logix.utils import get_rank


# --- Helper functions (init_config_from_yaml, load_config_from_dict) remain the same ---
def init_config_from_yaml(project: str, logix_config: Optional[str] = None):
    config_dict = {}
    if logix_config is not None:
        # Check if logix_config exists before opening
        if not os.path.exists(logix_config):
            # If config file doesn't exist, log a warning but proceed with defaults
             get_logger().warning(f"Configuration file not found at {logix_config}. Using default settings.")
        else:
            try:
                with open(logix_config, "r", encoding="utf-8") as config_file:
                    config_dict = yaml.safe_load(config_file)
                    if config_dict is None: # Handle empty YAML file
                        config_dict = {}
            except Exception as e:
                 get_logger().error(f"Error loading configuration file {logix_config}: {e}. Using default settings.")
                 config_dict = {}


    assert project is not None
    config_dict["project"] = project

    # Ensure nested configs are dictionaries if provided partially
    if 'logging' not in config_dict: config_dict['logging'] = {}
    if 'lora' not in config_dict: config_dict['lora'] = {}
    if 'fhe' not in config_dict: config_dict['fhe'] = {} # Add for FHE

    return Config(**config_dict)


def load_config_from_dict(config, config_dict: Dict[str, Any]):
    for name, value in config_dict.items():
        if hasattr(config, name): # Check if attribute exists before setting
            if isinstance(value, dict) and is_dataclass(getattr(config, name, None)):
                 load_config_from_dict(getattr(config, name), value)
            elif not isinstance(value, dict) or not is_dataclass(getattr(config, name, None)):
                 # Avoid overwriting a dataclass field with a non-dict value accidentally
                 # Or handle simple attribute setting
                 try:
                     setattr(config, name, value)
                 except AttributeError:
                      get_logger().warning(f"Could not set attribute {name} in config.")
        else:
            get_logger().warning(f"Attribute {name} not found in config class {type(config).__name__}. Skipping.")


@dataclass
class LoggingConfig:
    """
    Configuration for logging.

    Args:
        flush_threshold: Flush threshold for the buffer.
        num_workers: Number of workers used for log saving.
        cpu_offload: Offload statistic states to CPU.
        log_dtype: Data type for logging ('float32', 'float16', 'bfloat16', etc.)
    """

    log_dir: str = field(init=False) # Set in Config.__post_init__
    flush_threshold: int = field(
        default=1000000000, metadata={"help": "Flush threshold for the log buffer."}
    )
    num_workers: int = field(
        default=1, metadata={"help": "Number of workers used for logging."}
    )
    cpu_offload: int = field(
        default=False, metadata={"help": "Offload statistic states to CPU."}
    )
    log_dtype: str = field(default="float32", metadata={"help": "Data type for logging."}) # Changed default

    def get_dtype(self):
        if self.log_dtype == "none" or self.log_dtype is None:
            return None
        elif self.log_dtype == "float64":
            return torch.float64
        elif self.log_dtype == "float16":
            return torch.float16
        elif self.log_dtype == "bfloat16":
            return torch.bfloat16
        elif self.log_dtype == "int8":
            return torch.int8
        elif self.log_dtype == "float32":
            return torch.float32
        else:
            get_logger().warning(f"Unsupported log_dtype '{self.log_dtype}'. Defaulting to float32.")
            return torch.float32


@dataclass
class LoRAConfig:
    """
    Configuration for LoRA.

    Args:
        init: Initialization method for LoRA ('random', 'pca').
        rank: Rank for LoRA.
        parameter_sharing: Parameter sharing for LoRA.
        parameter_sharing_groups: Parameter sharing groups for LoRA.
    """

    init: str = field(
        default="random", metadata={"help": "Initialization method for LoRA ('random', 'pca')."}
    )
    rank: int = field(default=64, metadata={"help": "Rank for LoRA."})
    parameter_sharing: bool = field(
        default=False, metadata={"help": "Parameter sharing for LoRA."}
    )
    parameter_sharing_groups: Optional[List[str]] = field(
        default=None, metadata={"help": "Parameter sharing groups for LoRA."}
    )


# --- NEW: Define FHEConfig Dataclass ---
@dataclass
class FHEConfig:
    """
    Configuration for Fully Homomorphic Encryption (FHE) using CKKS.

    Args:
        scheme: The FHE scheme to use (currently only 'CKKS' relevant).
        n: Polynomial modulus degree. Must be a power of 2. Affects security and performance.
        scale: Scale factor for float-to-fixed-point conversion. Affects precision.
               Should ideally relate to intermediate qi_sizes.
        qi_sizes: List of bit sizes for the primes in the coefficient modulus chain.
                  Affects the number of multiplications possible (depth) and precision.
                  Example: [60, 30, 30, 60] for depth 2 multiplication.
    """
    scheme: str = field(default="CKKS", metadata={"help": "FHE scheme (should be CKKS)."})
    n: int = field(default=2**15, metadata={"help": "Polynomial modulus degree (power of 2)."})
    scale: float = field(default=2**40, metadata={"help": "Scale for fixed-point encoding."})
    qi_sizes: List[int] = field(default_factory=lambda: [60, 40, 40, 40, 60], metadata={"help": "List of prime bit sizes for CKKS modulus chain."})

    def __post_init__(self):
         # Basic validation
         if self.scheme.upper() != 'CKKS':
              raise ValueError("LogIX FHE integration currently only supports 'CKKS' scheme.")
         if not (self.n > 0 and (self.n & (self.n - 1) == 0)): # Check if power of 2
              raise ValueError(f"FHE parameter 'n' must be a power of 2, but got {self.n}")
         if not isinstance(self.qi_sizes, list) or len(self.qi_sizes) < 2:
              raise ValueError("'qi_sizes' must be a list of at least two integers.")


@dataclass
class InfluenceConfig:
    """
    Configuration for influence function computation (remains for plaintext version).

    Args:
        damping: Damping for influence.
        relative_damping: Compute the damping term based on singular values.
        mode: Mode for influence ('dot', 'cosine', 'l2').
        flatten: Whether to flatten gradients during loading.
    """
    log_dir: str = field(init=False) # Set in Config.__post_init__
    damping: Optional[float] = field( # Allow None
        default=1e-5, metadata={"help": "Damping strength for influence."}
    )
    relative_damping: bool = field(
        default=False,
        metadata={"help": "Compute the damping term based on singular values."},
    )
    mode: str = field(default="dot", metadata={"help": "Mode for influence ('dot', 'cosine', 'l2')."})
    flatten: bool = field(
        default=False, metadata={"help": "Whether to flatten logs when loading"}
    )


@dataclass
class Config:
    """
    Main LogIX Configuration class.
    Loads configurations from a YAML file and provides access to component-specific configurations.

    Args:
        project: Project name.
        root_dir: Root directory for logging.
        logging: Logging configuration.
        lora: LoRA configuration.
        fhe: FHE configuration. (NEW)
        influence: Influence function configuration. (NEW - separated from analysis for clarity)
    """

    project: str
    log_dir: str = field(init=False)
    root_dir: str = field(
        default="./logix_logs", metadata={"help": "Root directory for logging."} # Changed default
    )
    logging: Union[Dict[str, Any], LoggingConfig] = field(
        default_factory=LoggingConfig, metadata={"help": "Logging configuration."}
    )
    lora: Union[Dict[str, Any], LoRAConfig] = field(
        default_factory=LoRAConfig, metadata={"help": "LoRA configuration."}
    )
    # --- Add FHE and Influence config fields ---
    fhe: Union[Dict[str, Any], FHEConfig] = field(
        default_factory=FHEConfig, metadata={"help": "FHE (CKKS) configuration."}
    )
    influence: Union[Dict[str, Any], InfluenceConfig] = field(
        default_factory=InfluenceConfig, metadata={"help": "Influence function configuration."}
    )

    def __post_init__(self):
        # Convert dictionary subsections to dataclass instances if needed
        if isinstance(self.logging, dict):
            self.logging = LoggingConfig(**self.logging)
        if isinstance(self.lora, dict):
            self.lora = LoRAConfig(**self.lora)
        if isinstance(self.fhe, dict): # Handle FHE config
            self.fhe = FHEConfig(**self.fhe)
        if isinstance(self.influence, dict): # Handle Influence config
            self.influence = InfluenceConfig(**self.influence)

        self.log_dir = None
        self.configure_log_dir()

    def configure_log_dir(self) -> None:
        """
        Set single logging directory for all components.
        """
        self.log_dir = os.path.join(self.root_dir, self.project)
        self.logging.log_dir = self.log_dir
        self.influence.log_dir = self.log_dir # Set log_dir for influence config too

        # Create log directory if it doesn't exist (only on rank 0)
        if not os.path.exists(self.log_dir) and get_rank() == 0:
            try:
                 os.makedirs(self.log_dir, exist_ok=True) # Use exist_ok=True
            except OSError as e:
                 get_logger().error(f"Failed to create log directory {self.log_dir}: {e}")
                 raise

    def load_config(self, logix_config: str) -> None:
        """
        Load configuration from the saved YAML file.

        Args:
            logix_config: Path to the saved YAML file.
        """
        config_dict = {}
        if not os.path.exists(logix_config):
             get_logger().warning(f"Configuration file {logix_config} not found. Loading defaults.")
        else:
             try:
                 with open(logix_config, "r", encoding="utf-8") as config_file:
                     config_dict = yaml.safe_load(config_file)
                     if config_dict is None: config_dict = {}
             except Exception as e:
                 get_logger().error(f"Error loading configuration file {logix_config}: {e}. Loading defaults.")
                 config_dict = {}

        # Make sure project name isn't overwritten if it exists in file
        if 'project' in config_dict and config_dict['project'] != self.project:
            get_logger().warning(f"Overwriting initial project name '{self.project}' with value from config file '{config_dict['project']}'.")
            self.project = config_dict['project']

        # Update attributes using the loaded dictionary
        load_config_from_dict(self, config_dict)
        self.configure_log_dir() # Reconfigure log dir based on potentially updated root_dir/project

    def save_config(self, log_dir: Optional[str] = None) -> None:
        """
        Save configuration to a YAML file in the specified directory.
        Only rank 0 saves the file in distributed settings.
        """
        if get_rank() == 0:
            save_directory = log_dir or self.log_dir
            if not os.path.exists(save_directory):
                 try:
                      os.makedirs(save_directory, exist_ok=True)
                 except OSError as e:
                      get_logger().error(f"Failed to create directory {save_directory} for saving config: {e}")
                      return # Abort saving if directory creation fails

            config_file = os.path.join(save_directory, "config.yaml")
            # Convert the current config state (which includes FHEConfig etc.) to dict
            config_dict = asdict(self)

            try:
                with open(config_file, "w", encoding="utf-8") as f:
                    yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
                    get_logger().info(f"Configuration saved to {config_file}")
            except Exception as e:
                get_logger().error(f"Failed to save configuration to {config_file}: {e}")