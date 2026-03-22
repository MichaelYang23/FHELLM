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

from typing import Any, Iterable, Optional

import torch

from logix.utils import nested_dict


class BatchInfo:
    def __init__(self):
        self.data_id: Optional[Iterable[Any]] = None
        self.mask: Optional[torch.Tensor] = None
        self.log = nested_dict()
        # === ADDED ===
        self.hook_inputs = nested_dict() # Stores forward inputs from _grad_hook_fn
        self.hook_output_grads = nested_dict() # Stores output grads from _backward_hook_fn
        # === END ADDED ===

    def clear(self):
        self.data_id = None
        self.mask = None
        self.log.clear()
        # === ADDED ===
        self.hook_inputs.clear()
        self.hook_output_grads.clear()
        # === END ADDED ===