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

from functools import partial
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from logix.batch_info import BatchInfo
from logix.config import LoggingConfig
from logix.logging.log_saver import LogSaver
from logix.logging.option import LogOption
from logix.logging.utils import compute_per_sample_gradient
from logix.state import LogIXState
from logix.statistic import Log


class HookLogger:
    def __init__(
        self,
        config: LoggingConfig,
        state: LogIXState,
        binfo: BatchInfo,
    ) -> None:
        """
        Initializes the LoggingHandler with empty lists for hooks.
        """
        self.state = state
        self.binfo = binfo
        self.opt = LogOption()

        # parse config
        self.cpu_offload = config.cpu_offload
        self.dtype = config.get_dtype()

        # log saver
        self.log_saver = LogSaver(config=config, state=self.state)

        # hooks
        self.modules_to_hook = []
        self.modules_to_name = {}
        self.forward_hooks = []
        self.backward_hooks = []
        self.grad_hooks = []
        self.tensor_hooks = []

    def log(self, data_id: Any, mask: Optional[torch.Tensor] = None):
        """
        Add log state on exit.

        Args:
            data_id: The data ID associated with the current batch.
            mask (torch.Tensor): A mask to be applied to the activations.
        """
        log = self.binfo.log.copy()
        self.binfo.clear()
        self.binfo.data_id = data_id
        self.binfo.mask = mask

        self.update()

        return log

    def save_log(self):
        # save log to disk
        self.log_saver.buffer_write(binfo=self.binfo)
        self.log_saver.flush()

    def update(self, save: bool = False):
        # gradient plugin has to be excecuted after accumulating all gradients
        for stat in self.opt.grad[1:]:
            for module_name, _ in self.binfo.log.items():
                stat.update(
                    state=self.state,
                    binfo=self.binfo,
                    module=None,
                    module_name=module_name,
                    log_type="grad",
                    data=None,
                    cpu_offload=self.cpu_offload,
                )

        # Wait for all asynchronous CUDA operations to finish
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()

        # Write and flush the buffer if necessary
        if save:
            self.save_log()

    def _forward_hook_fn(
        self, module: nn.Module, inputs: Tuple[torch.Tensor], module_name: str
    ) -> None:
        """
        Internal forward hook function.

        Args:
            module: The module triggering the hook.
            inputs: The input to the module.
            module_name (str): The name of the module.
        """
        assert len(inputs) == 1

        activations = inputs[0]

        # If `mask` is not None, apply the mask to activations. This is
        # useful for example when you work with sequence models that use
        # padding. In this case, you can use the mask to ignore the padding
        if self.binfo.mask is not None:
            mask = self.binfo.mask
            if len(mask.shape) != len(activations.shape):
                assert len(mask.shape) == len(activations.shape) - 1
                if mask.shape[-1] == activations.shape[-2]:
                    activations = activations * mask.unsqueeze(-1)
            else:
                if mask.shape[-1] == activations.shape[-1]:
                    activations = activations * mask

        if self.dtype is not None:
            activations = activations.to(dtype=self.dtype)

        for plugin in self.opt.forward:
            plugin.update(
                state=self.state,
                binfo=self.binfo,
                module=module,
                module_name=module_name,
                log_type="forward",
                data=activations,
                cpu_offload=self.cpu_offload,
            )

    def _backward_hook_fn(
        self,
        module: nn.Module,
        grad_inputs: Tuple[torch.Tensor],
        grad_outputs: Tuple[torch.Tensor],
        module_name: str,
    ) -> None:
        """
        Internal backward hook function.

        Args:
            module: The module triggering the hook.
            grad_inputs: The gradient of the input to the module.
            grad_outputs: The gradient of the output from the module.
            module_name (str): The name of the module.
        """
        assert len(grad_outputs) == 1

        error = grad_outputs[0]

        if self.dtype is not None:
            error = error.to(dtype=self.dtype)

        for plugin in self.opt.backward:
            plugin.update(
                state=self.state,
                binfo=self.binfo,
                module=module,
                module_name=module_name,
                log_type="backward",
                data=error,
                cpu_offload=self.cpu_offload,
            )

    def _grad_hook_fn(
        self,
        module: nn.Module,
        inputs: Tuple[torch.Tensor],
        outputs: Tuple[torch.Tensor],
        module_name: str,
    ) -> None:
        """
        Internal gradient hook function.

        Args:
            module: The module triggering the hook.
            inputs: The input to the module.
            outputs: The output from the module.
            module_name (str): The name of the module.
        """
        assert len(inputs) == 1

        # In case, the same module is used multiple times in the forward pass,
        # we need to accumulate the gradients. We achieve this by using the
        # additional tensor hook on the output of the module.
        def _grad_backward_hook_fn(grad: torch.Tensor):
            if len(self.opt.grad) > 0:
                assert self.opt.grad[0] == Log
                per_sample_gradient = compute_per_sample_gradient(
                    inputs[0], grad, module
                )

                if self.dtype is not None:
                    per_sample_gradient = per_sample_gradient.to(dtype=self.dtype)

                for plugin in self.opt.grad[:1]:
                    plugin.update(
                        state=self.state,
                        binfo=self.binfo,
                        module=module,
                        module_name=module_name,
                        log_type="grad",
                        data=per_sample_gradient,
                        cpu_offload=self.cpu_offload,
                    )

        tensor_hook = outputs.register_hook(_grad_backward_hook_fn)
        self.tensor_hooks.append(tensor_hook)

    def _tensor_forward_hook_fn(self, tensor: torch.Tensor, tensor_name: str) -> None:
        """
        Internal forward hook function specifically designed for tensors.

        This method allows you to track the activations of tensors that are not
        necessarily tied to any specific module, but are still of interest.

        Args:
            tensor: The tensor triggering the hook.
            tensor_name (str): A string identifier for the tensor, useful for logging.
        """
        if self.dtype is not None:
            tensor = tensor.to(dtype=self.dtype)

        for plugin in self.opt.forward:
            plugin.update(
                state=self.state,
                binfo=self.binfo,
                module=None,
                module_name=tensor_name,
                log_type="forward",
                data=tensor,
                cpu_offload=self.cpu_offload,
            )

    def _tensor_backward_hook_fn(self, grad: torch.Tensor, tensor_name: str) -> None:
        """
        Internal backward hook function specifically designed for tensors.

        This method allows you to track the gradients associated with specific
        tensors that are not necessarily tied to any specific module, but are still of interest.

        Args:
            grad: The gradient tensor triggering the hook.
            tensor_name (str): A string identifier for the tensor whose gradient is being tracked.
        """
        if self.dtype is not None:
            grad = grad.to(dtype=self.dtype)

        for plugin in self.opt.backward:
            plugin.update(
                state=self.state,
                binfo=self.binfo,
                module=None,
                module_name=tensor_name,
                log_type="backward",
                data=grad,
                cpu_offload=self.cpu_offload,
            )

    def register_all_module_hooks(self) -> None:
        """
        Register all module hooks.
        """
        for module in self.modules_to_hook:
            # As each hook has its own function, we need to register all hooks
            # separately. We use partial functions to pass the module name to
            # the hook functions.
            module_name = self.modules_to_name[module]
            forward_hook = module.register_forward_pre_hook(
                partial(self._forward_hook_fn, module_name=module_name)
            )
            backward_hook = module.register_full_backward_hook(
                partial(self._backward_hook_fn, module_name=module_name)
            )
            grad_hook = module.register_forward_hook(
                partial(self._grad_hook_fn, module_name=module_name)
            )
            self.forward_hooks.append(forward_hook)
            self.backward_hooks.append(backward_hook)
            self.grad_hooks.append(grad_hook)

    def register_all_tensor_hooks(self, tensor_dict: Dict[str, torch.Tensor]) -> None:
        """
        Register all tensor hooks.

        Args:
            tensor_dict (dict): Dictionary containing tensor names as keys and tensors as values.
        """
        for tensor_name, tensor in tensor_dict.items():
            self._tensor_forward_hook_fn(tensor, tensor_name)
            tensor_hook = tensor.register_hook(
                partial(self._tensor_backward_hook_fn, tensor_name=tensor_name)
            )
            self.tensor_hooks.append(tensor_hook)

    def add_module(self, module_name: str, module: nn.Module) -> None:
        """
        Add a module to be hooked.

        Args:
            module_name (str): The name of the module.
            module: The module to be hooked.
        """
        self.modules_to_hook.append(module)
        self.modules_to_name[module] = module_name

    def setup(self, log_option_kwargs: Dict[str, Any]) -> None:
        """
        Update logging configurations.

        Args:
            log_option_kwargs: Logging configurations.
        """
        self.opt.setup(log_option_kwargs)

    def finalize(self):
        """
        Dump everything in the buffer to a disk.
        """
        self.log_saver.finalize()

    def clear(
        self, hook: bool = True, module: bool = True, buffer: bool = True
    ) -> None:
        """
        Clear all hooks and internal states.

        Args:
            hook (bool): Whether to clear the hooks.
            module (bool): Whether to clear the modules.
        """
        if hook:
            for hook in self.forward_hooks:
                hook.remove()
            self.forward_hooks.clear()
            for hook in self.backward_hooks:
                hook.remove()
            self.backward_hooks.clear()
            for hook in self.grad_hooks:
                hook.remove()
            self.grad_hooks.clear()
            for hook in self.tensor_hooks:
                hook.remove()
            self.tensor_hooks.clear()

        if module:
            self.modules_to_hook = []
            self.modules_to_name = {}

        if buffer:
            self.log_saver.buffer_clear()


# # Copyright 2023-present the LogIX team.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# # --- Add necessary imports ---
# from functools import partial
# from typing import Any, Dict, Optional, Tuple
# import torch
# import torch.nn as nn
# from logix.batch_info import BatchInfo
# from logix.config import LoggingConfig
# from logix.logging.log_saver import LogSaver
# from logix.logging.option import LogOption
# # *** Make sure compute_per_sample_gradient is imported ***
# from logix.logging.utils import compute_per_sample_gradient
# from logix.state import LogIXState
# # *** Make sure Log statistic is imported if used below ***
# from logix.statistic import Log
# from logix.utils import get_logger # Import get_logger
# # --- End Imports ---


# class HookLogger:
#     def __init__(
#         self,
#         config: LoggingConfig,
#         state: LogIXState,
#         binfo: BatchInfo,
#         # # *** Add modules_to_name mapping ***
#         # modules_to_name: Dict[nn.Module, str]
#     ) -> None:
#         """
#         Initializes the LoggingHandler with empty lists for hooks.
#         """
#         self.state = state
#         self.binfo = binfo
#         self.opt = LogOption()

#         # # *** Store modules_to_name ***
#         # self.modules_to_name_map = modules_to_name # {module_obj: name_str} mapping


#         # parse config
#         self.cpu_offload = config.cpu_offload
#         self.dtype = config.get_dtype()

#         # log saver
#         self.log_saver = LogSaver(config=config, state=self.state)

#         # hooks
#         self.modules_to_hook = []
#         self.modules_to_name = {}
#         self.forward_hooks = []
#         self.backward_hooks = []
#         self.grad_hooks = []
#         self.tensor_hooks = []

#     def log(self, data_id: Any, mask: Optional[torch.Tensor] = None):
#         """
#         Add log state on exit.

#         Args:
#             data_id: The data ID associated with the current batch.
#             mask (torch.Tensor): A mask to be applied to the activations.
#         """
#         log = self.binfo.log.copy()
#         self.binfo.clear()
#         self.binfo.data_id = data_id
#         self.binfo.mask = mask

#         self.update()

#         return log

#     def save_log(self):
#         # save log to disk
#         self.log_saver.buffer_write(binfo=self.binfo)
#         self.log_saver.flush()

#     def update(self, save: bool = False):
#         """
#         Update statistics AFTER forward and backward pass complete.
#         MODIFIED: Uses self.modules_to_name to retrieve module object.
#         """
#         # Compute Per-Sample Gradients
#         for module_name in self.binfo.hook_inputs.keys():
#             if module_name in self.binfo.hook_output_grads:
#                 fwd_activation = self.binfo.hook_inputs[module_name]
#                 bwd_gradient = self.binfo.hook_output_grads[module_name]
#                 # *** Retrieve module object using the INSTANCE dictionary ***
#                 module = self.modules_to_name.get(module_name) # Use the name->obj map

#                 if module is not None:
#                     per_sample_gradient = compute_per_sample_gradient(
#                         fwd_activation, bwd_gradient, module
#                     )

#                     if self.dtype is not None:
#                         per_sample_gradient = per_sample_gradient.to(dtype=self.dtype)

#                     if len(self.opt.grad) > 0 and self.opt.grad[0] == Log:
#                         # Ensure module object is passed to Log.update if needed
#                         self.opt.grad[0].update(
#                             state=self.state,
#                             binfo=self.binfo,
#                             module=module, # Pass the retrieved module
#                             module_name=module_name,
#                             log_type="grad",
#                             data=per_sample_gradient,
#                             cpu_offload=self.cpu_offload,
#                         )
#                 else:
#                     get_logger().warning(f"Module object not found for name '{module_name}' during gradient calculation in update.")
#             # else: # Optional: Log warning if backward grad is missing
#                 # get_logger().warning(f"Missing backward gradient for {module_name} in update.")


#         # gradient plugin has to be excecuted after accumulating all gradients
#         for stat in self.opt.grad[1:]:
#             for module_name, _ in self.binfo.log.items():
#                 stat.update(
#                     state=self.state,
#                     binfo=self.binfo,
#                     module=None,
#                     module_name=module_name,
#                     log_type="grad",
#                     data=None,
#                     cpu_offload=self.cpu_offload,
#                 )

#         # Wait for all asynchronous CUDA operations to finish
#         if torch.cuda.is_available():
#             torch.cuda.current_stream().synchronize()

#         # Write and flush the buffer if necessary
#         if save:
#             self.save_log()





#     def _forward_hook_fn(
#         self, module: nn.Module, inputs: Tuple[torch.Tensor], module_name: str
#     ) -> None:
#         """
#         Internal forward hook function.

#         Args:
#             module: The module triggering the hook.
#             inputs: The input to the module.
#             module_name (str): The name of the module.
#         """
#         assert len(inputs) == 1

#         activations = inputs[0]

#         # If `mask` is not None, apply the mask to activations. This is
#         # useful for example when you work with sequence models that use
#         # padding. In this case, you can use the mask to ignore the padding
#         if self.binfo.mask is not None:
#             mask = self.binfo.mask
#             if len(mask.shape) != len(activations.shape):
#                 assert len(mask.shape) == len(activations.shape) - 1
#                 if mask.shape[-1] == activations.shape[-2]:
#                     activations = activations * mask.unsqueeze(-1)
#             else:
#                 if mask.shape[-1] == activations.shape[-1]:
#                     activations = activations * mask

#         if self.dtype is not None:
#             activations = activations.to(dtype=self.dtype)

#         for plugin in self.opt.forward:
#             plugin.update(
#                 state=self.state,
#                 binfo=self.binfo,
#                 module=module,
#                 module_name=module_name,
#                 log_type="forward",
#                 data=activations,
#                 cpu_offload=self.cpu_offload,
#             )

#     def _backward_hook_fn(
#         self,
#         module: nn.Module,
#         grad_inputs: Tuple[torch.Tensor],
#         grad_outputs: Tuple[torch.Tensor],
#         module_name: str,
#     ) -> None:
#         """
#         Internal backward hook function.

#         Args:
#             module: The module triggering the hook.
#             grad_inputs: The gradient of the input to the module.
#             grad_outputs: The gradient of the output from the module.
#             module_name (str): The name of the module.
#         """
#         assert len(grad_outputs) == 1
#         error = grad_outputs[0]

#          # === Store the output gradient if it exists ===
#         if error is not None:
#             # Use detached clone to prevent holding graph history
#             self.binfo.hook_output_grads[module_name] = error.detach().clone()
#         # === End Store ===



#         if self.dtype is not None:
#             error = error.to(dtype=self.dtype)

#         for plugin in self.opt.backward:
#             plugin.update(
#                 state=self.state,
#                 binfo=self.binfo,
#                 module=module,
#                 module_name=module_name,
#                 log_type="backward",
#                 data=error,
#                 cpu_offload=self.cpu_offload,
#             )

#     def _grad_hook_fn(
#         self,
#         module: nn.Module,
#         inputs: Tuple[torch.Tensor],
#         outputs: Tuple[torch.Tensor],
#         module_name: str,
#     ) -> None:
#         """
#         Internal gradient hook function.

#         Args:
#             module: The module triggering the hook.
#             inputs: The input to the module.
#             outputs: The output from the module.
#             module_name (str): The name of the module.
#         """
#         assert len(inputs) == 1

#         # === Store the input activation ===
#         # Store detached clone. This assumes the *first* input is the activation.
#         self.binfo.hook_inputs[module_name] = inputs[0].detach().clone()
#         # === End Store ===

#         if not outputs.requires_grad:                 # outputs is the only element in the tuple
#             get_logger().debug(
#                 f"Skip gradhook for {module_name} (no requires_grad)"
#             )
#             return


#         # In case, the same module is used multiple times in the forward pass,
#         # we need to accumulate the gradients. We achieve this by using the
#         # additional tensor hook on the output of the module.
#         def _grad_backward_hook_fn(grad: torch.Tensor):
#             if len(self.opt.grad) > 0:
#                 assert self.opt.grad[0] == Log
#                 per_sample_gradient = compute_per_sample_gradient(
#                     inputs[0], grad, module
#                 )

#                 if self.dtype is not None:
#                     per_sample_gradient = per_sample_gradient.to(dtype=self.dtype)

#                 for plugin in self.opt.grad[:1]:
#                     plugin.update(
#                         state=self.state,
#                         binfo=self.binfo,
#                         module=module,
#                         module_name=module_name,
#                         log_type="grad",
#                         data=per_sample_gradient,
#                         cpu_offload=self.cpu_offload,
#                     )

#         tensor_hook = outputs.register_hook(_grad_backward_hook_fn)
#         self.tensor_hooks.append(tensor_hook)

#     def _tensor_forward_hook_fn(self, tensor: torch.Tensor, tensor_name: str) -> None:
#         """
#         Internal forward hook function specifically designed for tensors.

#         This method allows you to track the activations of tensors that are not
#         necessarily tied to any specific module, but are still of interest.

#         Args:
#             tensor: The tensor triggering the hook.
#             tensor_name (str): A string identifier for the tensor, useful for logging.
#         """
#         if self.dtype is not None:
#             tensor = tensor.to(dtype=self.dtype)

#         for plugin in self.opt.forward:
#             plugin.update(
#                 state=self.state,
#                 binfo=self.binfo,
#                 module=None,
#                 module_name=tensor_name,
#                 log_type="forward",
#                 data=tensor,
#                 cpu_offload=self.cpu_offload,
#             )

#     def _tensor_backward_hook_fn(self, grad: torch.Tensor, tensor_name: str) -> None:
#         """
#         Internal backward hook function specifically designed for tensors.

#         This method allows you to track the gradients associated with specific
#         tensors that are not necessarily tied to any specific module, but are still of interest.

#         Args:
#             grad: The gradient tensor triggering the hook.
#             tensor_name (str): A string identifier for the tensor whose gradient is being tracked.
#         """
#         if self.dtype is not None:
#             grad = grad.to(dtype=self.dtype)

#         for plugin in self.opt.backward:
#             plugin.update(
#                 state=self.state,
#                 binfo=self.binfo,
#                 module=None,
#                 module_name=tensor_name,
#                 log_type="backward",
#                 data=grad,
#                 cpu_offload=self.cpu_offload,
#             )

#     def register_all_module_hooks(self) -> None:
#         """
#         Register all module hooks.
#         """
#         for module in self.modules_to_hook:
#             # As each hook has its own function, we need to register all hooks
#             # separately. We use partial functions to pass the module name to
#             # the hook functions.
#             module_name = self.modules_to_name[module]
#             forward_hook = module.register_forward_pre_hook(
#                 partial(self._forward_hook_fn, module_name=module_name)
#             )
#             backward_hook = module.register_full_backward_hook(
#                 partial(self._backward_hook_fn, module_name=module_name)
#             )
#             grad_hook = module.register_forward_hook(
#                 partial(self._grad_hook_fn, module_name=module_name)
#             )
#             self.forward_hooks.append(forward_hook)
#             self.backward_hooks.append(backward_hook)
#             self.grad_hooks.append(grad_hook)

#     def register_all_tensor_hooks(self, tensor_dict: Dict[str, torch.Tensor]) -> None:
#         """
#         Register all tensor hooks.

#         Args:
#             tensor_dict (dict): Dictionary containing tensor names as keys and tensors as values.
#         """
#         for tensor_name, tensor in tensor_dict.items():
#             self._tensor_forward_hook_fn(tensor, tensor_name)
#             tensor_hook = tensor.register_hook(
#                 partial(self._tensor_backward_hook_fn, tensor_name=tensor_name)
#             )
#             self.tensor_hooks.append(tensor_hook)

#     def add_module(self, module_name: str, module: nn.Module) -> None:
#         """
#         Add a module to be hooked.

#         Args:
#             module_name (str): The name of the module.
#             module: The module to be hooked.
#         """
#         self.modules_to_hook.append(module)
#         self.modules_to_name[module] = module_name

#         # # Also populate the reverse map needed in __init__
#         # if not hasattr(self, 'modules_to_name_map'): self.modules_to_name_map = {}
#         # self.modules_to_name_map[module] = module_name # obj: name



#     def setup(self, log_option_kwargs: Dict[str, Any]) -> None:
#         """
#         Update logging configurations.

#         Args:
#             log_option_kwargs: Logging configurations.
#         """
#         self.opt.setup(log_option_kwargs)

#     def finalize(self):
#         """
#         Dump everything in the buffer to a disk.
#         """
#         self.log_saver.finalize()

#     def clear(
#         self, hook: bool = True, module: bool = True, buffer: bool = True
#     ) -> None:
#         """
#         Clear all hooks and internal states.

#         Args:
#             hook (bool): Whether to clear the hooks.
#             module (bool): Whether to clear the modules.
#         """
#         if hook:
#             for hook in self.forward_hooks:
#                 hook.remove()
#             self.forward_hooks.clear()
#             for hook in self.backward_hooks:
#                 hook.remove()
#             self.backward_hooks.clear()
#             for hook in self.grad_hooks:
#                 hook.remove()
#             self.grad_hooks.clear()
#             for hook in self.tensor_hooks:
#                 hook.remove()
#             self.tensor_hooks.clear()

#         if module:
#             self.modules_to_hook = []
#             self.modules_to_name = {}

#         if buffer:
#             self.log_saver.buffer_clear()
