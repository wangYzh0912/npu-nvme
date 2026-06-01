"""Fast initialization helper for NVMe checkpoint metadata.

Usage:
- python python/fast_init.py

Inputs:
- NVMe PCI address and metadata settings in script.
Outputs:
- Initializes metadata slots on NVMe and logs status.
"""
import mindspore.common.initializer as init_mod
from mindspore.common.initializer import Initializer, _register, initializer

@_register('noop')
class NoOpInitializer(Initializer):
    """
    A 'Fake' Initializer that does nothing.
    Used to bypass time-consuming random initialization when we intend to 
    overwrite the parameters immediately from NVMe/Checkpoint.
    """
    def _initialize(self, arr):
        # Do absolutely nothing.
        # 'arr' will retain uninitialized memory state (garbage values).
        # This is fine because we will overwrite it with DirectCheckpoint.load().
        pass

def replace_with_noop_initializer(model):
    """
    Iterate over all parameters in the model and replace their init_mode
    with NoOpInitializer.
    """
    print("[FastInit] Replacing original initializers with NoOpInitializer...", flush=True)
    count = 0
    for param in model.get_parameters():
        if param.init_mode is not None:
            # [Fix] init_mode must be a Tensor (wrapped initializer), not the Initializer object itself.
            # We use the `initializer` factory function to create a MetaTensor that holds the config.
            # This ensures that copy/clone operations work correctly.
            param.init_mode = initializer(NoOpInitializer(), shape=param.shape, dtype=param.dtype)
            count += 1
    print(f"[FastInit] Replaced {count} parameter initializers.", flush=True)
