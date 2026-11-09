"""Fast initialisation — skip random init when loading weights from NVMe.

Provides a NoOpInitializer that leaves parameter memory unmodified so
that DirectCheckpoint.load() can overwrite it with checkpoint data.
"""

from mindspore.common.initializer import Initializer, _register, initializer


@_register('noop')
class NoOpInitializer(Initializer):
    """A 'Fake' Initializer that does nothing.

    Used to bypass time-consuming random initialisation when parameters
    will be immediately overwritten with data loaded from an NVMe
    checkpoint via DirectCheckpoint.load().

    Note: ``arr`` retains uninitialised memory (garbage values).  This
    is safe only when every parameter is overwritten by a subsequent
    load() call.
    """
    def _initialize(self, arr):
        pass


def replace_with_noop_initializer(model):
    """Replace all parameter initializers with NoOpInitializer.

    Iterates over model.parameters_and_names() and sets each parameter's
    init_mode to a NoOpInitializer MetaTensor.
    """
    print("[FastInit] Replacing original initializers with NoOpInitializer...",
          flush=True)
    count = 0
    for param in model.get_parameters():
        if param.init_mode is not None:
            param.init_mode = initializer(
                NoOpInitializer(), shape=param.shape, dtype=param.dtype)
            count += 1
    print(f"[FastInit] Replaced {count} parameter initializers.", flush=True)
