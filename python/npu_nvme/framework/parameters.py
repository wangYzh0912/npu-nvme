"""MindSpore parameter addresses; private pointer access isolated here."""
def get_dev_ptr(tensor):
    """Return the NPU device pointer for a MindSpore Parameter or Tensor.

    Uses the private _data_ptr() API (available MS 2.4+).  Falls back to
    the undocumented data_ptr() on older builds.  Returns 0 on failure.

    NOTE: _data_ptr() is a PRIVATE MindSpore API and may break on any
    version upgrade.  ALL device-pointer access in this codebase should
    go through this function so there is a single point to fix.
    """
    import mindspore as ms
    ptr = 0
    try:
        data_obj = tensor.data if hasattr(tensor, "data") else tensor
        if hasattr(data_obj, "_data_ptr"):
            if isinstance(tensor, ms.Parameter) and hasattr(tensor, "is_inited") and not tensor.is_inited:
                tensor.init_data()
            ptr = int(data_obj._data_ptr())
        elif hasattr(data_obj, "data_ptr"):
            ptr = int(data_obj.data_ptr())
    except Exception:
        ptr = 0
    return ptr


