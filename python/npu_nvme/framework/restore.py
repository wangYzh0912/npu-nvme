"""Legacy framework restore target; strict unready transactions arrive in D1."""
import hashlib
import numpy as np
from training_state import decode_control_value


class LegacyStateTarget:
    def __init__(self, components, capture, framework, operations):
        self.components = components
        self.capture = capture
        self.framework = framework
        self.operations = operations

    def component_names(self):
        return self.capture.ordered_components(self.components)

    def prepare(self):
        return self.capture.prepare_state_components(self.components, with_checksums=False)

    def allocate(self, saved, targets):
        dev_buffers, host_buffers, control_buffers = [], [], {}
        for name, info in saved.items():
            if name.startswith("control/"):
                if info.get("dtype") != "uint8" or info.get("size", 0) <= 0:
                    raise ValueError(f"invalid control-state record: {name}")
                array = np.empty(int(info["size"]), dtype=np.uint8)
                item = {"name": name, "ptr": int(array.ctypes.data),
                        "size": int(info["size"]), "offset": int(info["offset"])}
                host_buffers.append(item)
                control_buffers[name] = (array, info)
                continue

            target = targets[name]
            saved_shape = list(info.get("shape", []))
            target_shape = list(target["shape"])
            # MindSpore may materialize scalar optimizer hyperparameters as
            # either [] or [1] in fresh processes.  They are byte-compatible
            # when dtype and payload size agree; keep strict shape checks for
            # all non-singleton tensors.
            singleton_shape = (
                int(info.get("size", -1)) == int(target["size"]) and
                np.prod(saved_shape or [1]) == 1 and
                np.prod(target_shape or [1]) == 1)
            if ((saved_shape != target_shape and not singleton_shape) or
                    np.dtype(info.get("dtype")) != np.dtype(target["dtype"]) or
                    int(info.get("size", -1)) != int(target["size"])):
                raise ValueError(f"shape/dtype/size mismatch for {name}")
            item = dict(target)
            item.update(offset=int(info["offset"]), size=int(info["size"]))
            if target["ptr"]:
                dev_buffers.append(item)
            else:
                array = np.empty(target["shape"], dtype=np.dtype(target["dtype"]))
                item["np_arr"] = array
                item["ptr"] = int(array.ctypes.data)
                host_buffers.append(item)

        return dev_buffers, host_buffers, control_buffers

    def apply(self, host_buffers):
        for item in host_buffers:
            if item["name"].startswith("control/"):
                continue
            parameter = item["param_ref"]
            self.operations.assign(parameter, self.framework.Tensor(item["np_arr"], dtype=parameter.dtype))
        if hasattr(self.framework.hal, "synchronize"):
            self.framework.hal.synchronize()


    def verify(self, targets, saved, record, verify_checksums):
        if verify_checksums and record.get("checksum") == "sha256":
            for name, target in targets.items():
                expected = saved[name].get("sha256")
                if not expected:
                    raise ValueError(f"missing checksum for {name}")
                actual_arr = np.ascontiguousarray(
                    target["param_ref"].value().asnumpy(),
                    dtype=np.dtype(target["dtype"])).reshape(-1)
                expected_bytes = int(saved[name].get("size", actual_arr.nbytes))
                if actual_arr.nbytes != expected_bytes:
                    raise ValueError(
                        f"parameter byte-size mismatch for {name}: "
                        f"actual={actual_arr.nbytes} expected={expected_bytes}")
                actual = hashlib.sha256(actual_arr.tobytes()).hexdigest()
                if actual != expected:
                    raise ValueError(f"parameter checksum mismatch for {name}")


    def decode(self, control_buffers):
        controls = {}
        for qualified_name, (array, info) in control_buffers.items():
            source_name = qualified_name.split("/", 1)[1]
            if not source_name or qualified_name != f"control/{source_name}":
                raise ValueError(f"invalid control-state namespace: {qualified_name}")
            controls[source_name] = decode_control_value(array, info)
        return controls
