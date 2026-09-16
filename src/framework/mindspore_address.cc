// Read actual allocation metadata, not Tensor.device (a placement preference).
// Built against the selected MindSpore installation; never dereference a raw
// Python-provided address or depend on offsets into a framework object.
#include <pybind11/pybind11.h>
#include "include/common/utils/tensor_py.h"
#include "ir/device_address.h"

namespace py = pybind11;

PYBIND11_MODULE(_address_native, module) {
  module.attr("framework_version") = NPU_NVME_MINDSPORE_VERSION;
  module.def("allocation", [](const py::object &value) {
    if (!mindspore::tensor::IsPyObjectTensorPy(value.ptr()))
      throw py::type_error("expected a MindSpore Tensor");
    auto tensor = mindspore::tensor::ConvertPyObjectToTensor(value.ptr());
    if (!tensor) throw py::value_error("null MindSpore tensor");
    auto address = tensor->device_address();
    py::dict result;
    result["provenance"] = "mindspore.DeviceAddress";
    if (!address) {
      result["placement"] = "host";
      result["device_id"] = py::none();
      result["pointer"] = 0;
      result["allocation_bytes"] = 0;
      return result;
    }
    const auto type = address->GetDeviceType();
    result["placement"] = type == mindspore::device::DeviceType::kCPU ? "host" :
                           type == mindspore::device::DeviceType::kAscend ? "device" : "unknown";
    result["device_id"] = address->device_id();
    result["pointer"] = reinterpret_cast<uintptr_t>(address->GetPtr());
    result["allocation_bytes"] = address->GetSize();
    return result;
  });
}
