import importlib.metadata as metadata

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision

assert torch.cuda.is_available(), "PyTorch cannot access CUDA."
assert torch.version.cuda is not None
assert torch.version.cuda.startswith("13."), "Expected a CUDA 13 build."

x = torch.ones((64, 64), device="cuda")
result = x @ x
torch.cuda.synchronize()
assert result[0, 0].item() == 64

try:
    metadata.version("onnxruntime")
except metadata.PackageNotFoundError:
    pass
else:
    raise RuntimeError("CPU-only onnxruntime is also installed.")

ort.preload_dlls()
assert "CUDAExecutionProvider" in ort.get_available_providers()

inputs = [
    onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [64, 64])
    for name in ("a", "b")
]
output = onnx.helper.make_tensor_value_info("c", onnx.TensorProto.FLOAT, [64, 64])
graph = onnx.helper.make_graph(
    [onnx.helper.make_node("MatMul", ["a", "b"], ["c"])],
    "cuda_check",
    inputs,
    [output],
)
model = onnx.helper.make_model(
    graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
)
model.ir_version = 9

options = ort.SessionOptions()
options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
session = ort.InferenceSession(
    model.SerializeToString(),
    sess_options=options,
    providers=["CUDAExecutionProvider"],
)
assert session.get_providers()[0] == "CUDAExecutionProvider"

value = np.ones((64, 64), dtype=np.float32)
actual = session.run(None, {"a": value, "b": value})[0]
np.testing.assert_allclose(actual, np.full((64, 64), 64, dtype=np.float32))

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA runtime:", torch.version.cuda)
print("ONNX Runtime:", ort.__version__)
print("Both CUDA computation checks passed.")
