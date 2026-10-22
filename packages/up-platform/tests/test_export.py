"""Limited CPU export compatibility; not vendor/native-kernel validation."""
import io

import pytest
import torch

from up_platform import configure_platform


def test_plain_pytorch_onnx_export_keeps_parameters_and_names():
    onnx = pytest.importorskip("onnx")
    ctx = configure_platform({"backend": "cpu"}, purpose="inference", reset=True)
    model = torch.nn.Linear(3, 2).eval()
    keys = list(model.state_dict())
    parameters = [id(value) for value in model.parameters()]
    result = io.BytesIO()
    ctx.inference.export_onnx(model, torch.ones(1, 3), result,
                              input_names=["input"], output_names=["output"], opset_version=17)
    graph = onnx.load_model_from_string(result.getvalue()).graph
    assert [node.op_type for node in graph.node] == ["Gemm"]
    assert [value.name for value in graph.initializer] == keys
    assert [id(value) for value in model.parameters()] == parameters
    assert list(model.state_dict()) == keys
    configure_platform({"backend": "cpu"}, purpose="training", reset=True)
