#!/usr/bin/env python3
"""Generate small ONNX test fixture models for weaver-ir integration tests.

Requirements:
    pip install onnx numpy

The script produces three models in crates/weaver-ir/tests/fixtures/:
  1. simple_mlp.onnx        — A 2-layer MLP (FC → ReLU → FC → Softmax)
  2. tiny_convnet.onnx       — Conv → BatchNorm → ReLU → MaxPool → GlobalAvgPool → FC
  3. mini_resnet_block.onnx  — Conv → BN → ReLU → Conv → BN → Add (residual) → ReLU

All models use a symbolic batch dimension ("N") so the parser's symbolic-dim
propagation is exercised.
"""

import os
import urllib.request
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
os.makedirs(FIXTURES_DIR, exist_ok=True)


def save(model: onnx.ModelProto, name: str) -> None:
    path = os.path.join(FIXTURES_DIR, name)
    onnx.save(model, path)
    print(f"  ✓ {path}  ({os.path.getsize(path)} bytes)")


# ─── 1. Simple MLP ──────────────────────────────────────────────────

def make_simple_mlp() -> onnx.ModelProto:
    """(N, 784) → FC(128) → ReLU → FC(10) → Softmax → (N, 10)"""

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 784])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N", 10])

    # Weights & biases as initializers (random, small)
    rng = np.random.default_rng(42)
    W1 = numpy_helper.from_array(rng.standard_normal((128, 784)).astype(np.float32), "W1")
    B1 = numpy_helper.from_array(np.zeros(128, dtype=np.float32), "B1")
    W2 = numpy_helper.from_array(rng.standard_normal((10, 128)).astype(np.float32), "W2")
    B2 = numpy_helper.from_array(np.zeros(10, dtype=np.float32), "B2")

    fc1 = helper.make_node("Gemm", ["X", "W1", "B1"], ["fc1_out"], name="fc1",
                           transB=1)
    relu1 = helper.make_node("Relu", ["fc1_out"], ["relu1_out"], name="relu1")
    fc2 = helper.make_node("Gemm", ["relu1_out", "W2", "B2"], ["fc2_out"],
                           name="fc2", transB=1)
    softmax = helper.make_node("Softmax", ["fc2_out"], ["Y"], name="softmax",
                               axis=1)

    graph = helper.make_graph(
        [fc1, relu1, fc2, softmax],
        "simple_mlp",
        [X],
        [Y],
        initializer=[W1, B1, W2, B2],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model


# ─── 2. Tiny ConvNet ────────────────────────────────────────────────

def make_tiny_convnet() -> onnx.ModelProto:
    """
    (N, 1, 28, 28) → Conv(8, 3×3, same) → BN → ReLU
                    → MaxPool(2×2) → GlobalAvgPool → Flatten → FC(10)
    """

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 1, 28, 28])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N", 10])

    rng = np.random.default_rng(123)

    # Conv weights: (out_ch, in_ch, kH, kW)
    conv_w = numpy_helper.from_array(
        rng.standard_normal((8, 1, 3, 3)).astype(np.float32), "conv_w"
    )
    conv_b = numpy_helper.from_array(np.zeros(8, dtype=np.float32), "conv_b")

    # BatchNorm params
    bn_scale = numpy_helper.from_array(np.ones(8, dtype=np.float32), "bn_scale")
    bn_bias = numpy_helper.from_array(np.zeros(8, dtype=np.float32), "bn_bias")
    bn_mean = numpy_helper.from_array(np.zeros(8, dtype=np.float32), "bn_mean")
    bn_var = numpy_helper.from_array(np.ones(8, dtype=np.float32), "bn_var")

    # FC weights after global avg pool: input is (N, 8), output is (N, 10)
    fc_w = numpy_helper.from_array(
        rng.standard_normal((10, 8)).astype(np.float32), "fc_w"
    )
    fc_b = numpy_helper.from_array(np.zeros(10, dtype=np.float32), "fc_b")

    # Reshape target for flatten: [0, -1]  (keep batch, flatten rest)
    reshape_shape = numpy_helper.from_array(
        np.array([0, -1], dtype=np.int64), "reshape_shape"
    )

    conv = helper.make_node(
        "Conv", ["X", "conv_w", "conv_b"], ["conv_out"], name="conv",
        kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1],
    )
    bn = helper.make_node(
        "BatchNormalization",
        ["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        ["bn_out"],
        name="bn",
    )
    relu = helper.make_node("Relu", ["bn_out"], ["relu_out"], name="relu")
    pool = helper.make_node(
        "MaxPool", ["relu_out"], ["pool_out"], name="maxpool",
        kernel_shape=[2, 2], strides=[2, 2],
    )
    gap = helper.make_node(
        "GlobalAveragePool", ["pool_out"], ["gap_out"], name="gap"
    )
    reshape = helper.make_node(
        "Reshape", ["gap_out", "reshape_shape"], ["flat_out"], name="flatten"
    )
    fc = helper.make_node(
        "Gemm", ["flat_out", "fc_w", "fc_b"], ["Y"], name="fc", transB=1
    )

    graph = helper.make_graph(
        [conv, bn, relu, pool, gap, reshape, fc],
        "tiny_convnet",
        [X],
        [Y],
        initializer=[conv_w, conv_b, bn_scale, bn_bias, bn_mean, bn_var,
                      fc_w, fc_b, reshape_shape],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model


# ─── 3. Mini ResNet Block ───────────────────────────────────────────

def make_mini_resnet_block() -> onnx.ModelProto:
    """
    (N, 16, 8, 8) → Conv(16, 3×3, same) → BN → ReLU
                   → Conv(16, 3×3, same) → BN → Add(residual) → ReLU
                   → GlobalAvgPool → Flatten → FC(10)
    """

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 16, 8, 8])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N", 10])

    rng = np.random.default_rng(7)

    def conv_init(name, out_ch, in_ch):
        w = numpy_helper.from_array(
            rng.standard_normal((out_ch, in_ch, 3, 3)).astype(np.float32),
            f"{name}_w",
        )
        b = numpy_helper.from_array(np.zeros(out_ch, dtype=np.float32), f"{name}_b")
        return w, b

    def bn_init(name, ch):
        return [
            numpy_helper.from_array(np.ones(ch, dtype=np.float32), f"{name}_scale"),
            numpy_helper.from_array(np.zeros(ch, dtype=np.float32), f"{name}_bias"),
            numpy_helper.from_array(np.zeros(ch, dtype=np.float32), f"{name}_mean"),
            numpy_helper.from_array(np.ones(ch, dtype=np.float32), f"{name}_var"),
        ]

    c1w, c1b = conv_init("conv1", 16, 16)
    bn1_params = bn_init("bn1", 16)
    c2w, c2b = conv_init("conv2", 16, 16)
    bn2_params = bn_init("bn2", 16)

    fc_w = numpy_helper.from_array(
        rng.standard_normal((10, 16)).astype(np.float32), "fc_w"
    )
    fc_b = numpy_helper.from_array(np.zeros(10, dtype=np.float32), "fc_b")
    reshape_shape = numpy_helper.from_array(
        np.array([0, -1], dtype=np.int64), "reshape_shape"
    )

    nodes = [
        helper.make_node("Conv", ["X", "conv1_w", "conv1_b"], ["c1"],
                         name="conv1", kernel_shape=[3,3], pads=[1,1,1,1]),
        helper.make_node("BatchNormalization",
                         ["c1", "bn1_scale", "bn1_bias", "bn1_mean", "bn1_var"],
                         ["bn1"], name="bn1"),
        helper.make_node("Relu", ["bn1"], ["r1"], name="relu1"),
        helper.make_node("Conv", ["r1", "conv2_w", "conv2_b"], ["c2"],
                         name="conv2", kernel_shape=[3,3], pads=[1,1,1,1]),
        helper.make_node("BatchNormalization",
                         ["c2", "bn2_scale", "bn2_bias", "bn2_mean", "bn2_var"],
                         ["bn2"], name="bn2"),
        helper.make_node("Add", ["bn2", "X"], ["res"], name="residual_add"),
        helper.make_node("Relu", ["res"], ["r2"], name="relu2"),
        helper.make_node("GlobalAveragePool", ["r2"], ["gap"], name="gap"),
        helper.make_node("Reshape", ["gap", "reshape_shape"], ["flat"],
                         name="flatten"),
        helper.make_node("Gemm", ["flat", "fc_w", "fc_b"], ["Y"],
                         name="fc", transB=1),
    ]

    inits = [c1w, c1b, *bn1_params, c2w, c2b, *bn2_params, fc_w, fc_b,
             reshape_shape]

    graph = helper.make_graph(nodes, "mini_resnet_block", [X], [Y],
                              initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model

RESNET50_2_7: str = 'https://huggingface.co/webml/models-moved/resolve/99601f3929bcd18f3de820271171a449f75ef8a8/resnet50-v2-7.onnx'

# Download external files
def download_test_file(
    url: str
) -> Path:
    filename = url.split("/")[-1]
    destination = Path(FIXTURES_DIR) / filename

    if not destination.exists():
        urllib.request.urlretrieve(url, destination)

    return destination


# ─── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating ONNX test fixtures:")
    save(make_simple_mlp(), "simple_mlp.onnx")
    save(make_tiny_convnet(), "tiny_convnet.onnx")
    save(make_mini_resnet_block(), "mini_resnet_block.onnx")

    print("Downloading models")
    download_test_file(RESNET50_2_7)
    print("Done.")
