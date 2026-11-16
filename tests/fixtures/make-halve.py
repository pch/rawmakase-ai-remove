"""Builds tests/fixtures/halve.onnx: LaMa's interface (image, mask in 512x512, output in 0..255) with a trivial body.
output = image * 255 * (1 - mask / 2): the hole comes back as half of each input channel, everything else as the input.
Per-channel halving shows the planes' layout, order and 0..1 scaling; the mask's orientation shows in where it halves."""
import sys

import numpy as np
import onnx
from onnx import TensorProto as T
from onnx import helper as h, numpy_helper as nh

c = lambda name, v: nh.from_array(np.array(v, np.float32), name)
graph = h.make_graph(
    [
        h.make_node("Mul", ["image", "k255"], ["scaled"]),
        h.make_node("Mul", ["mask", "half"], ["cut"]),
        h.make_node("Sub", ["one", "cut"], ["keep"]),
        h.make_node("Mul", ["scaled", "keep"], ["output"]),
    ],
    "halve",
    [h.make_tensor_value_info("image", T.FLOAT, ["batch", 3, 512, 512]), h.make_tensor_value_info("mask", T.FLOAT, ["batch", 1, 512, 512])],
    [h.make_tensor_value_info("output", T.FLOAT, ["batch", 3, 512, 512])],
    [c("k255", 255.0), c("half", 0.5), c("one", 1.0)],
)
model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, sys.argv[1])
