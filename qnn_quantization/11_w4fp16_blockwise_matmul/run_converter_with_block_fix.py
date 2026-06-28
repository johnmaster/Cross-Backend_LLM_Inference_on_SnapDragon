#!/usr/bin/env python3

import os
import runpy
import sys

import numpy as np

from qti.aisw.converters.common.converter_ir.op_graph_optimizations import (
    IROptimizations,
)


def contains_decimal(value):
    values = np.asarray(value)
    return bool(np.any(~np.isclose(values, np.rint(values))))


# QAIRT 2.47 calls round() directly on block zero-point arrays.
IROptimizations.contain_decimal_num = staticmethod(contains_decimal)

sdk_root = os.environ["QAIRT_SDK_ROOT"]
converter_name = "qnn-onnx-converter"
if len(sys.argv) > 1 and sys.argv[1] == "--qairt":
    del sys.argv[1]
    converter_name = "qairt-converter"
converter = os.path.join(
    sdk_root, "bin", "x86_64-linux-clang", converter_name
)
runpy.run_path(converter, run_name="__main__")
