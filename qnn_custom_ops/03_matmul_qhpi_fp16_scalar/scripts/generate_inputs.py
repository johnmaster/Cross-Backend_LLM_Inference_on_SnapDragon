from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input"
EXPECTED_DIR = ROOT / "test_data"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
EXPECTED_DIR.mkdir(parents=True, exist_ok=True)

lhs = np.array(
    [[[[1.0, 2.0, 3.0],
       [4.0, 5.0, 6.0]]]],
    dtype=np.float32,
)

rhs = np.array(
    [[[[7.0, 8.0],
       [9.0, 10.0],
       [11.0, 12.0]]]],
    dtype=np.float32,
)

# Match the kernel contract: FP16 inputs, FP32 accumulation, FP16 output.
lhs_fp16 = lhs.astype(np.float16)
rhs_fp16 = rhs.astype(np.float16)
expected = np.matmul(
    lhs_fp16.astype(np.float32),
    rhs_fp16.astype(np.float32),
).astype(np.float16)

lhs.tofile(INPUT_DIR / "lhs.raw")
rhs.tofile(INPUT_DIR / "rhs.raw")
expected.astype(np.float32).tofile(EXPECTED_DIR / "expected_float.raw")
expected.tofile(EXPECTED_DIR / "expected_fp16.raw")

print("lhs shape:", lhs.shape)
print("rhs shape:", rhs.shape)
print("expected:\n", expected)
print("wrote:", INPUT_DIR / "lhs.raw")
print("wrote:", INPUT_DIR / "rhs.raw")
print("wrote:", EXPECTED_DIR / "expected_float.raw")
print("wrote:", EXPECTED_DIR / "expected_fp16.raw")
