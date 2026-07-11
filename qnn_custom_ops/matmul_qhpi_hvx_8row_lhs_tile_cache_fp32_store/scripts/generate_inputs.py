from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input"
EXPECTED_DIR = ROOT / "test_data"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
EXPECTED_DIR.mkdir(parents=True, exist_ok=True)

BATCH = 1
HEADS = 1
ROWS = 128
REDUCTION = 256
COLUMNS = 256

rng = np.random.default_rng(20260625)
lhs = rng.uniform(
    low=-0.25,
    high=0.25,
    size=(BATCH, HEADS, ROWS, REDUCTION),
).astype(np.float32)
rhs = rng.uniform(
    low=-0.25,
    high=0.25,
    size=(BATCH, HEADS, REDUCTION, COLUMNS),
).astype(np.float32)

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
print("expected shape:", expected.shape)
print("lhs range:", float(lhs.min()), float(lhs.max()))
print("rhs range:", float(rhs.min()), float(rhs.max()))
print("expected range:", float(expected.min()), float(expected.max()))
print("wrote:", INPUT_DIR / "lhs.raw")
print("wrote:", INPUT_DIR / "rhs.raw")
print("wrote:", EXPECTED_DIR / "expected_float.raw")
print("wrote:", EXPECTED_DIR / "expected_fp16.raw")
