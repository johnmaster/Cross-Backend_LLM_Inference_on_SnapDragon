from pathlib import Path

import numpy as np


root = Path(__file__).resolve().parent.parent
output_dir = root / "test_data"
output_dir.mkdir(parents=True, exist_ok=True)

# lhs: [B, H, M, K] = [1, 1, 2, 3]
lhs = np.array(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ],
    dtype=np.float32,
).reshape(1, 1, 2, 3)

# rhs: [B, H, K, N] = [1, 1, 3, 2]
rhs = np.array(
    [
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
    ],
    dtype=np.float32,
).reshape(1, 1, 3, 2)

expected = np.matmul(lhs, rhs)

lhs.tofile(output_dir / "lhs.raw")
rhs.tofile(output_dir / "rhs.raw")
expected.tofile(output_dir / "expected.raw")

print("lhs shape:", lhs.shape)
print(lhs)

print("rhs shape:", rhs.shape)
print(rhs)

print("expected shape:", expected.shape)
print(expected)