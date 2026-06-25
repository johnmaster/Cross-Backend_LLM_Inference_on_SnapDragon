# QHPI HVX MatMul

This directory implements an FP16 batched MatMul custom op with the QHPI
External Op Package API.

## Current Implementation

- QHPI Op: `MatMulQhpiHvx`
- Op Package: `MatMulQhpiHvxOpPackage`
- Internal inputs and output: `QHPI_Float16`
- Tensor layout: `QHPI_Layout_Flat4`
- Runtime resource: `QHPI_RESOURCE_HVX`
- Shape contract: `[B,H,M,K] x [B,H,K,N] -> [B,H,M,N]`
- Graph boundary: FP32 app tensors, QNN `Cast` to/from FP16 around the custom op

The Hexagon/libnative path computes full 64-column output tiles with HVX
integer multiply-accumulate:

1. convert FP16 inputs to Q13 fixed-point values;
2. broadcast one Q13 lhs scalar with `Q6_Vh_vsplat_R`;
3. load 64 contiguous Q13 rhs columns with `vmemu`;
4. accumulate int16 x int16 into int32 lanes with
   `Q6_Ww_vmpyacc_WwVhVh`;
5. dequantize the int32 accumulators and write FP16 output.

Columns that do not fill a 64-wide tile use the scalar FP16-input,
FP32-accumulation fallback. This keeps arbitrary `N` correct.

The first FP16-floating HVX prototype using `Q6_Wsf_vmpyacc_WsfVhfVhf`
compiled and dispatched on this device, but returned zeros at runtime. The
checked-in implementation therefore uses the integer HVX path above as the
working correctness milestone.

## Demo Shape

The checked-in sample model uses:

```text
lhs    [1,1,128,256]
rhs    [1,1,256,256]
output [1,1,128,256]
```

That shape intentionally has `N=256`, so the sample exercises four full
64-column HVX tiles per output row. The larger `M` and `K` dimensions make it
useful for comparing future matrix multiplication optimizations.

## Files

```text
matmul_qhpi_hvx/
├── config/MatMulQhpiHvxOpPackage.xml
├── htp/MatMulQhpiHvxOpPackage/
│   ├── Makefile
│   ├── config/MatMulQhpiHvxOpPackage.xml
│   └── src/
│       ├── MatMulQhpiHvxOpPackageInterface.cpp
│       └── ops/MatMulQhpiHvx.cpp
├── model/custom_matmul_qhpi_hvx_model.cpp
├── input/
├── scripts/generate_inputs.py
└── test_data/
```

## Build

Generate or refresh the package skeleton if needed:

```bash
qnn-op-package-generator \
  --config_path "$REPO/qnn_custom_ops/matmul_qhpi_hvx/config/MatMulQhpiHvxOpPackage.xml" \
  --output_path "$REPO/qnn_custom_ops/matmul_qhpi_hvx/htp" \
  --debug
```

Build the HTP package, for example:

```bash
cd "$REPO/qnn_custom_ops/matmul_qhpi_hvx/htp/MatMulQhpiHvxOpPackage"
make htp_v75
make htp_aarch64
```

The expected package library names are:

```text
libQnnMatMulQhpiHvxOpPackage.so
libcustom_matmul_qhpi_hvx_model.so
```

## Verification

Regenerate sample inputs and expected outputs:

```bash
python3 qnn_custom_ops/matmul_qhpi_hvx/scripts/generate_inputs.py
```

After building the Hexagon shared object, inspect it for HVX integer multiply
instructions:

```bash
llvm-objdump -d build/hexagon-v75/libQnnMatMulQhpiHvxOpPackage.so | \
  grep -E "vmpy|vmpyacc|vsplat|vmem"
```

Compare device output against `test_data/expected_float.raw` at the FP16 result
tolerance. The current Q13 HVX path has been measured on the sample input at:

```text
max_abs_error  = 0.0002441704
mean_abs_error = 0.0000352946
allclose(atol=1e-3, rtol=1e-3) = true
```

## HVX Intrinsics Notes

`matmulqhpihvxCompute64ColumnsHvx()` is a local helper in this package, not a
Qualcomm API. The actual HVX intrinsics used inside it come from the Hexagon
SDK and QNN HTP headers.

Important source files:

```text
Hexagon SDK intrinsic declarations:
/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0/tools/HEXAGON_Tools/8.7.06/Tools/target/hexagon/include/hvx_hexagon_protos.h

QNN HTP convenience wrappers:
/home/lingbok/Qualcomm/qairt/2.47.0.260601/include/QNN/HTP/core/intrinsics.h
```

Useful search commands:

```bash
rg "Q6_.*vmpy|Q6_.*vadd|Q6_.*vsplat|Q6_.*vcvt|Q6_.*vmem" \
  /local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0/tools/HEXAGON_Tools/8.7.06/Tools/target/hexagon/include/hvx_hexagon_protos.h

rg "vmemu|q6op_|HVX_Vector|HVX_VectorPair" \
  /home/lingbok/Qualcomm/qairt/2.47.0.260601/include/QNN/HTP/core/intrinsics.h
```

The current kernel uses these HVX pieces:

```cpp
HVX_Vector
HVX_VectorPair
Q6_Vh_vsplat_R(...)
Q6_Ww_vmpyacc_WwVhVh(...)
Q6_V_lo_W(...)
Q6_V_hi_W(...)
vmemu(...)
```

`HVX_Vector` is one 128-byte HVX vector when compiled with
`-mhvx-length=128B`. For 16-bit lanes, that means 64 lanes per vector. For
32-bit lanes, that means 32 lanes per vector.

`HVX_VectorPair` is two HVX vectors. Many widening operations use a pair
because multiplying 64 int16 lanes produces 64 int32 results, which need
`64 * 4 = 256` bytes.

Name reading example:

```text
Q6_Ww_vmpyacc_WwVhVh
   Q6       Hexagon intrinsic namespace
   Ww       result is a vector pair of 32-bit words
   vmpyacc  vector multiply accumulate
   Ww       accumulator input is a vector pair of 32-bit words
   Vh       first multiplicand is a vector of 16-bit halfwords
   Vh       second multiplicand is a vector of 16-bit halfwords
```

So this intrinsic means roughly:

```cpp
int32_vector_pair += int16_vector * int16_vector;
```

In this package, the FP16 inputs are converted to Q13 fixed-point values before
the HVX multiply:

```cpp
const int16_t lhs_q13 = matmulqhpihvxFloatToQ13(...);
const HVX_Vector lhs_vec = Q6_Vh_vsplat_R(lhs_q13);
acc = Q6_Ww_vmpyacc_WwVhVh(acc, lhs_vec, vmemu(rhs_q13));
```

`Q6_Vh_vsplat_R()` broadcasts one scalar 16-bit value into all 64 halfword
lanes of an `HVX_Vector`.

`vmemu(ptr)` is a QNN HTP wrapper for unaligned HVX vector load/store. It is
used here for stack buffers and output buffers because it avoids requiring the
pointer to be 128-byte aligned:

```cpp
HVX_Vector rhs_vec = vmemu(rhs_q13);
vmemu(acc_lo_store) = Q6_V_lo_W(acc);
```

`Q6_V_lo_W()` and `Q6_V_hi_W()` split an `HVX_VectorPair` into its low and high
vectors. For `Q6_Ww_vmpyacc_WwVhVh`, the low vector contains even output lanes
and the high vector contains odd output lanes, so the writeback interleaves
them:

```cpp
output[col + 2 * lane]     = dequantized_low_lane;
output[col + 2 * lane + 1] = dequantized_high_lane;
```

### How We Know This Is HVX, Not HMX

This package uses HVX because:

- the code uses `HVX_Vector` and `HVX_VectorPair`;
- the intrinsics are declared in `hvx_hexagon_protos.h`;
- the generated code disassembles to vector instructions such as:

```text
v5:4.w += vmpy(v0.h,v1.h)
v0.h = vsplat(r2)
vmem(...)
```

Those are ordinary HVX vector-lane instructions. HMX would require HMX-specific
headers, intrinsics, matrix/tile layouts, and accumulator/control paths. This
package does not use any HMX intrinsic or HMX tile data path.
