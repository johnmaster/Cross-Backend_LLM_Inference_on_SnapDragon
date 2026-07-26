//==============================================================================
// Auto Generated Code for MatMulQhpiHvx8RowFp32StoreMultithread - QHPI Implementation
// Multiple kernels generated for different data type combinations
//==============================================================================

#include "HTP/core/constraints.h"
#include <cmath>
#include <cstdint>
#include <string>

// Plugin/QHPI includes - using correct header from hexnn_qhpi.h
#include "HTP/core/qhpi.h"

#if defined(__hexagon__)
#include "HTP/core/intrinsics.h"
#define MATMUL_QHPI_HVX_8ROW_FP32_STORE_MULTITHREAD_INTRINSICS 1
#endif

// Forward declarations for MatMulQhpiHvx8RowFp32StoreMultithread kernel matmulqhpihvx8rowfp32storemultithread_float_16_
static uint32_t matmulqhpihvx8rowfp32storemultithread_float_16_Execute(QHPI_RuntimeHandle *handle,
                                      uint32_t num_outputs, QHPI_Tensor **outputs,
                                      uint32_t num_inputs, const QHPI_Tensor *const *inputs);
static uint32_t matmulqhpihvxstaticrhsprecomputed_float_16_Precompute(
    QHPI_RuntimeHandle *handle, void *data, uint32_t num_outputs,
    QHPI_Tensor **outputs, uint32_t num_inputs,
    const QHPI_Tensor *const *inputs);
static uint32_t matmulqhpihvxstaticrhsprecomputed_float_16_Execute(
    QHPI_RuntimeHandle *handle, const void *precomputed_data);
static uint32_t matmulqhpihvxofflineq13rhs_float_16_Execute(
    QHPI_RuntimeHandle *handle, uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs, const QHPI_Tensor *const *inputs);
static uint32_t convertfp16todeviceq13_Execute(
    QHPI_RuntimeHandle *handle, uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs, const QHPI_Tensor *const *inputs);
static uint32_t qwen2decodeattentionpast128fp32_Execute(
    QHPI_RuntimeHandle *handle, uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs, const QHPI_Tensor *const *inputs);
static float matmulqhpihvx8rowfp32storemultithread_float_16_CostFunc(const uint32_t num_inputs, const QHPI_Tensor *const *inputs);

// Common forward declarations for MatMulQhpiHvx8RowFp32StoreMultithread
static const QHPI_Op* matmulqhpihvx8rowfp32storemultithreadEarlyRewrite(const QHPI_Op *op);
static QHPI_Shape matmulqhpihvx8rowfp32storemultithreadShapeRequired(const QHPI_Op *op);
static QHPI_Shape matmulqhpihvx8rowfp32storemultithreadShapeLegal(const QHPI_Op *op, const QHPI_Shape* shape);
static const QHPI_Op* matmulqhpihvx8rowfp32storemultithreadBuildTile(const QHPI_Op *op, const QHPI_Shape* start, const QHPI_Shape* extent);
static const QHPI_Op* matmulqhpihvx8rowfp32storemultithreadLateRewrite(const QHPI_Op *op);

static inline float matmulqhpihvx8rowfp32storemultithreadScalarDot(const __fp16 *lhs,
                                           const __fp16 *rhs,
                                           uint64_t lhs_base,
                                           uint64_t rhs_base,
                                           uint32_t k,
                                           uint32_t n,
                                           uint32_t col) {
  float sum = 0.0f;
  for (uint32_t reduction = 0; reduction < k; ++reduction) {
    sum += static_cast<float>(lhs[lhs_base + reduction]) *
           static_cast<float>(rhs[rhs_base + (uint64_t)reduction * n + col]);
  }
  return sum;
}

static inline float matmulqhpihvx8rowfp32storemultithreadScalarDotQ13(
    const int16_t *lhs,
    const __fp16 *rhs,
    uint64_t lhs_base,
    uint64_t rhs_base,
    uint32_t k,
    uint32_t n,
    uint32_t col) {
  float sum = 0.0f;
  for (uint32_t reduction = 0; reduction < k; ++reduction) {
    sum += (static_cast<float>(lhs[lhs_base + reduction]) / 8192.0f) *
           static_cast<float>(
               rhs[rhs_base + (uint64_t)reduction * n + col]);
  }
  return sum;
}

static inline int16_t matmulqhpihvx8rowfp32storemultithreadFloatToQ13(float value) {
  constexpr float scale = 8192.0f;
  float scaled = value * scale;
  // static_cast<int32_t> truncates toward zero, so bias by half an LSB first
  // to round to nearest for both signs instead of making negative values too small in magnitude.
  scaled += scaled >= 0.0f ? 0.5f : -0.5f;

  int32_t q = static_cast<int32_t>(scaled);
  if (q > 32767) {
    q = 32767;
  } else if (q < -32768) {
    q = -32768;
  }
  return static_cast<int16_t>(q);
}

namespace {
constexpr uint32_t kStaticRhsMagic = 0x51524853U;
constexpr uint32_t kStaticRhsPackedColumns = 512U;
constexpr uint32_t kStaticRhsMaxElements =
    896U * kStaticRhsPackedColumns;
constexpr uint32_t kStaticRhsAlignment = 128U;

struct StaticRhsPrecomputedHeader {
  uint32_t magic;
  uint32_t batch;
  uint32_t heads;
  uint32_t m;
  uint32_t k;
  uint32_t n;
  uintptr_t lhs;
  uintptr_t rhs_fp16;
  uintptr_t output;
};

constexpr uint32_t kStaticRhsPrecomputedBytes =
    sizeof(StaticRhsPrecomputedHeader) + kStaticRhsAlignment - 1U +
    kStaticRhsMaxElements * sizeof(int16_t);

static inline int16_t *staticRhsPackedData(void *data) {
  const uintptr_t first =
      reinterpret_cast<uintptr_t>(data) + sizeof(StaticRhsPrecomputedHeader);
  return reinterpret_cast<int16_t *>(
      (first + kStaticRhsAlignment - 1U) &
      ~(static_cast<uintptr_t>(kStaticRhsAlignment) - 1U));
}

static inline const int16_t *staticRhsPackedData(const void *data) {
  return staticRhsPackedData(const_cast<void *>(data));
}
}  // namespace

#if MATMUL_QHPI_HVX_8ROW_FP32_STORE_MULTITHREAD_INTRINSICS
static inline HVX_Vector
matmulqhpihvx8rowfp32storemultithreadConvert64Fp16ToQ13(
    const __fp16 *values) {
  const HVX_Vector fp16_values = vmemu(values);

  // This uses QAIRT's qf32 intermediate path and avoids the direct half-float
  // vector multiply, which returns zero on the target device.
  return hnnx::s16_from_hf_rnd_sat<13>(fp16_values);
}

static inline void
matmulqhpihvx8rowfp32storemultithreadPackLhsRowsToQ13(
    const __fp16 *lhs,
    int16_t *lhs_q13,
    uint64_t lhs_base,
    uint32_t rows,
    uint32_t k) {
  for (uint32_t row = 0; row < rows; ++row) {
    const uint64_t src_base = lhs_base + (uint64_t)row * k;
    const uint64_t dst_base = (uint64_t)row * k;
    uint32_t reduction = 0;
    for (; reduction + 64 <= k; reduction += 64) {
      const HVX_Vector q13 =
          matmulqhpihvx8rowfp32storemultithreadConvert64Fp16ToQ13(
              &lhs[src_base + reduction]);
      vmemu(&lhs_q13[dst_base + reduction]) = q13;
    }
    for (; reduction < k; ++reduction) {
      lhs_q13[dst_base + reduction] =
          matmulqhpihvx8rowfp32storemultithreadFloatToQ13(
              static_cast<float>(lhs[src_base + reduction]));
    }
  }
}

static inline HVX_Vector
matmulqhpihvx8rowfp32storemultithreadConvertQ26ToFp32(HVX_Vector values) {
  const HVX_Vector zero = Q6_V_vzero();
  const HVX_Vector fp32 = convert_s32_to_sf(values);

  // Multiplication by 2^-26 is an exact FP32 exponent adjustment. Avoid the
  // vector FP multiply path, which returns zero on the target QHPI runtime.
  const HVX_Vector exponent_delta = Q6_V_vsplat_R(-218103808);
  const HVX_Vector scaled = Q6_Vw_vadd_VwVw(fp32, exponent_delta);
  const HVX_VectorPred is_zero = Q6_Q_vcmp_eq_VwVw(values, zero);
  return Q6_V_vmux_QVV(is_zero, zero, scaled);
}

static inline void matmulqhpihvx8rowfp32storemultithreadCompute1x64Hvx(const __fp16 *lhs,
                                                       const __fp16 *rhs,
                                                       float *output,
                                                       uint64_t lhs_base,
                                                       uint64_t rhs_base,
                                                       uint64_t output_base,
                                                       uint32_t k,
                                                       uint32_t n,
                                                       uint32_t col) {
#if defined(MATMUL_QHPI_HVX_8ROW_FP32_STORE_MULTITHREAD_DEBUG_STORE_RHS)
  (void)lhs;
  (void)lhs_base;
  (void)k;
  const HVX_Vector rhs_vec = vmemu(&rhs[rhs_base + col]);
  vmemu(&output[output_base + col]) = rhs_vec;
#else
  HVX_Vector zero = Q6_V_vzero();
  HVX_VectorPair acc = Q6_W_vcombine_VV(zero, zero);

  for (uint32_t reduction = 0; reduction < k; ++reduction) {
    const int16_t lhs_q13 =
        matmulqhpihvx8rowfp32storemultithreadFloatToQ13(static_cast<float>(lhs[lhs_base + reduction]));
    const HVX_Vector lhs_vec = Q6_Vh_vsplat_R(static_cast<int32_t>(lhs_q13));

    const uint64_t rhs_row_base = rhs_base + (uint64_t)reduction * n + col;
    const HVX_Vector rhs_vec =
        matmulqhpihvx8rowfp32storemultithreadConvert64Fp16ToQ13(
            &rhs[rhs_row_base]);

    acc = Q6_Ww_vmpyacc_WwVhVh(acc, lhs_vec, rhs_vec);
  }

  const HVX_Vector even =
      matmulqhpihvx8rowfp32storemultithreadConvertQ26ToFp32(Q6_V_lo_W(acc));
  const HVX_Vector odd =
      matmulqhpihvx8rowfp32storemultithreadConvertQ26ToFp32(Q6_V_hi_W(acc));
  const HVX_VectorPair interleaved = Q6_W_vshuff_VVR(odd, even, -4);
  vmemu(&output[output_base + col]) = Q6_V_lo_W(interleaved);
  vmemu(&output[output_base + col + 32]) = Q6_V_hi_W(interleaved);
#endif
}

static inline void matmulqhpihvx8rowfp32storemultithreadCompute8x64Hvx(
    const int16_t *lhs,
    const __fp16 *rhs,
    float *output,
    uint64_t lhs_base,
    uint64_t rhs_base,
    uint64_t output_base,
    uint32_t k,
    uint32_t n,
    uint32_t col) {
#if defined(MATMUL_QHPI_HVX_8ROW_FP32_STORE_MULTITHREAD_DEBUG_STORE_RHS)
  (void)lhs;
  (void)lhs_base;
  (void)k;
  const HVX_Vector rhs_vec = vmemu(&rhs[rhs_base + col]);
  for (uint32_t row = 0; row < 8; ++row) {
    vmemu(&output[output_base + (uint64_t)row * n + col]) = rhs_vec;
  }
#else
  const HVX_Vector zero = Q6_V_vzero();
  HVX_VectorPair acc0 = Q6_W_vcombine_VV(zero, zero);
  HVX_VectorPair acc1 = Q6_W_vcombine_VV(zero, zero);
  HVX_VectorPair acc2 = Q6_W_vcombine_VV(zero, zero);
  HVX_VectorPair acc3 = Q6_W_vcombine_VV(zero, zero);
  HVX_VectorPair acc4 = Q6_W_vcombine_VV(zero, zero);
  HVX_VectorPair acc5 = Q6_W_vcombine_VV(zero, zero);
  HVX_VectorPair acc6 = Q6_W_vcombine_VV(zero, zero);
  HVX_VectorPair acc7 = Q6_W_vcombine_VV(zero, zero);

  for (uint32_t reduction = 0; reduction < k; ++reduction) {
    const uint64_t rhs_row_base =
        rhs_base + (uint64_t)reduction * n + col;
    const HVX_Vector rhs_vec =
        matmulqhpihvx8rowfp32storemultithreadConvert64Fp16ToQ13(
            &rhs[rhs_row_base]);

    const HVX_Vector lhs0 = Q6_Vh_vsplat_R(lhs[lhs_base + reduction]);
    const HVX_Vector lhs1 = Q6_Vh_vsplat_R(lhs[lhs_base + k + reduction]);
    const HVX_Vector lhs2 =
        Q6_Vh_vsplat_R(lhs[lhs_base + (uint64_t)2 * k + reduction]);
    const HVX_Vector lhs3 =
        Q6_Vh_vsplat_R(lhs[lhs_base + (uint64_t)3 * k + reduction]);
    const HVX_Vector lhs4 =
        Q6_Vh_vsplat_R(lhs[lhs_base + (uint64_t)4 * k + reduction]);
    const HVX_Vector lhs5 =
        Q6_Vh_vsplat_R(lhs[lhs_base + (uint64_t)5 * k + reduction]);
    const HVX_Vector lhs6 =
        Q6_Vh_vsplat_R(lhs[lhs_base + (uint64_t)6 * k + reduction]);
    const HVX_Vector lhs7 =
        Q6_Vh_vsplat_R(lhs[lhs_base + (uint64_t)7 * k + reduction]);

    acc0 = Q6_Ww_vmpyacc_WwVhVh(acc0, lhs0, rhs_vec);
    acc1 = Q6_Ww_vmpyacc_WwVhVh(acc1, lhs1, rhs_vec);
    acc2 = Q6_Ww_vmpyacc_WwVhVh(acc2, lhs2, rhs_vec);
    acc3 = Q6_Ww_vmpyacc_WwVhVh(acc3, lhs3, rhs_vec);
    acc4 = Q6_Ww_vmpyacc_WwVhVh(acc4, lhs4, rhs_vec);
    acc5 = Q6_Ww_vmpyacc_WwVhVh(acc5, lhs5, rhs_vec);
    acc6 = Q6_Ww_vmpyacc_WwVhVh(acc6, lhs6, rhs_vec);
    acc7 = Q6_Ww_vmpyacc_WwVhVh(acc7, lhs7, rhs_vec);
  }

  const HVX_VectorPair accumulators[8] = {
      acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7};
  for (uint32_t row = 0; row < 8; ++row) {
    const uint64_t row_output_base = output_base + (uint64_t)row * n + col;
    const HVX_Vector even = matmulqhpihvx8rowfp32storemultithreadConvertQ26ToFp32(
        Q6_V_lo_W(accumulators[row]));
    const HVX_Vector odd = matmulqhpihvx8rowfp32storemultithreadConvertQ26ToFp32(
        Q6_V_hi_W(accumulators[row]));
    const HVX_VectorPair interleaved = Q6_W_vshuff_VVR(odd, even, -4);
    vmemu(&output[row_output_base]) = Q6_V_lo_W(interleaved);
    vmemu(&output[row_output_base + 32]) = Q6_V_hi_W(interleaved);
  }
#endif
}

static inline void matmulqhpihvx8rowfp32storemultithreadCompute4x64Hvx(
    const int16_t *lhs,
    const __fp16 *rhs,
    float *output,
    uint64_t lhs_base,
    uint64_t rhs_base,
    uint64_t output_base,
    uint32_t k,
    uint32_t n,
    uint32_t col) {
  const HVX_Vector zero = Q6_V_vzero();
  HVX_VectorPair acc0 = Q6_W_vcombine_VV(zero, zero);
  HVX_VectorPair acc1 = Q6_W_vcombine_VV(zero, zero);
  HVX_VectorPair acc2 = Q6_W_vcombine_VV(zero, zero);
  HVX_VectorPair acc3 = Q6_W_vcombine_VV(zero, zero);

  for (uint32_t reduction = 0; reduction < k; ++reduction) {
    const uint64_t rhs_row_base =
        rhs_base + (uint64_t)reduction * n + col;
    const HVX_Vector rhs_vec =
        matmulqhpihvx8rowfp32storemultithreadConvert64Fp16ToQ13(
            &rhs[rhs_row_base]);

    const HVX_Vector lhs0 = Q6_Vh_vsplat_R(lhs[lhs_base + reduction]);
    const HVX_Vector lhs1 = Q6_Vh_vsplat_R(lhs[lhs_base + k + reduction]);
    const HVX_Vector lhs2 =
        Q6_Vh_vsplat_R(lhs[lhs_base + (uint64_t)2 * k + reduction]);
    const HVX_Vector lhs3 =
        Q6_Vh_vsplat_R(lhs[lhs_base + (uint64_t)3 * k + reduction]);

    acc0 = Q6_Ww_vmpyacc_WwVhVh(acc0, lhs0, rhs_vec);
    acc1 = Q6_Ww_vmpyacc_WwVhVh(acc1, lhs1, rhs_vec);
    acc2 = Q6_Ww_vmpyacc_WwVhVh(acc2, lhs2, rhs_vec);
    acc3 = Q6_Ww_vmpyacc_WwVhVh(acc3, lhs3, rhs_vec);
  }

  const HVX_VectorPair accumulators[4] = {acc0, acc1, acc2, acc3};
  for (uint32_t row = 0; row < 4; ++row) {
    const uint64_t row_output_base = output_base + (uint64_t)row * n + col;
    const HVX_Vector even =
        matmulqhpihvx8rowfp32storemultithreadConvertQ26ToFp32(
            Q6_V_lo_W(accumulators[row]));
    const HVX_Vector odd =
        matmulqhpihvx8rowfp32storemultithreadConvertQ26ToFp32(
            Q6_V_hi_W(accumulators[row]));
    const HVX_VectorPair interleaved = Q6_W_vshuff_VVR(odd, even, -4);
    vmemu(&output[row_output_base]) = Q6_V_lo_W(interleaved);
    vmemu(&output[row_output_base + 32]) = Q6_V_hi_W(interleaved);
  }
}

static inline void matmulqhpihvxstaticrhsprecomputedCompute4x64Hvx(
    const int16_t *lhs,
    const int16_t *rhs,
    float *output,
    uint64_t rhs_base,
    uint64_t output_base,
    uint32_t k,
    uint32_t rhs_stride,
    uint32_t output_stride,
    uint32_t col) {
  const HVX_Vector zero = Q6_V_vzero();
  HVX_VectorPair acc0 = Q6_W_vcombine_VV(zero, zero);
  HVX_VectorPair acc1 = Q6_W_vcombine_VV(zero, zero);
  HVX_VectorPair acc2 = Q6_W_vcombine_VV(zero, zero);
  HVX_VectorPair acc3 = Q6_W_vcombine_VV(zero, zero);

  for (uint32_t reduction = 0; reduction < k; ++reduction) {
    const HVX_Vector rhs_vec =
        vmemu(&rhs[rhs_base + (uint64_t)reduction * rhs_stride + col]);
    acc0 = Q6_Ww_vmpyacc_WwVhVh(
        acc0, Q6_Vh_vsplat_R(lhs[reduction]), rhs_vec);
    acc1 = Q6_Ww_vmpyacc_WwVhVh(
        acc1, Q6_Vh_vsplat_R(lhs[k + reduction]), rhs_vec);
    acc2 = Q6_Ww_vmpyacc_WwVhVh(
        acc2, Q6_Vh_vsplat_R(lhs[(uint64_t)2 * k + reduction]), rhs_vec);
    acc3 = Q6_Ww_vmpyacc_WwVhVh(
        acc3, Q6_Vh_vsplat_R(lhs[(uint64_t)3 * k + reduction]), rhs_vec);
  }

  const HVX_VectorPair accumulators[4] = {acc0, acc1, acc2, acc3};
  for (uint32_t row = 0; row < 4; ++row) {
    const uint64_t row_output_base =
        output_base + (uint64_t)row * output_stride + col;
    const HVX_Vector even =
        matmulqhpihvx8rowfp32storemultithreadConvertQ26ToFp32(
            Q6_V_lo_W(accumulators[row]));
    const HVX_Vector odd =
        matmulqhpihvx8rowfp32storemultithreadConvertQ26ToFp32(
            Q6_V_hi_W(accumulators[row]));
    const HVX_VectorPair interleaved = Q6_W_vshuff_VVR(odd, even, -4);
    vmemu(&output[row_output_base]) = Q6_V_lo_W(interleaved);
    vmemu(&output[row_output_base + 32]) = Q6_V_hi_W(interleaved);
  }
}

static inline void matmulqhpihvxofflineq13Store64(
    HVX_VectorPair accumulator,
    float *output) {
  const HVX_Vector even =
      matmulqhpihvx8rowfp32storemultithreadConvertQ26ToFp32(
          Q6_V_lo_W(accumulator));
  const HVX_Vector odd =
      matmulqhpihvx8rowfp32storemultithreadConvertQ26ToFp32(
          Q6_V_hi_W(accumulator));
  const HVX_VectorPair interleaved = Q6_W_vshuff_VVR(odd, even, -4);
  vmemu(output) = Q6_V_lo_W(interleaved);
  vmemu(output + 32) = Q6_V_hi_W(interleaved);
}

// Process two adjacent 64-column vectors per reduction step. This preserves
// the accumulation order of the 4x64 kernel for every output while reusing
// each of the four LHS splats across twice as many columns.
static inline void matmulqhpihvxofflineq13Compute4x128Hvx(
    const int16_t *lhs,
    const int16_t *rhs,
    float *output,
    uint64_t rhs_base,
    uint64_t output_base,
    uint32_t k,
    uint32_t rhs_stride,
    uint32_t output_stride,
    uint32_t col) {
  const HVX_Vector zero = Q6_V_vzero();
  HVX_VectorPair acc00 = Q6_W_vcombine_VV(zero, zero);
  HVX_VectorPair acc01 = Q6_W_vcombine_VV(zero, zero);
  HVX_VectorPair acc10 = Q6_W_vcombine_VV(zero, zero);
  HVX_VectorPair acc11 = Q6_W_vcombine_VV(zero, zero);
  HVX_VectorPair acc20 = Q6_W_vcombine_VV(zero, zero);
  HVX_VectorPair acc21 = Q6_W_vcombine_VV(zero, zero);
  HVX_VectorPair acc30 = Q6_W_vcombine_VV(zero, zero);
  HVX_VectorPair acc31 = Q6_W_vcombine_VV(zero, zero);

  for (uint32_t reduction = 0; reduction < k; ++reduction) {
    const uint64_t rhs_row =
        rhs_base + (uint64_t)reduction * rhs_stride + col;
    const HVX_Vector rhs0 = vmemu(&rhs[rhs_row]);
    const HVX_Vector rhs1 = vmemu(&rhs[rhs_row + 64]);
    const HVX_Vector lhs0 = Q6_Vh_vsplat_R(lhs[reduction]);
    const HVX_Vector lhs1 = Q6_Vh_vsplat_R(lhs[k + reduction]);
    const HVX_Vector lhs2 =
        Q6_Vh_vsplat_R(lhs[(uint64_t)2 * k + reduction]);
    const HVX_Vector lhs3 =
        Q6_Vh_vsplat_R(lhs[(uint64_t)3 * k + reduction]);
    acc00 = Q6_Ww_vmpyacc_WwVhVh(acc00, lhs0, rhs0);
    acc01 = Q6_Ww_vmpyacc_WwVhVh(acc01, lhs0, rhs1);
    acc10 = Q6_Ww_vmpyacc_WwVhVh(acc10, lhs1, rhs0);
    acc11 = Q6_Ww_vmpyacc_WwVhVh(acc11, lhs1, rhs1);
    acc20 = Q6_Ww_vmpyacc_WwVhVh(acc20, lhs2, rhs0);
    acc21 = Q6_Ww_vmpyacc_WwVhVh(acc21, lhs2, rhs1);
    acc30 = Q6_Ww_vmpyacc_WwVhVh(acc30, lhs3, rhs0);
    acc31 = Q6_Ww_vmpyacc_WwVhVh(acc31, lhs3, rhs1);

  }

  matmulqhpihvxofflineq13Store64(
      acc00, &output[output_base + col]);
  matmulqhpihvxofflineq13Store64(
      acc01, &output[output_base + col + 64]);
  matmulqhpihvxofflineq13Store64(
      acc10, &output[output_base + output_stride + col]);
  matmulqhpihvxofflineq13Store64(
      acc11, &output[output_base + output_stride + col + 64]);
  matmulqhpihvxofflineq13Store64(
      acc20, &output[output_base + (uint64_t)2 * output_stride + col]);
  matmulqhpihvxofflineq13Store64(
      acc21,
      &output[output_base + (uint64_t)2 * output_stride + col + 64]);
  matmulqhpihvxofflineq13Store64(
      acc30, &output[output_base + (uint64_t)3 * output_stride + col]);
  matmulqhpihvxofflineq13Store64(
      acc31,
      &output[output_base + (uint64_t)3 * output_stride + col + 64]);
}

// Decode specialization: one query row. Keeping only two accumulators avoids
// the unused-row work and register pressure of the prefill 4x128 microkernel.
static inline void matmulqhpihvxofflineq13Compute1x128Hvx(
    const int16_t *lhs,
    const int16_t *rhs,
    float *output,
    uint64_t rhs_base,
    uint64_t output_base,
    uint32_t k,
    uint32_t rhs_stride,
    uint32_t col) {
  const HVX_Vector zero = Q6_V_vzero();
  HVX_VectorPair acc0 = Q6_W_vcombine_VV(zero, zero);
  HVX_VectorPair acc1 = Q6_W_vcombine_VV(zero, zero);
  for (uint32_t reduction = 0; reduction < k; ++reduction) {
    const uint64_t rhs_row =
        rhs_base + (uint64_t)reduction * rhs_stride + col;
    const HVX_Vector lhs_value = Q6_Vh_vsplat_R(lhs[reduction]);
    acc0 = Q6_Ww_vmpyacc_WwVhVh(acc0, lhs_value, vmemu(&rhs[rhs_row]));
    acc1 =
        Q6_Ww_vmpyacc_WwVhVh(acc1, lhs_value, vmemu(&rhs[rhs_row + 64]));
  }
  matmulqhpihvxofflineq13Store64(acc0, &output[output_base + col]);
  matmulqhpihvxofflineq13Store64(acc1, &output[output_base + col + 64]);
}

#endif

/*
 * QHPI Registration using hexnn_ffi.h API for MatMulQhpiHvx8RowFp32StoreMultithread
 * Multiple kernels for different data type combinations
 */


// Input tensor signatures for MatMulQhpiHvx8RowFp32StoreMultithread kernel matmulqhpihvx8rowfp32storemultithread_float_16_
// Includes both regular inputs and parameters as inputs
static QHPI_Tensor_Signature_v1 matmulqhpihvx8rowfp32storemultithread_float_16_InputSignatures[] = {

    {
        .element_type = QHPI_Float16,
        .layout = QHPI_Layout_Flat4,
        .storage = QHPI_Storage_Direct,
        .mem_placement = QHPI_MemLoc_DDR_OR_TCM
    },

    {
        .element_type = QHPI_Float16,
        .layout = QHPI_Layout_Flat4,
        .storage = QHPI_Storage_Direct,
        .mem_placement = QHPI_MemLoc_DDR_OR_TCM
    }
};

static QHPI_Tensor_Signature_v1
    matmulqhpihvxstaticrhsprecomputed_float_16_InputSignatures[] = {
    {
        .element_type = QHPI_Float16,
        .layout = QHPI_Layout_Flat4,
        .storage = QHPI_Storage_Direct,
        .mem_placement = QHPI_MemLoc_DDR_OR_TCM
    },
    {
        .element_type = QHPI_Float16,
        .layout = QHPI_Layout_Flat4,
        .storage = QHPI_Storage_Direct,
        .mem_placement = QHPI_MemLoc_DDR_OR_TCM
    },
    {
        .element_type = QHPI_Float32,
        .layout = QHPI_Layout_Flat4,
        .storage = QHPI_Storage_Direct,
        .mem_placement = QHPI_MemLoc_DDR_OR_TCM
    }
};

static QHPI_Tensor_Signature_v1
    matmulqhpihvxofflineq13rhs_float_16_InputSignatures[] = {
    {
        .element_type = QHPI_Float16,
        .layout = QHPI_Layout_Flat4,
        .storage = QHPI_Storage_Direct,
        .mem_placement = QHPI_MemLoc_DDR_OR_TCM
    },
    {
        // QHPI exposes QINT16 but no unquantized INT16 signature. Accept the
        // model's signed fixed-point tensor here; this dedicated op type
        // defines the raw storage contract as Q13 int16_t.
        .element_type = QHPI_ELEMENT_TYPE_ANY,
        .layout = QHPI_Layout_Flat4,
        .storage = QHPI_Storage_Direct,
        .mem_placement = QHPI_MemLoc_DDR_OR_TCM
    }
};

static QHPI_Tensor_Signature_v1 matmulqhpihvx8rowfp32storemultithread_float_16_OutputSignatures[] = {

    {
        .element_type = QHPI_Float32,
        .layout = QHPI_Layout_Flat4,
        .storage = QHPI_Storage_Direct,
        .mem_placement = QHPI_MemLoc_DDR_OR_TCM
    }
};

static QHPI_Tensor_Signature_v1 convertfp16todeviceq13_InputSignatures[] = {
    {
        .element_type = QHPI_Float16,
        .layout = QHPI_Layout_Flat4,
        .storage = QHPI_Storage_Direct,
        .mem_placement = QHPI_MemLoc_DDR_OR_TCM
    }
};

static QHPI_Tensor_Signature_v1 convertfp16todeviceq13_OutputSignatures[] = {
    {
        // QHPI has no unquantized INT16 element enum. The dedicated probe
        // model declares QNN_DATATYPE_INT_16 and consumes the raw bytes.
        .element_type = QHPI_ELEMENT_TYPE_ANY,
        .layout = QHPI_Layout_Flat4,
        .storage = QHPI_Storage_Direct,
        .mem_placement = QHPI_MemLoc_DDR_OR_TCM
    }
};

static QHPI_Tensor_Signature_v1
    qwen2decodeattentionpast128fp32_InputSignatures[] = {
    {
        .element_type = QHPI_Float32,
        .layout = QHPI_Layout_Flat4,
        .storage = QHPI_Storage_Direct,
        .mem_placement = QHPI_MemLoc_DDR_OR_TCM
    },
    {
        .element_type = QHPI_Float32,
        .layout = QHPI_Layout_Flat4,
        .storage = QHPI_Storage_Direct,
        .mem_placement = QHPI_MemLoc_DDR_OR_TCM
    },
    {
        .element_type = QHPI_Float32,
        .layout = QHPI_Layout_Flat4,
        .storage = QHPI_Storage_Direct,
        .mem_placement = QHPI_MemLoc_DDR_OR_TCM
    }
};

static QHPI_Tensor_Signature_v1
    qwen2decodeattentionpast128fp32_OutputSignatures[] = {
    {
        .element_type = QHPI_Float32,
        .layout = QHPI_Layout_Flat4,
        .storage = QHPI_Storage_Direct,
        .mem_placement = QHPI_MemLoc_DDR_OR_TCM
    }
};

// Kernel definition for MatMulQhpiHvx8RowFp32StoreMultithread kernel matmulqhpihvx8rowfp32storemultithread_float_16_
static QHPI_Kernel_v1 matmulqhpihvx8rowfp32storemultithread_float_16_Kernel = {
    .function_name = "matmulqhpihvx8rowfp32storemultithread_float_16_Execute",
    .function = matmulqhpihvx8rowfp32storemultithread_float_16_Execute,
    .resources = QHPI_RESOURCE_HVX,
    .source_destructive = false,
    .multithreaded = true,
    .variable_inputs = false,
    .variable_outputs = false,
    .min_inputs = 2,
    .input_signature = matmulqhpihvx8rowfp32storemultithread_float_16_InputSignatures,
    .min_outputs = 1,
    .output_signature = matmulqhpihvx8rowfp32storemultithread_float_16_OutputSignatures,
    .cost_function = matmulqhpihvx8rowfp32storemultithread_float_16_CostFunc,
    .sync_block_size = 0,
    .precomputed_data_size = 0,
    .do_precomputation_function = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate = nullptr
};

static QHPI_Kernel_v1 matmulqhpihvxstaticrhsprecomputed_float_16_Kernel = {
    .function_name = "matmulqhpihvxstaticrhsprecomputed_float_16_Execute",
    .function = nullptr,
    .resources = QHPI_RESOURCE_HVX,
    .source_destructive = false,
    .multithreaded = true,
    .variable_inputs = false,
    .variable_outputs = false,
    .min_inputs = 3,
    .input_signature =
        matmulqhpihvxstaticrhsprecomputed_float_16_InputSignatures,
    .min_outputs = 1,
    .output_signature =
        matmulqhpihvx8rowfp32storemultithread_float_16_OutputSignatures,
    .cost_function =
        matmulqhpihvx8rowfp32storemultithread_float_16_CostFunc,
    .sync_block_size = 0,
    .precomputed_data_size = kStaticRhsPrecomputedBytes,
    .do_precomputation_function =
        matmulqhpihvxstaticrhsprecomputed_float_16_Precompute,
    .function_with_precomputed_data =
        matmulqhpihvxstaticrhsprecomputed_float_16_Execute,
    .predicate = nullptr
};

static QHPI_Kernel_v1 matmulqhpihvxofflineq13rhs_float_16_Kernel = {
    .function_name = "matmulqhpihvxofflineq13rhs_float_16_Execute",
    .function = matmulqhpihvxofflineq13rhs_float_16_Execute,
    .resources = QHPI_RESOURCE_HVX,
    .source_destructive = false,
    .multithreaded = true,
    .variable_inputs = false,
    .variable_outputs = false,
    .min_inputs = 2,
    .input_signature =
        matmulqhpihvxofflineq13rhs_float_16_InputSignatures,
    .min_outputs = 1,
    .output_signature =
        matmulqhpihvx8rowfp32storemultithread_float_16_OutputSignatures,
    .cost_function =
        matmulqhpihvx8rowfp32storemultithread_float_16_CostFunc,
    .sync_block_size = 0,
    .precomputed_data_size = 0,
    .do_precomputation_function = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate = nullptr
};

static QHPI_Kernel_v1 convertfp16todeviceq13_Kernel = {
    .function_name = "convertfp16todeviceq13_Execute",
    .function = convertfp16todeviceq13_Execute,
    .resources = QHPI_RESOURCE_HVX,
    .source_destructive = false,
    .multithreaded = true,
    .variable_inputs = false,
    .variable_outputs = false,
    .min_inputs = 1,
    .input_signature = convertfp16todeviceq13_InputSignatures,
    .min_outputs = 1,
    .output_signature = convertfp16todeviceq13_OutputSignatures,
    .cost_function =
        matmulqhpihvx8rowfp32storemultithread_float_16_CostFunc,
    .sync_block_size = 0,
    .precomputed_data_size = 0,
    .do_precomputation_function = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate = nullptr
};

static QHPI_Kernel_v1 qwen2decodeattentionpast128fp32_Kernel = {
    .function_name = "qwen2decodeattentionpast128fp32_Execute",
    .function = qwen2decodeattentionpast128fp32_Execute,
    .resources = QHPI_RESOURCE_HVX,
    .source_destructive = false,
    .multithreaded = true,
    .variable_inputs = false,
    .variable_outputs = false,
    .min_inputs = 3,
    .input_signature = qwen2decodeattentionpast128fp32_InputSignatures,
    .min_outputs = 1,
    .output_signature = qwen2decodeattentionpast128fp32_OutputSignatures,
    .cost_function =
        matmulqhpihvx8rowfp32storemultithread_float_16_CostFunc,
    .sync_block_size = 0,
    .precomputed_data_size = 0,
    .do_precomputation_function = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate = nullptr
};

// Array of all kernels for MatMulQhpiHvx8RowFp32StoreMultithread
static QHPI_Kernel_v1 matmulqhpihvx8rowfp32storemultithreadKernels[] = {

    matmulqhpihvx8rowfp32storemultithread_float_16_Kernel
};

// Operator info for MatMulQhpiHvx8RowFp32StoreMultithread - exported for package registration
QHPI_OpInfo_v1 matmulqhpihvx8rowfp32storemultithreadOpInfo = {
    .name = THIS_PKG_NAME_STR "::" "MatMulQhpiHvx8RowFp32StoreMultithread",
    .num_kernels = 1,
    .kernels = matmulqhpihvx8rowfp32storemultithreadKernels,
    .early_rewrite = matmulqhpihvx8rowfp32storemultithreadEarlyRewrite,
    .shape_required = matmulqhpihvx8rowfp32storemultithreadShapeRequired,
    .shape_legalized = matmulqhpihvx8rowfp32storemultithreadShapeLegal,
    .build_tile = matmulqhpihvx8rowfp32storemultithreadBuildTile,
    .late_rewrite = matmulqhpihvx8rowfp32storemultithreadLateRewrite
};

static QHPI_Kernel_v1 matmulqhpihvxstaticrhsprecomputedKernels[] = {
    matmulqhpihvxstaticrhsprecomputed_float_16_Kernel
};

QHPI_OpInfo_v1 matmulqhpihvxstaticrhsprecomputedOpInfo = {
    .name = THIS_PKG_NAME_STR "::" "MatMulQhpiHvxStaticRhsPrecomputed",
    .num_kernels = 1,
    .kernels = matmulqhpihvxstaticrhsprecomputedKernels,
    .early_rewrite = matmulqhpihvx8rowfp32storemultithreadEarlyRewrite,
    .shape_required = matmulqhpihvx8rowfp32storemultithreadShapeRequired,
    .shape_legalized = matmulqhpihvx8rowfp32storemultithreadShapeLegal,
    .build_tile = matmulqhpihvx8rowfp32storemultithreadBuildTile,
    .late_rewrite = matmulqhpihvx8rowfp32storemultithreadLateRewrite
};

static QHPI_Kernel_v1 matmulqhpihvxofflineq13rhsKernels[] = {
    matmulqhpihvxofflineq13rhs_float_16_Kernel
};

QHPI_OpInfo_v1 matmulqhpihvxofflineq13rhsOpInfo = {
    .name = THIS_PKG_NAME_STR "::"
            "MatMulQhpiHvxOfflineQ13RhsFp32StoreMultithread",
    .num_kernels = 1,
    .kernels = matmulqhpihvxofflineq13rhsKernels,
    .early_rewrite = matmulqhpihvx8rowfp32storemultithreadEarlyRewrite,
    .shape_required = matmulqhpihvx8rowfp32storemultithreadShapeRequired,
    .shape_legalized = matmulqhpihvx8rowfp32storemultithreadShapeLegal,
    .build_tile = matmulqhpihvx8rowfp32storemultithreadBuildTile,
    .late_rewrite = matmulqhpihvx8rowfp32storemultithreadLateRewrite
};

static QHPI_Kernel_v1 convertfp16todeviceq13Kernels[] = {
    convertfp16todeviceq13_Kernel
};

QHPI_OpInfo_v1 convertfp16todeviceq13OpInfo = {
    .name = THIS_PKG_NAME_STR "::ConvertFp16ToDeviceQ13",
    .num_kernels = 1,
    .kernels = convertfp16todeviceq13Kernels,
    .early_rewrite = matmulqhpihvx8rowfp32storemultithreadEarlyRewrite,
    .shape_required = matmulqhpihvx8rowfp32storemultithreadShapeRequired,
    .shape_legalized = matmulqhpihvx8rowfp32storemultithreadShapeLegal,
    .build_tile = matmulqhpihvx8rowfp32storemultithreadBuildTile,
    .late_rewrite = matmulqhpihvx8rowfp32storemultithreadLateRewrite
};

static QHPI_Kernel_v1 qwen2decodeattentionpast128fp32Kernels[] = {
    qwen2decodeattentionpast128fp32_Kernel
};

QHPI_OpInfo_v1 qwen2decodeattentionpast128fp32OpInfo = {
    .name = THIS_PKG_NAME_STR "::Qwen2DecodeAttentionPast128Fp32",
    .num_kernels = 1,
    .kernels = qwen2decodeattentionpast128fp32Kernels,
    .early_rewrite = matmulqhpihvx8rowfp32storemultithreadEarlyRewrite,
    .shape_required = matmulqhpihvx8rowfp32storemultithreadShapeRequired,
    .shape_legalized = matmulqhpihvx8rowfp32storemultithreadShapeLegal,
    .build_tile = matmulqhpihvx8rowfp32storemultithreadBuildTile,
    .late_rewrite = matmulqhpihvx8rowfp32storemultithreadLateRewrite
};


/* QHPI execute function implementation for MatMulQhpiHvx8RowFp32StoreMultithread kernel matmulqhpihvx8rowfp32storemultithread_float_16_ */
static uint32_t matmulqhpihvx8rowfp32storemultithread_float_16_Execute(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs,
    QHPI_Tensor **outputs,
    uint32_t num_inputs,
    const QHPI_Tensor *const *inputs) {
  if (num_inputs != 2 || num_outputs != 1) {
    return QHPI_ErrorFatal;
  }

  if (inputs == nullptr || outputs == nullptr ||
      inputs[0] == nullptr || inputs[1] == nullptr ||
      outputs[0] == nullptr) {
    return QHPI_ErrorFatal;
  }

  const QHPI_Shape lhs_shape = qhpi_tensor_shape(inputs[0]);
  const QHPI_Shape rhs_shape = qhpi_tensor_shape(inputs[1]);
  const QHPI_Shape out_shape = qhpi_tensor_shape(outputs[0]);

  if (lhs_shape.rank != 4 ||
      rhs_shape.rank != 4 ||
      out_shape.rank != 4) {
    return QHPI_ErrorFatal;
  }

  const uint32_t batch = lhs_shape.dims[0];
  const uint32_t heads = lhs_shape.dims[1];
  const uint32_t m     = lhs_shape.dims[2];
  const uint32_t k     = lhs_shape.dims[3];

  const uint32_t rhs_batch = rhs_shape.dims[0];
  const uint32_t rhs_heads = rhs_shape.dims[1];
  const uint32_t rhs_k     = rhs_shape.dims[2];
  const uint32_t n         = rhs_shape.dims[3];

  if (batch != rhs_batch ||
      heads != rhs_heads ||
      k != rhs_k) {
    return QHPI_ErrorFatal;
  }

  if (out_shape.dims[0] != batch ||
      out_shape.dims[1] != heads ||
      out_shape.dims[2] != m ||
      out_shape.dims[3] != n) {
    return QHPI_ErrorFatal;
  }

  const __fp16 *lhs =
      static_cast<const __fp16 *>(qhpi_tensor_raw_data(inputs[0]));

  const __fp16 *rhs =
      static_cast<const __fp16 *>(qhpi_tensor_raw_data(inputs[1]));

  float *output =
      static_cast<float *>(qhpi_tensor_raw_data(outputs[0]));

  if (lhs == nullptr || rhs == nullptr || output == nullptr) {
    return QHPI_ErrorFatal;
  }

  constexpr uint32_t kMaxTileCacheReduction = 1024;
  if (k > kMaxTileCacheReduction) {
    return QHPI_ErrorFatal;
  }

  const uint32_t num_slices = qhpi_num_slices(handle);
  const uint32_t slice = qhpi_slice_number(handle);
  if (num_slices == 0 || slice >= num_slices) {
    return QHPI_ErrorFatal;
  }

  // An 8x64 tile reuses each converted RHS vector across eight output rows.
  // Self-slicing assigns disjoint row tiles to HVX workers, so no shared
  // synchronization is required.
  for (uint32_t b = 0; b < batch; ++b) {
    for (uint32_t h = 0; h < heads; ++h) {
      const uint64_t rhs_base =
          ((uint64_t)b * heads + h) * k * n;
      uint32_t full_rows = 0;
#if MATMUL_QHPI_HVX_8ROW_FP32_STORE_MULTITHREAD_INTRINSICS && \
    !defined(MATMUL_QHPI_HVX_8ROW_FP32_STORE_MULTITHREAD_FORCE_SCALAR)
      // Qwen2.5-0.5B seq16 has exactly two 8-row tiles. Use 4-row tiles
      // when four workers are available so every worker receives useful
      // work. Larger matrices retain the 8-row path for better RHS reuse.
      const bool use_four_row_tiles = m == 16 && num_slices >= 4;
      const uint32_t tile_rows = use_four_row_tiles ? 4 : 8;
      full_rows = m - (m % tile_rows);
      for (uint32_t row = slice * tile_rows;
           row < full_rows;
           row += num_slices * tile_rows) {
        const uint64_t lhs_base =
            (((uint64_t)b * heads + h) * m + row) * k;
        const uint64_t output_base =
            (((uint64_t)b * heads + h) * m + row) * n;

        // This cache is private to the current QHPI worker invocation. Convert
        // each LHS value once per row tile, then reuse it across all
        // 64-column output blocks.
        alignas(128) int16_t lhs_tile_q13[8 * kMaxTileCacheReduction];
        matmulqhpihvx8rowfp32storemultithreadPackLhsRowsToQ13(
            lhs, lhs_tile_q13, lhs_base, tile_rows, k);

        uint32_t col = 0;
        for (; col + 64 <= n; col += 64) {
          if (use_four_row_tiles) {
            matmulqhpihvx8rowfp32storemultithreadCompute4x64Hvx(
                lhs_tile_q13,
                rhs,
                output,
                0,
                rhs_base,
                output_base,
                k,
                n,
                col);
          } else {
            matmulqhpihvx8rowfp32storemultithreadCompute8x64Hvx(
                lhs_tile_q13,
                rhs,
                output,
                0,
                rhs_base,
                output_base,
                k,
                n,
                col);
          }
        }

        for (; col < n; ++col) {
          for (uint32_t tile_row = 0; tile_row < tile_rows; ++tile_row) {
            const uint64_t tile_lhs_base =
                (uint64_t)tile_row * k;
            const uint64_t tile_output_base =
                output_base + (uint64_t)tile_row * n;
            output[tile_output_base + col] =
                matmulqhpihvx8rowfp32storemultithreadScalarDotQ13(
                    lhs_tile_q13,
                    rhs,
                    tile_lhs_base,
                    rhs_base,
                    k,
                    n,
                    col);
          }
        }
      }
#endif

      for (uint32_t row = full_rows + slice;
           row < m;
           row += num_slices) {
        const uint64_t lhs_base =
            (((uint64_t)b * heads + h) * m + row) * k;
        const uint64_t output_base =
            (((uint64_t)b * heads + h) * m + row) * n;
        uint32_t col = 0;
#if MATMUL_QHPI_HVX_8ROW_FP32_STORE_MULTITHREAD_INTRINSICS && \
    !defined(MATMUL_QHPI_HVX_8ROW_FP32_STORE_MULTITHREAD_FORCE_SCALAR)
        for (; col + 64 <= n; col += 64) {
          matmulqhpihvx8rowfp32storemultithreadCompute1x64Hvx(
              lhs, rhs, output, lhs_base, rhs_base, output_base, k, n, col);
        }
#endif
        for (; col < n; ++col) {
          output[output_base + col] =
              matmulqhpihvx8rowfp32storemultithreadScalarDot(
                  lhs, rhs, lhs_base, rhs_base, k, n, col);
        }
      }
    }
  }

  return QHPI_Success;
}

static uint32_t matmulqhpihvxstaticrhsprecomputed_float_16_Precompute(
    QHPI_RuntimeHandle *handle,
    void *data,
    uint32_t num_outputs,
    QHPI_Tensor **outputs,
    uint32_t num_inputs,
    const QHPI_Tensor *const *inputs) {
  (void)handle;
  if (data == nullptr || num_inputs != 3 || num_outputs != 1 ||
      inputs == nullptr || outputs == nullptr || inputs[0] == nullptr ||
      inputs[1] == nullptr || inputs[2] == nullptr || outputs[0] == nullptr) {
    return QHPI_ErrorFatal;
  }

  const QHPI_Shape lhs_shape = qhpi_tensor_shape(inputs[0]);
  const QHPI_Shape rhs_shape = qhpi_tensor_shape(inputs[1]);
  const QHPI_Shape static_rhs_shape = qhpi_tensor_shape(inputs[2]);
  const QHPI_Shape out_shape = qhpi_tensor_shape(outputs[0]);
  if (lhs_shape.rank != 4 || rhs_shape.rank != 4 ||
      static_rhs_shape.rank != 4 || out_shape.rank != 4) {
    return QHPI_ErrorFatal;
  }

  const uint32_t batch = lhs_shape.dims[0];
  const uint32_t heads = lhs_shape.dims[1];
  const uint32_t m = lhs_shape.dims[2];
  const uint32_t k = lhs_shape.dims[3];
  const uint32_t n = rhs_shape.dims[3];
  const uint32_t packed_columns =
      n < kStaticRhsPackedColumns ? n : kStaticRhsPackedColumns;
  const uint64_t rhs_elements =
      (uint64_t)batch * heads * k * packed_columns;
  if (batch != rhs_shape.dims[0] || heads != rhs_shape.dims[1] ||
      k != rhs_shape.dims[2] ||
      static_rhs_shape.dims[0] != batch ||
      static_rhs_shape.dims[1] != heads ||
      static_rhs_shape.dims[2] != k ||
      static_rhs_shape.dims[3] != n ||
      out_shape.dims[0] != batch || out_shape.dims[1] != heads ||
      out_shape.dims[2] != m || out_shape.dims[3] != n ||
      k > 1024 || (m % 4) != 0 || (n % 64) != 0 ||
      rhs_elements > kStaticRhsMaxElements) {
    return QHPI_ErrorFatal;
  }

  const __fp16 *lhs =
      static_cast<const __fp16 *>(qhpi_tensor_raw_data(inputs[0]));
  const __fp16 *rhs_fp16 =
      static_cast<const __fp16 *>(qhpi_tensor_raw_data(inputs[1]));
  const float *static_rhs =
      static_cast<const float *>(qhpi_tensor_raw_data(inputs[2]));
  float *output =
      static_cast<float *>(qhpi_tensor_raw_data(outputs[0]));
  if (lhs == nullptr || rhs_fp16 == nullptr ||
      static_rhs == nullptr || output == nullptr) {
    return QHPI_ErrorFatal;
  }

  auto *header = static_cast<StaticRhsPrecomputedHeader *>(data);
  header->magic = kStaticRhsMagic;
  header->batch = batch;
  header->heads = heads;
  header->m = m;
  header->k = k;
  header->n = n;
  header->lhs = reinterpret_cast<uintptr_t>(lhs);
  header->rhs_fp16 = reinterpret_cast<uintptr_t>(rhs_fp16);
  header->output = reinterpret_cast<uintptr_t>(output);

  int16_t *packed_rhs = staticRhsPackedData(data);
  uint64_t packed_index = 0;
  for (uint32_t b = 0; b < batch; ++b) {
    for (uint32_t h = 0; h < heads; ++h) {
      const uint64_t rhs_base = ((uint64_t)b * heads + h) * k * n;
      for (uint32_t reduction = 0; reduction < k; ++reduction) {
        const uint64_t rhs_row = rhs_base + (uint64_t)reduction * n;
        for (uint32_t col = 0; col < packed_columns; ++col) {
          packed_rhs[packed_index++] =
              matmulqhpihvx8rowfp32storemultithreadFloatToQ13(
                  static_rhs[rhs_row + col]);
        }
      }
  }
  }
  return QHPI_Success;
}

static uint32_t matmulqhpihvxstaticrhsprecomputed_float_16_Execute(
    QHPI_RuntimeHandle *handle,
    const void *precomputed_data) {
  if (precomputed_data == nullptr) {
    return QHPI_ErrorFatal;
  }
  const auto *header =
      static_cast<const StaticRhsPrecomputedHeader *>(precomputed_data);
  if (header->magic != kStaticRhsMagic) {
    return QHPI_ErrorFatal;
  }

#if MATMUL_QHPI_HVX_8ROW_FP32_STORE_MULTITHREAD_INTRINSICS && \
    !defined(MATMUL_QHPI_HVX_8ROW_FP32_STORE_MULTITHREAD_FORCE_SCALAR)
  const __fp16 *lhs = reinterpret_cast<const __fp16 *>(header->lhs);
  const __fp16 *rhs_fp16 =
      reinterpret_cast<const __fp16 *>(header->rhs_fp16);
  float *output = reinterpret_cast<float *>(header->output);
  const int16_t *rhs = staticRhsPackedData(precomputed_data);
  const uint32_t num_slices = qhpi_num_slices(handle);
  const uint32_t slice = qhpi_slice_number(handle);
  if (lhs == nullptr || rhs_fp16 == nullptr || output == nullptr ||
      num_slices == 0 ||
      slice >= num_slices) {
    return QHPI_ErrorFatal;
  }

  constexpr uint32_t kMaxReduction = 1024;
  for (uint32_t b = 0; b < header->batch; ++b) {
    for (uint32_t h = 0; h < header->heads; ++h) {
      const uint64_t rhs_base =
          ((uint64_t)b * header->heads + h) * header->k * header->n;
      const uint64_t packed_rhs_base =
          ((uint64_t)b * header->heads + h) * header->k *
          kStaticRhsPackedColumns;
      for (uint32_t row = slice * 4; row < header->m;
           row += num_slices * 4) {
        const uint64_t lhs_base =
            (((uint64_t)b * header->heads + h) * header->m + row) *
            header->k;
        const uint64_t output_base =
            (((uint64_t)b * header->heads + h) * header->m + row) *
            header->n;
        alignas(128) int16_t lhs_q13[4 * kMaxReduction];
        matmulqhpihvx8rowfp32storemultithreadPackLhsRowsToQ13(
            lhs, lhs_q13, lhs_base, 4, header->k);
        uint32_t col = 0;
        for (; col < kStaticRhsPackedColumns && col < header->n; col += 64) {
          matmulqhpihvxstaticrhsprecomputedCompute4x64Hvx(
              lhs_q13, rhs, output, packed_rhs_base, output_base,
              header->k, kStaticRhsPackedColumns, header->n, col);
        }
        for (; col < header->n; col += 64) {
          matmulqhpihvx8rowfp32storemultithreadCompute4x64Hvx(
              lhs_q13, rhs_fp16, output, 0, rhs_base, output_base,
              header->k, header->n, col);
        }
      }
    }
  }
  return QHPI_Success;
#else
  (void)handle;
  return QHPI_ErrorFatal;
#endif
}

static uint32_t matmulqhpihvxofflineq13rhs_float_16_Execute(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs,
    QHPI_Tensor **outputs,
    uint32_t num_inputs,
    const QHPI_Tensor *const *inputs) {
  if (num_inputs != 2 || num_outputs != 1 || inputs == nullptr ||
      outputs == nullptr || inputs[0] == nullptr || inputs[1] == nullptr ||
      outputs[0] == nullptr) {
    return QHPI_ErrorFatal;
  }

  const QHPI_Shape lhs_shape = qhpi_tensor_shape(inputs[0]);
  const QHPI_Shape rhs_shape = qhpi_tensor_shape(inputs[1]);
  const QHPI_Shape out_shape = qhpi_tensor_shape(outputs[0]);
  if (lhs_shape.rank != 4 || rhs_shape.rank != 4 || out_shape.rank != 4) {
    return QHPI_ErrorFatal;
  }

  const uint32_t batch = lhs_shape.dims[0];
  const uint32_t heads = lhs_shape.dims[1];
  const uint32_t m = lhs_shape.dims[2];
  const uint32_t k = lhs_shape.dims[3];
  const uint32_t n = rhs_shape.dims[3];
  if (batch != rhs_shape.dims[0] || heads != rhs_shape.dims[1] ||
      k != rhs_shape.dims[2] || out_shape.dims[0] != batch ||
      out_shape.dims[1] != heads || out_shape.dims[2] != m ||
      out_shape.dims[3] != n || k > 1024 ||
      (m != 1 && (m % 4) != 0) ||
      (n % 64) != 0) {
    return QHPI_ErrorFatal;
  }

  const __fp16 *lhs =
      static_cast<const __fp16 *>(qhpi_tensor_raw_data(inputs[0]));
  const int16_t *rhs =
      static_cast<const int16_t *>(qhpi_tensor_raw_data(inputs[1]));
  float *output =
      static_cast<float *>(qhpi_tensor_raw_data(outputs[0]));
  const uint32_t num_slices = qhpi_num_slices(handle);
  const uint32_t slice = qhpi_slice_number(handle);
  if (lhs == nullptr || rhs == nullptr || output == nullptr ||
      num_slices == 0 || slice >= num_slices) {
    return QHPI_ErrorFatal;
  }

#if MATMUL_QHPI_HVX_8ROW_FP32_STORE_MULTITHREAD_INTRINSICS && \
    !defined(MATMUL_QHPI_HVX_8ROW_FP32_STORE_MULTITHREAD_FORCE_SCALAR)
  constexpr uint32_t kMaxReduction = 1024;
  for (uint32_t b = 0; b < batch; ++b) {
    for (uint32_t h = 0; h < heads; ++h) {
      const uint64_t rhs_base = ((uint64_t)b * heads + h) * k * n;
      if (m == 1) {
        // QHPI invokes every slice. A single decode row is deliberately owned
        // by slice zero so it is computed exactly once.
        if (slice != 0) {
          continue;
        }
        const uint64_t lhs_base = ((uint64_t)b * heads + h) * k;
        const uint64_t output_base = ((uint64_t)b * heads + h) * n;
        alignas(128) int16_t lhs_q13[kMaxReduction];
        matmulqhpihvx8rowfp32storemultithreadPackLhsRowsToQ13(
            lhs, lhs_q13, lhs_base, 1, k);
        uint32_t col = 0;
        for (; col + 128 <= n; col += 128) {
          matmulqhpihvxofflineq13Compute1x128Hvx(
              lhs_q13, rhs, output, rhs_base, output_base, k, n, col);
        }
        // The current Qwen projection has N=896 and therefore no tail.
        // Reject other one-row shapes instead of reading nonexistent rows
        // through the 4x64 fallback.
        if (col != n) {
          return QHPI_ErrorFatal;
        }
        continue;
      }
      for (uint32_t row = slice * 4; row < m; row += num_slices * 4) {
        const uint64_t lhs_base =
            (((uint64_t)b * heads + h) * m + row) * k;
        const uint64_t output_base =
            (((uint64_t)b * heads + h) * m + row) * n;
        alignas(128) int16_t lhs_q13[4 * kMaxReduction];
        matmulqhpihvx8rowfp32storemultithreadPackLhsRowsToQ13(
            lhs, lhs_q13, lhs_base, 4, k);
        uint32_t col = 0;
        for (; col + 128 <= n; col += 128) {
          matmulqhpihvxofflineq13Compute4x128Hvx(
              lhs_q13, rhs, output, rhs_base, output_base, k, n, n, col);
        }
        for (; col < n; col += 64) {
          matmulqhpihvxstaticrhsprecomputedCompute4x64Hvx(
              lhs_q13, rhs, output, rhs_base, output_base, k, n, n, col);
        }
      }
    }
  }
  return QHPI_Success;
#else
  return QHPI_ErrorFatal;
#endif
}

static uint32_t convertfp16todeviceq13_Execute(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs,
    QHPI_Tensor **outputs,
    uint32_t num_inputs,
    const QHPI_Tensor *const *inputs) {
  if (num_inputs != 1 || num_outputs != 1 || inputs == nullptr ||
      outputs == nullptr || inputs[0] == nullptr || outputs[0] == nullptr) {
    return QHPI_ErrorFatal;
  }
  const QHPI_Shape input_shape = qhpi_tensor_shape(inputs[0]);
  const QHPI_Shape output_shape = qhpi_tensor_shape(outputs[0]);
  if (input_shape.rank != output_shape.rank || input_shape.rank == 0) {
    return QHPI_ErrorFatal;
  }
  uint64_t elements = 1;
  for (uint32_t dim = 0; dim < input_shape.rank; ++dim) {
    if (input_shape.dims[dim] != output_shape.dims[dim]) {
      return QHPI_ErrorFatal;
    }
    elements *= input_shape.dims[dim];
  }
  const __fp16 *input =
      static_cast<const __fp16 *>(qhpi_tensor_raw_data(inputs[0]));
  int16_t *output =
      static_cast<int16_t *>(qhpi_tensor_raw_data(outputs[0]));
  const uint32_t num_slices = qhpi_num_slices(handle);
  const uint32_t slice = qhpi_slice_number(handle);
  if (input == nullptr || output == nullptr || num_slices == 0 ||
      slice >= num_slices) {
    return QHPI_ErrorFatal;
  }
#if MATMUL_QHPI_HVX_8ROW_FP32_STORE_MULTITHREAD_INTRINSICS
  constexpr uint64_t kVectorElements = 64;
  const uint64_t vector_count = elements / kVectorElements;
  for (uint64_t vector = slice; vector < vector_count;
       vector += num_slices) {
    const uint64_t offset = vector * kVectorElements;
    const HVX_Vector converted =
        matmulqhpihvx8rowfp32storemultithreadConvert64Fp16ToQ13(
            &input[offset]);
    vmemu(&output[offset]) = converted;
  }
  if (slice == 0) {
    for (uint64_t offset = vector_count * kVectorElements;
         offset < elements; ++offset) {
      output[offset] =
          matmulqhpihvx8rowfp32storemultithreadFloatToQ13(
              static_cast<float>(input[offset]));
    }
  }
  return QHPI_Success;
#else
  (void)elements;
  return QHPI_ErrorFatal;
#endif
}

/*
 * Fixed-shape Qwen2.5-0.5B decode attention for past_len=128:
 *   Q [1,14,1,64], K [1,129,64,2] (NHWC), V [1,2,129,64].
 *
 * Each QHPI worker owns complete query heads. Online softmax fuses score,
 * normalization, and value accumulation without materializing the
 * [14,129] score/probability tensors.
 */
static uint32_t qwen2decodeattentionpast128fp32_Execute(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs,
    QHPI_Tensor **outputs,
    uint32_t num_inputs,
    const QHPI_Tensor *const *inputs) {
  if (num_inputs != 3 || num_outputs != 1 || inputs == nullptr ||
      outputs == nullptr || inputs[0] == nullptr || inputs[1] == nullptr ||
      inputs[2] == nullptr || outputs[0] == nullptr) {
    return QHPI_ErrorFatal;
  }

  const QHPI_Shape q_shape = qhpi_tensor_shape(inputs[0]);
  const QHPI_Shape k_shape = qhpi_tensor_shape(inputs[1]);
  const QHPI_Shape v_shape = qhpi_tensor_shape(inputs[2]);
  const QHPI_Shape o_shape = qhpi_tensor_shape(outputs[0]);
  if (q_shape.rank != 4 || k_shape.rank != 4 || v_shape.rank != 4 ||
      o_shape.rank != 4 ||
      q_shape.dims[0] != 1 || q_shape.dims[1] != 14 ||
      q_shape.dims[2] != 1 || q_shape.dims[3] != 64 ||
      k_shape.dims[0] != 1 || k_shape.dims[1] != 129 ||
      k_shape.dims[2] != 64 || k_shape.dims[3] != 2 ||
      v_shape.dims[0] != 1 || v_shape.dims[1] != 2 ||
      v_shape.dims[2] != 129 || v_shape.dims[3] != 64 ||
      o_shape.dims[0] != 1 || o_shape.dims[1] != 14 ||
      o_shape.dims[2] != 1 || o_shape.dims[3] != 64) {
    return QHPI_ErrorFatal;
  }

  const float *query =
      static_cast<const float *>(qhpi_tensor_raw_data(inputs[0]));
  const float *key =
      static_cast<const float *>(qhpi_tensor_raw_data(inputs[1]));
  const float *value =
      static_cast<const float *>(qhpi_tensor_raw_data(inputs[2]));
  float *output =
      static_cast<float *>(qhpi_tensor_raw_data(outputs[0]));
  const uint32_t num_slices = qhpi_num_slices(handle);
  const uint32_t slice = qhpi_slice_number(handle);
  if (query == nullptr || key == nullptr || value == nullptr ||
      output == nullptr || num_slices == 0 || slice >= num_slices) {
    return QHPI_ErrorFatal;
  }

  constexpr uint32_t kQueryHeads = 14;
  constexpr uint32_t kGroupsPerKvHead = 7;
  constexpr uint32_t kSequence = 129;
  constexpr uint32_t kHeadDim = 64;
  constexpr float kScale = 0.125f;

  for (uint32_t query_head = slice; query_head < kQueryHeads;
       query_head += num_slices) {
    const uint32_t kv_head = query_head / kGroupsPerKvHead;
    const float *q = query + query_head * kHeadDim;
    const float *v_head =
        value + static_cast<uint64_t>(kv_head) * kSequence * kHeadDim;
    float accumulator[kHeadDim] = {};

    float running_max = -3.402823466e+38f;
    float running_sum = 0.0f;
    for (uint32_t token = 0; token < kSequence; ++token) {
      float score = 0.0f;
      const float *k_token =
          key + static_cast<uint64_t>(token) * kHeadDim * 2 + kv_head;
      for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
        score += q[dim] * k_token[dim * 2];
      }
      score *= kScale;

      const float next_max = score > running_max ? score : running_max;
      const float old_weight = std::exp(running_max - next_max);
      const float new_weight = std::exp(score - next_max);
      const float *v_token = v_head + token * kHeadDim;
      for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
        accumulator[dim] =
            accumulator[dim] * old_weight + v_token[dim] * new_weight;
      }
      running_sum = running_sum * old_weight + new_weight;
      running_max = next_max;
    }

    const float inverse_sum = 1.0f / running_sum;
    float *out = output + query_head * kHeadDim;
    for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
      out[dim] = accumulator[dim] * inverse_sum;
    }
  }
  return QHPI_Success;
}

static float matmulqhpihvx8rowfp32storemultithread_float_16_CostFunc(const uint32_t num_inputs, const QHPI_Tensor *const *inputs)
{
  /*
   * Cost estimation function for MatMulQhpiHvx8RowFp32StoreMultithread kernel matmulqhpihvx8rowfp32storemultithread_float_16_
   * Return approximate number of cycles needed for this operation
   * with the specific data type combination. Used for estimating cycle
   * performance of a graph.
   *
   * Parameters:
   * - num_inputs: Number of input tensors
   * - inputs: Array of input tensor pointers
   */

  float cost = 1000.0;  // add cost computation here based on tensor sizes and data types
  return cost;
}

/*
 * Common stub implementations for MatMulQhpiHvx8RowFp32StoreMultithread QHPI_OpInfo functions
 * These are shared across all kernels and provide default no-op implementations
 */

static const QHPI_Op* matmulqhpihvx8rowfp32storemultithreadEarlyRewrite(const QHPI_Op *op)
{
  /*
   * Early rewrite function for MatMulQhpiHvx8RowFp32StoreMultithread
   * Called during graph optimization phase
   * Return the original op if no rewriting is needed, or a new op if rewriting is required
   */
  return op;  // No rewriting by default
}

static QHPI_Shape matmulqhpihvx8rowfp32storemultithreadShapeRequired(const QHPI_Op *op)
{
  /*
   * Shape required function for MatMulQhpiHvx8RowFp32StoreMultithread
   * Specifies required input shapes for the operation
   * Return empty shape if no specific shape requirements
   */
  QHPI_Shape empty_shape = {0};  // Empty shape by default
  return empty_shape;
}

static QHPI_Shape matmulqhpihvx8rowfp32storemultithreadShapeLegal(const QHPI_Op *op, const QHPI_Shape* shape)
{
  /*
   * Shape legal function for MatMulQhpiHvx8RowFp32StoreMultithread
   * Validates if a given shape is legal for this operation
   * Return the shape if legal, or modified shape if not legal
   */
  return *shape;  // Accept the provided shape by default
}

static const QHPI_Op* matmulqhpihvx8rowfp32storemultithreadBuildTile(const QHPI_Op *op, const QHPI_Shape* start, const QHPI_Shape* extent)
{
  /*
   * Build tile function for MatMulQhpiHvx8RowFp32StoreMultithread
   * Creates a tiled version of the operation
   * Return a new op that operates on the specified tile, or op if tiling is not supported
   */
  return op;  // No tiling support by default
}

static const QHPI_Op* matmulqhpihvx8rowfp32storemultithreadLateRewrite(const QHPI_Op *op)
{
  /*
   * Late rewrite function for MatMulQhpiHvx8RowFp32StoreMultithread
   * Called during late optimization phase
   * Return the original op if no rewriting is needed, or a new op if rewriting is required
   */
  return op;  // No rewriting by default
}

// Array of all MatMulQhpiHvx8RowFp32StoreMultithread operations for registration
static QHPI_OpInfo_v1 matmulqhpihvx8rowfp32storemultithread_ops[] = {
    matmulqhpihvx8rowfp32storemultithreadOpInfo,
    matmulqhpihvxofflineq13rhsOpInfo,
    convertfp16todeviceq13OpInfo,
    qwen2decodeattentionpast128fp32OpInfo
};

// Registration function for MatMulQhpiHvx8RowFp32StoreMultithread operations
extern "C" void register_matmulqhpihvx8rowfp32storemultithread_ops()
{
    qhpi_register_ops_v1(sizeof(matmulqhpihvx8rowfp32storemultithread_ops) / sizeof(matmulqhpihvx8rowfp32storemultithread_ops[0]), matmulqhpihvx8rowfp32storemultithread_ops, THIS_PKG_NAME_STR);
}
