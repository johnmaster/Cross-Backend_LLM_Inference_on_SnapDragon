//==============================================================================
// Auto Generated Code for MatMulQhpiHvx8RowVectorConvert - QHPI Implementation
// Multiple kernels generated for different data type combinations
//==============================================================================

#include "HTP/core/constraints.h"
#include <cstdint>
#include <string>

// Plugin/QHPI includes - using correct header from hexnn_qhpi.h
#include "HTP/core/qhpi.h"

#if defined(__hexagon__)
#include "HTP/core/intrinsics.h"
#define MATMUL_QHPI_HVX_8ROW_VECTOR_CONVERT_INTRINSICS 1
#endif

// Forward declarations for MatMulQhpiHvx8RowVectorConvert kernel matmulqhpihvx8rowvectorconvert_float_16_
static uint32_t matmulqhpihvx8rowvectorconvert_float_16_Execute(QHPI_RuntimeHandle *handle,
                                      uint32_t num_outputs, QHPI_Tensor **outputs,
                                      uint32_t num_inputs, const QHPI_Tensor *const *inputs);
static float matmulqhpihvx8rowvectorconvert_float_16_CostFunc(const uint32_t num_inputs, const QHPI_Tensor *const *inputs);

// Common forward declarations for MatMulQhpiHvx8RowVectorConvert
static const QHPI_Op* matmulqhpihvx8rowvectorconvertEarlyRewrite(const QHPI_Op *op);
static QHPI_Shape matmulqhpihvx8rowvectorconvertShapeRequired(const QHPI_Op *op);
static QHPI_Shape matmulqhpihvx8rowvectorconvertShapeLegal(const QHPI_Op *op, const QHPI_Shape* shape);
static const QHPI_Op* matmulqhpihvx8rowvectorconvertBuildTile(const QHPI_Op *op, const QHPI_Shape* start, const QHPI_Shape* extent);
static const QHPI_Op* matmulqhpihvx8rowvectorconvertLateRewrite(const QHPI_Op *op);

static inline float matmulqhpihvx8rowvectorconvertScalarDot(const __fp16 *lhs,
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

static inline int16_t matmulqhpihvx8rowvectorconvertFloatToQ13(float value) {
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

#if MATMUL_QHPI_HVX_8ROW_VECTOR_CONVERT_INTRINSICS
static inline HVX_Vector
matmulqhpihvx8rowvectorconvertConvert64Fp16ToQ13(
    const __fp16 *values) {
  const HVX_Vector fp16_values = vmemu(values);

  // This uses QAIRT's qf32 intermediate path and avoids the direct half-float
  // vector multiply, which returns zero on the target device.
  return hnnx::s16_from_hf_rnd_sat<13>(fp16_values);
}

static inline void matmulqhpihvx8rowvectorconvertCompute1x64Hvx(const __fp16 *lhs,
                                                       const __fp16 *rhs,
                                                       __fp16 *output,
                                                       uint64_t lhs_base,
                                                       uint64_t rhs_base,
                                                       uint64_t output_base,
                                                       uint32_t k,
                                                       uint32_t n,
                                                       uint32_t col) {
#if defined(MATMUL_QHPI_HVX_8ROW_VECTOR_CONVERT_DEBUG_STORE_RHS)
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
        matmulqhpihvx8rowvectorconvertFloatToQ13(static_cast<float>(lhs[lhs_base + reduction]));
    const HVX_Vector lhs_vec = Q6_Vh_vsplat_R(static_cast<int32_t>(lhs_q13));

    const uint64_t rhs_row_base = rhs_base + (uint64_t)reduction * n + col;
    const HVX_Vector rhs_vec =
        matmulqhpihvx8rowvectorconvertConvert64Fp16ToQ13(
            &rhs[rhs_row_base]);

    acc = Q6_Ww_vmpyacc_WwVhVh(acc, lhs_vec, rhs_vec);
  }

  alignas(128) int32_t acc_lo_store[32];
  alignas(128) int32_t acc_hi_store[32];
  vmemu(acc_lo_store) = Q6_V_lo_W(acc);
  vmemu(acc_hi_store) = Q6_V_hi_W(acc);

  constexpr float inv_scale = 1.0f / (8192.0f * 8192.0f);
  for (uint32_t lane = 0; lane < 32; ++lane) {
    output[output_base + col + 2 * lane] =
        static_cast<__fp16>(static_cast<float>(acc_lo_store[lane]) * inv_scale);
    output[output_base + col + 2 * lane + 1] =
        static_cast<__fp16>(static_cast<float>(acc_hi_store[lane]) * inv_scale);
  }
#endif
}

static inline void matmulqhpihvx8rowvectorconvertCompute8x64Hvx(
    const __fp16 *lhs,
    const __fp16 *rhs,
    __fp16 *output,
    uint64_t lhs_base,
    uint64_t rhs_base,
    uint64_t output_base,
    uint32_t k,
    uint32_t n,
    uint32_t col) {
#if defined(MATMUL_QHPI_HVX_8ROW_VECTOR_CONVERT_DEBUG_STORE_RHS)
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
        matmulqhpihvx8rowvectorconvertConvert64Fp16ToQ13(
            &rhs[rhs_row_base]);

    const HVX_Vector lhs0 = Q6_Vh_vsplat_R(
        matmulqhpihvx8rowvectorconvertFloatToQ13(
            static_cast<float>(lhs[lhs_base + reduction])));
    const HVX_Vector lhs1 = Q6_Vh_vsplat_R(
        matmulqhpihvx8rowvectorconvertFloatToQ13(
            static_cast<float>(lhs[lhs_base + k + reduction])));
    const HVX_Vector lhs2 = Q6_Vh_vsplat_R(
        matmulqhpihvx8rowvectorconvertFloatToQ13(
            static_cast<float>(lhs[lhs_base + (uint64_t)2 * k + reduction])));
    const HVX_Vector lhs3 = Q6_Vh_vsplat_R(
        matmulqhpihvx8rowvectorconvertFloatToQ13(
            static_cast<float>(lhs[lhs_base + (uint64_t)3 * k + reduction])));
    const HVX_Vector lhs4 = Q6_Vh_vsplat_R(
        matmulqhpihvx8rowvectorconvertFloatToQ13(
            static_cast<float>(lhs[lhs_base + (uint64_t)4 * k + reduction])));
    const HVX_Vector lhs5 = Q6_Vh_vsplat_R(
        matmulqhpihvx8rowvectorconvertFloatToQ13(
            static_cast<float>(lhs[lhs_base + (uint64_t)5 * k + reduction])));
    const HVX_Vector lhs6 = Q6_Vh_vsplat_R(
        matmulqhpihvx8rowvectorconvertFloatToQ13(
            static_cast<float>(lhs[lhs_base + (uint64_t)6 * k + reduction])));
    const HVX_Vector lhs7 = Q6_Vh_vsplat_R(
        matmulqhpihvx8rowvectorconvertFloatToQ13(
            static_cast<float>(lhs[lhs_base + (uint64_t)7 * k + reduction])));

    acc0 = Q6_Ww_vmpyacc_WwVhVh(acc0, lhs0, rhs_vec);
    acc1 = Q6_Ww_vmpyacc_WwVhVh(acc1, lhs1, rhs_vec);
    acc2 = Q6_Ww_vmpyacc_WwVhVh(acc2, lhs2, rhs_vec);
    acc3 = Q6_Ww_vmpyacc_WwVhVh(acc3, lhs3, rhs_vec);
    acc4 = Q6_Ww_vmpyacc_WwVhVh(acc4, lhs4, rhs_vec);
    acc5 = Q6_Ww_vmpyacc_WwVhVh(acc5, lhs5, rhs_vec);
    acc6 = Q6_Ww_vmpyacc_WwVhVh(acc6, lhs6, rhs_vec);
    acc7 = Q6_Ww_vmpyacc_WwVhVh(acc7, lhs7, rhs_vec);
  }

  alignas(128) int32_t acc_lo_store[8][32];
  alignas(128) int32_t acc_hi_store[8][32];
  vmemu(acc_lo_store[0]) = Q6_V_lo_W(acc0);
  vmemu(acc_hi_store[0]) = Q6_V_hi_W(acc0);
  vmemu(acc_lo_store[1]) = Q6_V_lo_W(acc1);
  vmemu(acc_hi_store[1]) = Q6_V_hi_W(acc1);
  vmemu(acc_lo_store[2]) = Q6_V_lo_W(acc2);
  vmemu(acc_hi_store[2]) = Q6_V_hi_W(acc2);
  vmemu(acc_lo_store[3]) = Q6_V_lo_W(acc3);
  vmemu(acc_hi_store[3]) = Q6_V_hi_W(acc3);
  vmemu(acc_lo_store[4]) = Q6_V_lo_W(acc4);
  vmemu(acc_hi_store[4]) = Q6_V_hi_W(acc4);
  vmemu(acc_lo_store[5]) = Q6_V_lo_W(acc5);
  vmemu(acc_hi_store[5]) = Q6_V_hi_W(acc5);
  vmemu(acc_lo_store[6]) = Q6_V_lo_W(acc6);
  vmemu(acc_hi_store[6]) = Q6_V_hi_W(acc6);
  vmemu(acc_lo_store[7]) = Q6_V_lo_W(acc7);
  vmemu(acc_hi_store[7]) = Q6_V_hi_W(acc7);

  constexpr float inv_scale = 1.0f / (8192.0f * 8192.0f);
  for (uint32_t row = 0; row < 8; ++row) {
    const uint64_t row_output_base = output_base + (uint64_t)row * n + col;
    for (uint32_t lane = 0; lane < 32; ++lane) {
      output[row_output_base + 2 * lane] = static_cast<__fp16>(
          static_cast<float>(acc_lo_store[row][lane]) * inv_scale);
      output[row_output_base + 2 * lane + 1] = static_cast<__fp16>(
          static_cast<float>(acc_hi_store[row][lane]) * inv_scale);
    }
  }
#endif
}
#endif

/*
 * QHPI Registration using hexnn_ffi.h API for MatMulQhpiHvx8RowVectorConvert
 * Multiple kernels for different data type combinations
 */


// Input tensor signatures for MatMulQhpiHvx8RowVectorConvert kernel matmulqhpihvx8rowvectorconvert_float_16_
// Includes both regular inputs and parameters as inputs
static QHPI_Tensor_Signature_v1 matmulqhpihvx8rowvectorconvert_float_16_InputSignatures[] = {

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

static QHPI_Tensor_Signature_v1 matmulqhpihvx8rowvectorconvert_float_16_OutputSignatures[] = {

    {
        .element_type = QHPI_Float16,
        .layout = QHPI_Layout_Flat4,
        .storage = QHPI_Storage_Direct,
        .mem_placement = QHPI_MemLoc_DDR_OR_TCM
    }
};

// Kernel definition for MatMulQhpiHvx8RowVectorConvert kernel matmulqhpihvx8rowvectorconvert_float_16_
static QHPI_Kernel_v1 matmulqhpihvx8rowvectorconvert_float_16_Kernel = {
    .function_name = "matmulqhpihvx8rowvectorconvert_float_16_Execute",
    .function = matmulqhpihvx8rowvectorconvert_float_16_Execute,
    .resources = QHPI_RESOURCE_HVX,
    .source_destructive = false,
    .multithreaded = false,
    .variable_inputs = false,
    .variable_outputs = false,
    .min_inputs = 2,
    .input_signature = matmulqhpihvx8rowvectorconvert_float_16_InputSignatures,
    .min_outputs = 1,
    .output_signature = matmulqhpihvx8rowvectorconvert_float_16_OutputSignatures,
    .cost_function = matmulqhpihvx8rowvectorconvert_float_16_CostFunc,
    .sync_block_size = 0,
    .precomputed_data_size = 0,
    .do_precomputation_function = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate = nullptr
};

// Array of all kernels for MatMulQhpiHvx8RowVectorConvert
static QHPI_Kernel_v1 matmulqhpihvx8rowvectorconvertKernels[] = {

    matmulqhpihvx8rowvectorconvert_float_16_Kernel
};

// Operator info for MatMulQhpiHvx8RowVectorConvert - exported for package registration
QHPI_OpInfo_v1 matmulqhpihvx8rowvectorconvertOpInfo = {
    .name = THIS_PKG_NAME_STR "::" "MatMulQhpiHvx8RowVectorConvert",
    .num_kernels = 1,
    .kernels = matmulqhpihvx8rowvectorconvertKernels,
    .early_rewrite = matmulqhpihvx8rowvectorconvertEarlyRewrite,
    .shape_required = matmulqhpihvx8rowvectorconvertShapeRequired,
    .shape_legalized = matmulqhpihvx8rowvectorconvertShapeLegal,
    .build_tile = matmulqhpihvx8rowvectorconvertBuildTile,
    .late_rewrite = matmulqhpihvx8rowvectorconvertLateRewrite
};


/* QHPI execute function implementation for MatMulQhpiHvx8RowVectorConvert kernel matmulqhpihvx8rowvectorconvert_float_16_ */
static uint32_t matmulqhpihvx8rowvectorconvert_float_16_Execute(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs,
    QHPI_Tensor **outputs,
    uint32_t num_inputs,
    const QHPI_Tensor *const *inputs) {
  (void)handle;

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

  __fp16 *output =
      static_cast<__fp16 *>(qhpi_tensor_raw_data(outputs[0]));

  if (lhs == nullptr || rhs == nullptr || output == nullptr) {
    return QHPI_ErrorFatal;
  }

  // An 8x64 tile reuses each converted RHS vector across eight output rows.
  // Incomplete row and column tiles retain the baseline paths.
  for (uint32_t b = 0; b < batch; ++b) {
    for (uint32_t h = 0; h < heads; ++h) {
      const uint64_t rhs_base =
          ((uint64_t)b * heads + h) * k * n;
      uint32_t row = 0;
#if MATMUL_QHPI_HVX_8ROW_VECTOR_CONVERT_INTRINSICS && \
    !defined(MATMUL_QHPI_HVX_8ROW_VECTOR_CONVERT_FORCE_SCALAR)
      for (; row + 8 <= m; row += 8) {
        const uint64_t lhs_base =
            (((uint64_t)b * heads + h) * m + row) * k;
        const uint64_t output_base =
            (((uint64_t)b * heads + h) * m + row) * n;

        uint32_t col = 0;
        for (; col + 64 <= n; col += 64) {
          matmulqhpihvx8rowvectorconvertCompute8x64Hvx(
              lhs, rhs, output, lhs_base, rhs_base, output_base, k, n, col);
        }

        for (; col < n; ++col) {
          for (uint32_t tile_row = 0; tile_row < 8; ++tile_row) {
            const uint64_t tile_lhs_base =
                lhs_base + (uint64_t)tile_row * k;
            const uint64_t tile_output_base =
                output_base + (uint64_t)tile_row * n;
            output[tile_output_base + col] =
                static_cast<__fp16>(matmulqhpihvx8rowvectorconvertScalarDot(
                    lhs, rhs, tile_lhs_base, rhs_base, k, n, col));
          }
        }
      }
#endif

      for (; row < m; ++row) {
        const uint64_t lhs_base =
            (((uint64_t)b * heads + h) * m + row) * k;
        const uint64_t output_base =
            (((uint64_t)b * heads + h) * m + row) * n;
        uint32_t col = 0;
#if MATMUL_QHPI_HVX_8ROW_VECTOR_CONVERT_INTRINSICS && \
    !defined(MATMUL_QHPI_HVX_8ROW_VECTOR_CONVERT_FORCE_SCALAR)
        for (; col + 64 <= n; col += 64) {
          matmulqhpihvx8rowvectorconvertCompute1x64Hvx(
              lhs, rhs, output, lhs_base, rhs_base, output_base, k, n, col);
        }
#endif
        for (; col < n; ++col) {
          output[output_base + col] =
              static_cast<__fp16>(matmulqhpihvx8rowvectorconvertScalarDot(
                  lhs, rhs, lhs_base, rhs_base, k, n, col));
        }
      }
    }
  }

  return QHPI_Success;
}

static float matmulqhpihvx8rowvectorconvert_float_16_CostFunc(const uint32_t num_inputs, const QHPI_Tensor *const *inputs)
{
  /*
   * Cost estimation function for MatMulQhpiHvx8RowVectorConvert kernel matmulqhpihvx8rowvectorconvert_float_16_
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
 * Common stub implementations for MatMulQhpiHvx8RowVectorConvert QHPI_OpInfo functions
 * These are shared across all kernels and provide default no-op implementations
 */

static const QHPI_Op* matmulqhpihvx8rowvectorconvertEarlyRewrite(const QHPI_Op *op)
{
  /*
   * Early rewrite function for MatMulQhpiHvx8RowVectorConvert
   * Called during graph optimization phase
   * Return the original op if no rewriting is needed, or a new op if rewriting is required
   */
  return op;  // No rewriting by default
}

static QHPI_Shape matmulqhpihvx8rowvectorconvertShapeRequired(const QHPI_Op *op)
{
  /*
   * Shape required function for MatMulQhpiHvx8RowVectorConvert
   * Specifies required input shapes for the operation
   * Return empty shape if no specific shape requirements
   */
  QHPI_Shape empty_shape = {0};  // Empty shape by default
  return empty_shape;
}

static QHPI_Shape matmulqhpihvx8rowvectorconvertShapeLegal(const QHPI_Op *op, const QHPI_Shape* shape)
{
  /*
   * Shape legal function for MatMulQhpiHvx8RowVectorConvert
   * Validates if a given shape is legal for this operation
   * Return the shape if legal, or modified shape if not legal
   */
  return *shape;  // Accept the provided shape by default
}

static const QHPI_Op* matmulqhpihvx8rowvectorconvertBuildTile(const QHPI_Op *op, const QHPI_Shape* start, const QHPI_Shape* extent)
{
  /*
   * Build tile function for MatMulQhpiHvx8RowVectorConvert
   * Creates a tiled version of the operation
   * Return a new op that operates on the specified tile, or op if tiling is not supported
   */
  return op;  // No tiling support by default
}

static const QHPI_Op* matmulqhpihvx8rowvectorconvertLateRewrite(const QHPI_Op *op)
{
  /*
   * Late rewrite function for MatMulQhpiHvx8RowVectorConvert
   * Called during late optimization phase
   * Return the original op if no rewriting is needed, or a new op if rewriting is required
   */
  return op;  // No rewriting by default
}

// Array of all MatMulQhpiHvx8RowVectorConvert operations for registration
static QHPI_OpInfo_v1 matmulqhpihvx8rowvectorconvert_ops[] = {
    matmulqhpihvx8rowvectorconvertOpInfo
};

// Registration function for MatMulQhpiHvx8RowVectorConvert operations
extern "C" void register_matmulqhpihvx8rowvectorconvert_ops()
{
    qhpi_register_ops_v1(sizeof(matmulqhpihvx8rowvectorconvert_ops) / sizeof(matmulqhpihvx8rowvectorconvert_ops[0]), matmulqhpihvx8rowvectorconvert_ops, THIS_PKG_NAME_STR);
}
