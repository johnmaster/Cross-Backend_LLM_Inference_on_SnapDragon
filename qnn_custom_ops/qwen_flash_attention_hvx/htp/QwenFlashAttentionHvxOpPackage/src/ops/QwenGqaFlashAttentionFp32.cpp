//==============================================================================
// Auto Generated Code for QwenGqaFlashAttentionFp32 - QHPI Implementation
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
#define QWEN_FLASH_ATTN_HVX_INTRINSICS 1
#else
#define QWEN_FLASH_ATTN_HVX_INTRINSICS 0
#endif

#ifndef QWEN_FLASH_AV_QFLOAT
#define QWEN_FLASH_AV_QFLOAT 0
#endif

static inline void qwenFlashScaleAccumulator(float *accumulator,
                                             float scale) {
#if QWEN_FLASH_ATTN_HVX_INTRINSICS
  const HVX_Vector scale_vector = q6op_V_vsplat_float32(scale);
  const HVX_Vector accumulator0 = vmemu(accumulator);
  const HVX_Vector accumulator1 = vmemu(accumulator + 32);
  vmemu(accumulator) =
      Q6_Vsf_vmpy_VsfVsf(accumulator0, scale_vector);
  vmemu(accumulator + 32) =
      Q6_Vsf_vmpy_VsfVsf(accumulator1, scale_vector);
#else
  for (uint32_t dim = 0; dim < 64; ++dim) {
    accumulator[dim] *= scale;
  }
#endif
}

static inline void qwenFlashAccumulateValue(float *accumulator,
                                            const float *value,
                                            float weight) {
#if QWEN_FLASH_ATTN_HVX_INTRINSICS
  const HVX_Vector weight_vector = q6op_V_vsplat_float32(weight);
  const HVX_Vector value0 = vmemu(value);
  const HVX_Vector value1 = vmemu(value + 32);
  const HVX_Vector weighted0 = Q6_Vsf_vmpy_VsfVsf(value0, weight_vector);
  const HVX_Vector weighted1 = Q6_Vsf_vmpy_VsfVsf(value1, weight_vector);
  vmemu(accumulator) =
      Q6_Vsf_vadd_VsfVsf(vmemu(accumulator), weighted0);
  vmemu(accumulator + 32) =
      Q6_Vsf_vadd_VsfVsf(vmemu(accumulator + 32), weighted1);
#else
  for (uint32_t dim = 0; dim < 64; ++dim) {
    accumulator[dim] += weight * value[dim];
  }
#endif
}

static inline float qwenFlashDotProduct64(const float *lhs,
                                          const float *rhs) {
#if QWEN_FLASH_ATTN_HVX_INTRINSICS
  // v75 has vector FP32 multiply, but no direct FP32 horizontal reduction.
  // Keep the expensive 64 multiplies in HVX and reduce the two vectors in
  // scalar order. The aligned spill also makes the generated code predictable.
  alignas(128) float products[64];
  vmemu(products) = Q6_Vsf_vmpy_VsfVsf(vmemu(lhs), vmemu(rhs));
  vmemu(products + 32) =
      Q6_Vsf_vmpy_VsfVsf(vmemu(lhs + 32), vmemu(rhs + 32));
  float sum = 0.0f;
  for (uint32_t dim = 0; dim < 64; ++dim) {
    sum += products[dim];
  }
  return sum;
#else
  float sum = 0.0f;
  for (uint32_t dim = 0; dim < 64; ++dim) {
    sum += lhs[dim] * rhs[dim];
  }
  return sum;
#endif
}

// Forward declarations for QwenGqaFlashAttentionFp32 kernel qwengqaflashattentionfp32_float_32_
static uint32_t qwengqaflashattentionfp32_float_32_Execute(QHPI_RuntimeHandle *handle,
                                      uint32_t num_outputs, QHPI_Tensor **outputs,
                                      uint32_t num_inputs, const QHPI_Tensor *const *inputs);
static float qwengqaflashattentionfp32_float_32_CostFunc(const uint32_t num_inputs, const QHPI_Tensor *const *inputs);

// Common forward declarations for QwenGqaFlashAttentionFp32
static const QHPI_Op* qwengqaflashattentionfp32EarlyRewrite(const QHPI_Op *op);
static QHPI_Shape qwengqaflashattentionfp32ShapeRequired(const QHPI_Op *op);
static QHPI_Shape qwengqaflashattentionfp32ShapeLegal(const QHPI_Op *op, const QHPI_Shape* shape);
static const QHPI_Op* qwengqaflashattentionfp32BuildTile(const QHPI_Op *op, const QHPI_Shape* start, const QHPI_Shape* extent);
static const QHPI_Op* qwengqaflashattentionfp32LateRewrite(const QHPI_Op *op);

/*
 * QHPI Registration using hexnn_ffi.h API for QwenGqaFlashAttentionFp32
 * Multiple kernels for different data type combinations
 */


// Input tensor signatures for QwenGqaFlashAttentionFp32 kernel qwengqaflashattentionfp32_float_32_
// Includes both regular inputs and parameters as inputs
static QHPI_Tensor_Signature_v1 qwengqaflashattentionfp32_float_32_InputSignatures[] = {

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

static QHPI_Tensor_Signature_v1 qwengqaflashattentionfp32_float_32_OutputSignatures[] = {

    {
        .element_type = QHPI_Float32,
        .layout = QHPI_Layout_Flat4,
        .storage = QHPI_Storage_Direct,
        .mem_placement = QHPI_MemLoc_DDR_OR_TCM
    }
};

// Kernel definition for QwenGqaFlashAttentionFp32 kernel qwengqaflashattentionfp32_float_32_
static QHPI_Kernel_v1 qwengqaflashattentionfp32_float_32_Kernel = {
    .function_name = "qwengqaflashattentionfp32_float_32_Execute",
    .function = qwengqaflashattentionfp32_float_32_Execute,
    .resources = QHPI_RESOURCE_HVX,
    .source_destructive = false,
    .multithreaded = true,
    .variable_inputs = false,
    .variable_outputs = false,
    .min_inputs = 3,
    .input_signature = qwengqaflashattentionfp32_float_32_InputSignatures,
    .min_outputs = 1,
    .output_signature = qwengqaflashattentionfp32_float_32_OutputSignatures,
    .cost_function = qwengqaflashattentionfp32_float_32_CostFunc,
    .sync_block_size = 0,
    .precomputed_data_size = 0,
    .do_precomputation_function = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate = nullptr
};

// Array of all kernels for QwenGqaFlashAttentionFp32
static QHPI_Kernel_v1 qwengqaflashattentionfp32Kernels[] = {

    qwengqaflashattentionfp32_float_32_Kernel
};

// Operator info for QwenGqaFlashAttentionFp32 - exported for package registration
QHPI_OpInfo_v1 qwengqaflashattentionfp32OpInfo = {
    .name = THIS_PKG_NAME_STR "::" "QwenGqaFlashAttentionFp32",
    .num_kernels = 1,
    .kernels = qwengqaflashattentionfp32Kernels,
    .early_rewrite = qwengqaflashattentionfp32EarlyRewrite,
    .shape_required = qwengqaflashattentionfp32ShapeRequired,
    .shape_legalized = qwengqaflashattentionfp32ShapeLegal,
    .build_tile = qwengqaflashattentionfp32BuildTile,
    .late_rewrite = qwengqaflashattentionfp32LateRewrite
};


/* QHPI execute function implementation for QwenGqaFlashAttentionFp32 kernel qwengqaflashattentionfp32_float_32_ */
static uint32_t qwengqaflashattentionfp32_float_32_Execute(
    QHPI_RuntimeHandle *handle, uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs, const QHPI_Tensor *const *inputs) {
  if (handle == nullptr || num_inputs != 3 || num_outputs != 1 ||
      inputs == nullptr || outputs == nullptr || inputs[0] == nullptr ||
      inputs[1] == nullptr || inputs[2] == nullptr || outputs[0] == nullptr) {
    return QHPI_ErrorFatal;
  }

  const QHPI_Shape q_shape = qhpi_tensor_shape(inputs[0]);
  const QHPI_Shape k_shape = qhpi_tensor_shape(inputs[1]);
  const QHPI_Shape v_shape = qhpi_tensor_shape(inputs[2]);
  const QHPI_Shape o_shape = qhpi_tensor_shape(outputs[0]);
  if (q_shape.rank != 4 || k_shape.rank != 4 || v_shape.rank != 4 ||
      o_shape.rank != 4) {
    return QHPI_ErrorFatal;
  }

  const uint32_t batch = q_shape.dims[0];
  const uint32_t query_heads = q_shape.dims[1];
  const uint32_t query_length = q_shape.dims[2];
  const uint32_t head_dim = q_shape.dims[3];
  constexpr uint32_t kv_heads = 2;
  constexpr uint32_t groups = 7;
  const bool key_head_contiguous =
      k_shape.dims[1] == kv_heads && k_shape.dims[3] == head_dim;
  const bool key_head_interleaved =
      k_shape.dims[2] == head_dim && k_shape.dims[3] == kv_heads;
  const uint32_t kv_length =
      key_head_contiguous ? k_shape.dims[2] : k_shape.dims[1];
  if (batch == 0 || query_heads != kv_heads * groups || query_length == 0 ||
      head_dim != 64 || kv_length == 0 || query_length > kv_length ||
      k_shape.dims[0] != batch ||
      (!key_head_contiguous && !key_head_interleaved) ||
      v_shape.dims[0] != batch ||
      v_shape.dims[1] != kv_heads || v_shape.dims[2] != kv_length ||
      v_shape.dims[3] != head_dim || o_shape.dims[0] != batch ||
      o_shape.dims[1] != query_heads ||
      o_shape.dims[2] != query_length || o_shape.dims[3] != head_dim) {
    return QHPI_ErrorFatal;
  }

  const float *query =
      static_cast<const float *>(qhpi_tensor_raw_data(inputs[0]));
  const float *key =
      static_cast<const float *>(qhpi_tensor_raw_data(inputs[1]));
  const float *value =
      static_cast<const float *>(qhpi_tensor_raw_data(inputs[2]));
  float *output = static_cast<float *>(qhpi_tensor_raw_data(outputs[0]));
  const uint32_t slices = qhpi_num_slices(handle);
  const uint32_t slice = qhpi_slice_number(handle);
  if (query == nullptr || key == nullptr || value == nullptr ||
      output == nullptr || slices == 0 || slice >= slices) {
    return QHPI_ErrorFatal;
  }

  constexpr uint32_t kHeadDim = 64;
  constexpr uint32_t kKvBlock = 32;
  constexpr float kAttentionScale = 0.125f;
  constexpr float kNegativeInfinity = -3.402823466e+38f;
  const uint32_t work_items = batch * query_heads * query_length;

  for (uint32_t work = slice; work < work_items; work += slices) {
    uint32_t index = work;
    const uint32_t query_index = index % query_length;
    index /= query_length;
    const uint32_t query_head = index % query_heads;
    const uint32_t batch_index = index / query_heads;
    const uint32_t kv_head = query_head / groups;

    const uint64_t q_offset =
        ((static_cast<uint64_t>(batch_index) * query_heads + query_head) *
             query_length +
         query_index) *
        kHeadDim;
    const uint64_t kv_head_offset =
        (static_cast<uint64_t>(batch_index) * kv_heads + kv_head) *
        kv_length * kHeadDim;
    const float *q = query + q_offset;

    // Decode queries occupy the last query_length positions in the supplied
    // K/V tensor. This also gives the standard causal boundary when
    // query_length == kv_length for prefill fixtures.
    const uint32_t visible_tokens =
        kv_length - query_length + query_index + 1;
    float running_max = kNegativeInfinity;
    float running_sum = 0.0f;
#if QWEN_FLASH_ATTN_HVX_INTRINSICS
    // Keep both accumulator vectors in registers across all KV tokens.
    HVX_Vector accumulator0 = Q6_V_vzero();
    HVX_Vector accumulator1 = Q6_V_vzero();
#else
    float accumulator[kHeadDim] = {};
#endif

    for (uint32_t block_begin = 0; block_begin < visible_tokens;
         block_begin += kKvBlock) {
      const uint32_t remaining = visible_tokens - block_begin;
      const uint32_t block_tokens =
          remaining < kKvBlock ? remaining : kKvBlock;
      float scores[kKvBlock];
      float block_max = kNegativeInfinity;

      for (uint32_t token_in_block = 0; token_in_block < block_tokens;
           ++token_in_block) {
        const uint32_t token = block_begin + token_in_block;
        float score = 0.0f;
        if (key_head_contiguous) {
          const uint64_t k_offset =
              ((static_cast<uint64_t>(batch_index) * kv_heads + kv_head) *
                   kv_length +
               token) *
              kHeadDim;
          score = qwenFlashDotProduct64(q, key + k_offset);
        } else {
          const uint64_t k_token_offset =
              (static_cast<uint64_t>(batch_index) * kv_length + token) *
              kHeadDim * kv_heads;
          for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
            score += q[dim] *
                     key[k_token_offset + dim * kv_heads + kv_head];
          }
        }
        score *= kAttentionScale;
        scores[token_in_block] = score;
        block_max = score > block_max ? score : block_max;
      }

      const float next_max =
          block_max > running_max ? block_max : running_max;
      const float old_scale = std::exp(running_max - next_max);
      running_sum *= old_scale;
#if QWEN_FLASH_ATTN_HVX_INTRINSICS
      const HVX_Vector old_scale_sf = q6op_V_vsplat_float32(old_scale);
#if QWEN_FLASH_AV_QFLOAT
      const HVX_Vector old_scale_qf =
          Q6_Vqf32_vadd_VsfVsf(old_scale_sf, Q6_V_vzero());
      accumulator0 =
          Q6_Vqf32_vmpy_Vqf32Vqf32(accumulator0, old_scale_qf);
      accumulator1 =
          Q6_Vqf32_vmpy_Vqf32Vqf32(accumulator1, old_scale_qf);
#else
      accumulator0 =
          Q6_Vsf_vmpy_VsfVsf(accumulator0, old_scale_sf);
      accumulator1 =
          Q6_Vsf_vmpy_VsfVsf(accumulator1, old_scale_sf);
#endif
#else
      qwenFlashScaleAccumulator(accumulator, old_scale);
#endif

      for (uint32_t token_in_block = 0; token_in_block < block_tokens;
           ++token_in_block) {
        const uint32_t token = block_begin + token_in_block;
        const float weight = std::exp(scores[token_in_block] - next_max);
        const float *v_token =
            value + kv_head_offset + static_cast<uint64_t>(token) * kHeadDim;
        running_sum += weight;
#if QWEN_FLASH_ATTN_HVX_INTRINSICS
        const HVX_Vector weight_vector = q6op_V_vsplat_float32(weight);
#if QWEN_FLASH_AV_QFLOAT
        accumulator0 = Q6_Vqf32_vadd_Vqf32Vqf32(
            accumulator0,
            Q6_Vqf32_vmpy_VsfVsf(vmemu(v_token), weight_vector));
        accumulator1 = Q6_Vqf32_vadd_Vqf32Vqf32(
            accumulator1,
            Q6_Vqf32_vmpy_VsfVsf(vmemu(v_token + 32), weight_vector));
#else
        accumulator0 = Q6_Vsf_vadd_VsfVsf(
            accumulator0,
            Q6_Vsf_vmpy_VsfVsf(vmemu(v_token), weight_vector));
        accumulator1 = Q6_Vsf_vadd_VsfVsf(
            accumulator1,
            Q6_Vsf_vmpy_VsfVsf(vmemu(v_token + 32), weight_vector));
#endif
#else
        qwenFlashAccumulateValue(accumulator, v_token, weight);
#endif
      }
      running_max = next_max;
    }

    const float inverse_sum = 1.0f / running_sum;
    float *out = output + q_offset;
#if QWEN_FLASH_ATTN_HVX_INTRINSICS
    const HVX_Vector inverse_sum_sf = q6op_V_vsplat_float32(inverse_sum);
#if QWEN_FLASH_AV_QFLOAT
    const HVX_Vector inverse_sum_qf =
        Q6_Vqf32_vadd_VsfVsf(inverse_sum_sf, Q6_V_vzero());
    vmemu(out) = Q6_Vsf_equals_Vqf32(
        Q6_Vqf32_vmpy_Vqf32Vqf32(accumulator0, inverse_sum_qf));
    vmemu(out + 32) = Q6_Vsf_equals_Vqf32(
        Q6_Vqf32_vmpy_Vqf32Vqf32(accumulator1, inverse_sum_qf));
#else
    vmemu(out) = Q6_Vsf_vmpy_VsfVsf(accumulator0, inverse_sum_sf);
    vmemu(out + 32) =
        Q6_Vsf_vmpy_VsfVsf(accumulator1, inverse_sum_sf);
#endif
#else
    qwenFlashScaleAccumulator(accumulator, inverse_sum);
    for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
      out[dim] = accumulator[dim];
    }
#endif
  }
  return QHPI_Success;
}

static float qwengqaflashattentionfp32_float_32_CostFunc(const uint32_t num_inputs, const QHPI_Tensor *const *inputs)
{
  /*
   * Cost estimation function for QwenGqaFlashAttentionFp32 kernel qwengqaflashattentionfp32_float_32_
   * Return approximate number of cycles needed for this operation
   * with the specific data type combination. Used for estimating cycle
   * performance of a graph.
   *
   * Parameters:
   * - num_inputs: Number of input tensors
   * - inputs: Array of input tensor pointers
   */

  if (num_inputs != 3 || inputs == nullptr || inputs[0] == nullptr ||
      inputs[1] == nullptr) {
    return 1.0e30f;
  }
  const QHPI_Shape q_shape = qhpi_tensor_shape(inputs[0]);
  const QHPI_Shape k_shape = qhpi_tensor_shape(inputs[1]);
  if (q_shape.rank != 4 || k_shape.rank != 4) {
    return 1.0e30f;
  }
  return static_cast<float>(q_shape.dims[0]) * q_shape.dims[1] *
         q_shape.dims[2] * k_shape.dims[1] * q_shape.dims[3] * 2.0f;
}

/*
 * Common stub implementations for QwenGqaFlashAttentionFp32 QHPI_OpInfo functions
 * These are shared across all kernels and provide default no-op implementations
 */

static const QHPI_Op* qwengqaflashattentionfp32EarlyRewrite(const QHPI_Op *op)
{
  /*
   * Early rewrite function for QwenGqaFlashAttentionFp32
   * Called during graph optimization phase
   * Return the original op if no rewriting is needed, or a new op if rewriting is required
   */
  return op;  // No rewriting by default
}

static QHPI_Shape qwengqaflashattentionfp32ShapeRequired(const QHPI_Op *op)
{
  /*
   * Shape required function for QwenGqaFlashAttentionFp32
   * Specifies required input shapes for the operation
   * Return empty shape if no specific shape requirements
   */
  QHPI_Shape empty_shape = {0};  // Empty shape by default
  return empty_shape;
}

static QHPI_Shape qwengqaflashattentionfp32ShapeLegal(const QHPI_Op *op, const QHPI_Shape* shape)
{
  /*
   * Shape legal function for QwenGqaFlashAttentionFp32
   * Validates if a given shape is legal for this operation
   * Return the shape if legal, or modified shape if not legal
   */
  return *shape;  // Accept the provided shape by default
}

static const QHPI_Op* qwengqaflashattentionfp32BuildTile(const QHPI_Op *op, const QHPI_Shape* start, const QHPI_Shape* extent)
{
  /*
   * Build tile function for QwenGqaFlashAttentionFp32
   * Creates a tiled version of the operation
   * Return a new op that operates on the specified tile, or op if tiling is not supported
   */
  return op;  // No tiling support by default
}

static const QHPI_Op* qwengqaflashattentionfp32LateRewrite(const QHPI_Op *op)
{
  /*
   * Late rewrite function for QwenGqaFlashAttentionFp32
   * Called during late optimization phase
   * Return the original op if no rewriting is needed, or a new op if rewriting is required
   */
  return op;  // No rewriting by default
}

// Array of all QwenGqaFlashAttentionFp32 operations for registration
static QHPI_OpInfo_v1 qwengqaflashattentionfp32_ops[] = {
    qwengqaflashattentionfp32OpInfo
};

// Registration function for QwenGqaFlashAttentionFp32 operations
extern "C" void register_qwengqaflashattentionfp32_ops()
{
    qhpi_register_ops_v1(sizeof(qwengqaflashattentionfp32_ops) / sizeof(qwengqaflashattentionfp32_ops[0]), qwengqaflashattentionfp32_ops, THIS_PKG_NAME_STR);
}
