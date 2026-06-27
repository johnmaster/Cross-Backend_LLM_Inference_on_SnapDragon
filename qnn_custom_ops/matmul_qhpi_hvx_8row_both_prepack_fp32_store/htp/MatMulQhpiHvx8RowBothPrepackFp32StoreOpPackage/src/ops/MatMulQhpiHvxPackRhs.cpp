#include "HTP/core/constraints.h"
#include "HTP/core/qhpi.h"

#include <cstdint>

#if defined(__hexagon__)
#include "HTP/core/intrinsics.h"
#define MATMUL_QHPI_HVX_PACK_RHS_INTRINSICS 1
#endif

static inline int16_t matmulqhpihvxpackrhsFloatToQ13(float value) {
  constexpr float scale = 8192.0f;
  float scaled = value * scale;
  scaled += scaled >= 0.0f ? 0.5f : -0.5f;
  int32_t q = static_cast<int32_t>(scaled);
  if (q > 32767) q = 32767;
  if (q < -32768) q = -32768;
  return static_cast<int16_t>(q);
}

static uint32_t matmulqhpihvxpackrhs_float_16_Execute(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs,
    QHPI_Tensor **outputs,
    uint32_t num_inputs,
    const QHPI_Tensor *const *inputs) {
  (void)handle;
  if (num_inputs != 1 || num_outputs != 1 || inputs == nullptr ||
      outputs == nullptr || inputs[0] == nullptr || outputs[0] == nullptr) {
    return QHPI_ErrorFatal;
  }

  const QHPI_Shape input_shape = qhpi_tensor_shape(inputs[0]);
  const QHPI_Shape output_shape = qhpi_tensor_shape(outputs[0]);
  if (qhpi_element_type_size(qhpi_tensor_type(outputs[0])) !=
          sizeof(int16_t) ||
      input_shape.rank != output_shape.rank) {
    return QHPI_ErrorFatal;
  }

  uint64_t elements = 1;
  for (uint32_t dimension = 0; dimension < input_shape.rank; ++dimension) {
    if (input_shape.dims[dimension] != output_shape.dims[dimension]) {
      return QHPI_ErrorFatal;
    }
    elements *= input_shape.dims[dimension];
  }

  const __fp16 *input =
      static_cast<const __fp16 *>(qhpi_tensor_raw_data(inputs[0]));
  int16_t *output =
      static_cast<int16_t *>(qhpi_tensor_raw_data(outputs[0]));
  if (input == nullptr || output == nullptr) {
    return QHPI_ErrorFatal;
  }

  uint64_t index = 0;
#if MATMUL_QHPI_HVX_PACK_RHS_INTRINSICS
  for (; index + 64 <= elements; index += 64) {
    vmemu(&output[index]) =
        hnnx::s16_from_hf_rnd_sat<13>(vmemu(&input[index]));
  }
#endif
  for (; index < elements; ++index) {
    output[index] =
        matmulqhpihvxpackrhsFloatToQ13(static_cast<float>(input[index]));
  }
  return QHPI_Success;
}

static float matmulqhpihvxpackrhs_float_16_CostFunc(
    const uint32_t num_inputs,
    const QHPI_Tensor *const *inputs) {
  (void)num_inputs;
  (void)inputs;
  return 1000.0f;
}

static QHPI_Tensor_Signature_v1
    matmulqhpihvxpackrhs_float_16_InputSignatures[] = {{
        .element_type = QHPI_Float16,
        .layout = QHPI_Layout_Flat4,
        .storage = QHPI_Storage_Direct,
        .mem_placement = QHPI_MemLoc_DDR_OR_TCM,
    }};

// Float16 is a package-private 16-bit carrier for raw Q13 bits.
static QHPI_Tensor_Signature_v1
    matmulqhpihvxpackrhs_float_16_OutputSignatures[] = {{
        .element_type = QHPI_Float16,
        .layout = QHPI_Layout_Flat4,
        .storage = QHPI_Storage_Direct,
        .mem_placement = QHPI_MemLoc_DDR_OR_TCM,
    }};

static QHPI_Kernel_v1 matmulqhpihvxpackrhs_float_16_Kernel = {
    .function_name = "matmulqhpihvxpackrhs_float_16_Execute",
    .function = matmulqhpihvxpackrhs_float_16_Execute,
    .resources = QHPI_RESOURCE_HVX,
    .source_destructive = false,
    .multithreaded = false,
    .variable_inputs = false,
    .variable_outputs = false,
    .min_inputs = 1,
    .input_signature = matmulqhpihvxpackrhs_float_16_InputSignatures,
    .min_outputs = 1,
    .output_signature = matmulqhpihvxpackrhs_float_16_OutputSignatures,
    .cost_function = matmulqhpihvxpackrhs_float_16_CostFunc,
    .sync_block_size = 0,
    .precomputed_data_size = 0,
    .do_precomputation_function = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate = nullptr,
};

static QHPI_Kernel_v1 matmulqhpihvxpackrhsKernels[] = {
    matmulqhpihvxpackrhs_float_16_Kernel,
};

QHPI_OpInfo_v1 matmulqhpihvxpackrhsOpInfo = {
    .name = THIS_PKG_NAME_STR "::MatMulQhpiHvxPackRhs",
    .num_kernels = 1,
    .kernels = matmulqhpihvxpackrhsKernels,
    .early_rewrite = nullptr,
    .shape_required = nullptr,
    .shape_legalized = nullptr,
    .build_tile = nullptr,
    .late_rewrite = nullptr,
};
