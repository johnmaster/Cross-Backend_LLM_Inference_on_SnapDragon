//==============================================================================
// Auto Generated Code for MatMulQhpiFp16Scalar - QHPI Implementation
// Multiple kernels generated for different data type combinations
//==============================================================================

#include "HTP/core/constraints.h"
#include <string>

// Plugin/QHPI includes - using correct header from hexnn_qhpi.h
#include "HTP/core/qhpi.h"

// Forward declarations for MatMulQhpiFp16Scalar kernel matmulqhpifp16scalar_float_16_
static uint32_t matmulqhpifp16scalar_float_16_Execute(QHPI_RuntimeHandle *handle,
                                      uint32_t num_outputs, QHPI_Tensor **outputs,
                                      uint32_t num_inputs, const QHPI_Tensor *const *inputs);
static float matmulqhpifp16scalar_float_16_CostFunc(const uint32_t num_inputs, const QHPI_Tensor *const *inputs);

// Common forward declarations for MatMulQhpiFp16Scalar
static const QHPI_Op* matmulqhpifp16scalarEarlyRewrite(const QHPI_Op *op);
static QHPI_Shape matmulqhpifp16scalarShapeRequired(const QHPI_Op *op);
static QHPI_Shape matmulqhpifp16scalarShapeLegal(const QHPI_Op *op, const QHPI_Shape* shape);
static const QHPI_Op* matmulqhpifp16scalarBuildTile(const QHPI_Op *op, const QHPI_Shape* start, const QHPI_Shape* extent);
static const QHPI_Op* matmulqhpifp16scalarLateRewrite(const QHPI_Op *op);

/*
 * QHPI Registration using hexnn_ffi.h API for MatMulQhpiFp16Scalar
 * Multiple kernels for different data type combinations
 */


// Input tensor signatures for MatMulQhpiFp16Scalar kernel matmulqhpifp16scalar_float_16_
// Includes both regular inputs and parameters as inputs
static QHPI_Tensor_Signature_v1 matmulqhpifp16scalar_float_16_InputSignatures[] = {

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

static QHPI_Tensor_Signature_v1 matmulqhpifp16scalar_float_16_OutputSignatures[] = {

    {
        .element_type = QHPI_Float16,
        .layout = QHPI_Layout_Flat4,
        .storage = QHPI_Storage_Direct,
        .mem_placement = QHPI_MemLoc_DDR_OR_TCM
    }
};

// Kernel definition for MatMulQhpiFp16Scalar kernel matmulqhpifp16scalar_float_16_
static QHPI_Kernel_v1 matmulqhpifp16scalar_float_16_Kernel = {
    .function_name = "matmulqhpifp16scalar_float_16_Execute",
    .function = matmulqhpifp16scalar_float_16_Execute,
    .resources = QHPI_RESOURCE_MAIN,
    .source_destructive = false,
    .multithreaded = false,
    .variable_inputs = false,
    .variable_outputs = false,
    .min_inputs = 2,
    .input_signature = matmulqhpifp16scalar_float_16_InputSignatures,
    .min_outputs = 1,
    .output_signature = matmulqhpifp16scalar_float_16_OutputSignatures,
    .cost_function = matmulqhpifp16scalar_float_16_CostFunc,
    .sync_block_size = 0,
    .precomputed_data_size = 0,
    .do_precomputation_function = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate = nullptr
};

// Array of all kernels for MatMulQhpiFp16Scalar
static QHPI_Kernel_v1 matmulqhpifp16scalarKernels[] = {

    matmulqhpifp16scalar_float_16_Kernel
};

// Operator info for MatMulQhpiFp16Scalar - exported for package registration
QHPI_OpInfo_v1 matmulqhpifp16scalarOpInfo = {
    .name = THIS_PKG_NAME_STR "::" "MatMulQhpiFp16Scalar",
    .num_kernels = 1,
    .kernels = matmulqhpifp16scalarKernels,
    .early_rewrite = matmulqhpifp16scalarEarlyRewrite,
    .shape_required = matmulqhpifp16scalarShapeRequired,
    .shape_legalized = matmulqhpifp16scalarShapeLegal,
    .build_tile = matmulqhpifp16scalarBuildTile,
    .late_rewrite = matmulqhpifp16scalarLateRewrite
};


/* QHPI execute function implementation for MatMulQhpiFp16Scalar kernel matmulqhpifp16scalar_float_16_ */
static uint32_t matmulqhpifp16scalar_float_16_Execute(
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

  // FP16 reference implementation executed on a scalar HTP worker.
  for (uint32_t b = 0; b < batch; ++b) {
    for (uint32_t h = 0; h < heads; ++h) {
      for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t col = 0; col < n; ++col) {
          float sum = 0.0f;

          for (uint32_t reduction = 0; reduction < k; ++reduction) {
            const uint64_t lhs_index =
                (((uint64_t)b * heads + h) * m + row) * k + reduction;

            const uint64_t rhs_index =
                (((uint64_t)b * heads + h) * k + reduction) * n + col;

            sum += static_cast<float>(lhs[lhs_index]) *
                   static_cast<float>(rhs[rhs_index]);
          }

          const uint64_t output_index =
              (((uint64_t)b * heads + h) * m + row) * n + col;

          output[output_index] = static_cast<__fp16>(sum);
        }
      }
    }
  }

  return QHPI_Success;
}

static float matmulqhpifp16scalar_float_16_CostFunc(const uint32_t num_inputs, const QHPI_Tensor *const *inputs)
{
  /*
   * Cost estimation function for MatMulQhpiFp16Scalar kernel matmulqhpifp16scalar_float_16_
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
 * Common stub implementations for MatMulQhpiFp16Scalar QHPI_OpInfo functions
 * These are shared across all kernels and provide default no-op implementations
 */

static const QHPI_Op* matmulqhpifp16scalarEarlyRewrite(const QHPI_Op *op)
{
  /*
   * Early rewrite function for MatMulQhpiFp16Scalar
   * Called during graph optimization phase
   * Return the original op if no rewriting is needed, or a new op if rewriting is required
   */
  return op;  // No rewriting by default
}

static QHPI_Shape matmulqhpifp16scalarShapeRequired(const QHPI_Op *op)
{
  /*
   * Shape required function for MatMulQhpiFp16Scalar
   * Specifies required input shapes for the operation
   * Return empty shape if no specific shape requirements
   */
  QHPI_Shape empty_shape = {0};  // Empty shape by default
  return empty_shape;
}

static QHPI_Shape matmulqhpifp16scalarShapeLegal(const QHPI_Op *op, const QHPI_Shape* shape)
{
  /*
   * Shape legal function for MatMulQhpiFp16Scalar
   * Validates if a given shape is legal for this operation
   * Return the shape if legal, or modified shape if not legal
   */
  return *shape;  // Accept the provided shape by default
}

static const QHPI_Op* matmulqhpifp16scalarBuildTile(const QHPI_Op *op, const QHPI_Shape* start, const QHPI_Shape* extent)
{
  /*
   * Build tile function for MatMulQhpiFp16Scalar
   * Creates a tiled version of the operation
   * Return a new op that operates on the specified tile, or op if tiling is not supported
   */
  return op;  // No tiling support by default
}

static const QHPI_Op* matmulqhpifp16scalarLateRewrite(const QHPI_Op *op)
{
  /*
   * Late rewrite function for MatMulQhpiFp16Scalar
   * Called during late optimization phase
   * Return the original op if no rewriting is needed, or a new op if rewriting is required
   */
  return op;  // No rewriting by default
}

// Array of all MatMulQhpiFp16Scalar operations for registration
static QHPI_OpInfo_v1 matmulqhpifp16scalar_ops[] = {
    matmulqhpifp16scalarOpInfo
};

// Registration function for MatMulQhpiFp16Scalar operations
extern "C" void register_matmulqhpifp16scalar_ops()
{
    qhpi_register_ops_v1(sizeof(matmulqhpifp16scalar_ops) / sizeof(matmulqhpifp16scalar_ops[0]), matmulqhpifp16scalar_ops, THIS_PKG_NAME_STR);
}
