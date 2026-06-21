//==============================================================================
// Auto Generated Code for MatMulQhpiScalar - QHPI Implementation
// Multiple kernels generated for different data type combinations
//==============================================================================

#include "HTP/core/constraints.h"
#include <string>

// Plugin/QHPI includes - using correct header from hexnn_qhpi.h
#include "HTP/core/qhpi.h"


// Forward declarations for MatMulQhpiScalar kernel matmulqhpiscalar_float_32_
static uint32_t matmulqhpiscalar_float_32_Execute(QHPI_RuntimeHandle *handle,
                                      uint32_t num_outputs, QHPI_Tensor **outputs,
                                      uint32_t num_inputs, const QHPI_Tensor *const *inputs);
static float matmulqhpiscalar_float_32_CostFunc(const uint32_t num_inputs, const QHPI_Tensor *const *inputs);

// Common forward declarations for MatMulQhpiScalar
static const QHPI_Op* matmulqhpiscalarEarlyRewrite(const QHPI_Op *op);
static QHPI_Shape matmulqhpiscalarShapeRequired(const QHPI_Op *op);
static QHPI_Shape matmulqhpiscalarShapeLegal(const QHPI_Op *op, const QHPI_Shape* shape);
static const QHPI_Op* matmulqhpiscalarBuildTile(const QHPI_Op *op, const QHPI_Shape* start, const QHPI_Shape* extent);
static const QHPI_Op* matmulqhpiscalarLateRewrite(const QHPI_Op *op);

/*
 * QHPI Registration using hexnn_ffi.h API for MatMulQhpiScalar
 * Multiple kernels for different data type combinations
 */


// Input tensor signatures for MatMulQhpiScalar kernel matmulqhpiscalar_float_32_
// Includes both regular inputs and parameters as inputs
static QHPI_Tensor_Signature_v1 matmulqhpiscalar_float_32_InputSignatures[] = {

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

static QHPI_Tensor_Signature_v1 matmulqhpiscalar_float_32_OutputSignatures[] = {

    {
        .element_type = QHPI_Float32,
        .layout = QHPI_Layout_Flat4,
        .storage = QHPI_Storage_Direct,
        .mem_placement = QHPI_MemLoc_DDR_OR_TCM
    }
};

// Kernel definition for MatMulQhpiScalar kernel matmulqhpiscalar_float_32_
static QHPI_Kernel_v1 matmulqhpiscalar_float_32_Kernel = {
    .function_name = "matmulqhpiscalar_float_32_Execute",
    .function = matmulqhpiscalar_float_32_Execute,
    .resources = QHPI_RESOURCE_HVX,
    .source_destructive = false,
    .multithreaded = false,
    .variable_inputs = false,
    .variable_outputs = false,
    .min_inputs = 2,
    .input_signature = matmulqhpiscalar_float_32_InputSignatures,
    .min_outputs = 1,
    .output_signature = matmulqhpiscalar_float_32_OutputSignatures,
    .cost_function = matmulqhpiscalar_float_32_CostFunc,
    .sync_block_size = 0,
    .precomputed_data_size = 0,
    .do_precomputation_function = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate = nullptr
};

// Array of all kernels for MatMulQhpiScalar
static QHPI_Kernel_v1 matmulqhpiscalarKernels[] = {

    matmulqhpiscalar_float_32_Kernel
};

// Operator info for MatMulQhpiScalar - exported for package registration
QHPI_OpInfo_v1 matmulqhpiscalarOpInfo = {
    .name = THIS_PKG_NAME_STR "::" "MatMulQhpiScalar",
    .num_kernels = 1,
    .kernels = matmulqhpiscalarKernels,
    .early_rewrite = matmulqhpiscalarEarlyRewrite,
    .shape_required = matmulqhpiscalarShapeRequired,
    .shape_legalized = matmulqhpiscalarShapeLegal,
    .build_tile = matmulqhpiscalarBuildTile,
    .late_rewrite = matmulqhpiscalarLateRewrite
};


/* QHPI execute function implementation for MatMulQhpiScalar kernel matmulqhpiscalar_float_32_ */
static uint32_t matmulqhpiscalar_float_32_Execute(QHPI_RuntimeHandle *handle,
                                      uint32_t num_outputs, QHPI_Tensor **outputs,
                                      uint32_t num_inputs, const QHPI_Tensor *const *inputs)
{
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

  const uint32_t batch    = lhs_shape.dims[0];
  const uint32_t heads    = lhs_shape.dims[1];
  const uint32_t m        = lhs_shape.dims[2];
  const uint32_t k        = lhs_shape.dims[3];

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

  const float *lhs = static_cast<const float*>(qhpi_tensor_raw_data(inputs[0]));
  const float *rhs =
      static_cast<const float *>(qhpi_tensor_raw_data(inputs[1]));

  float *output =
      static_cast<float *>(qhpi_tensor_raw_data(outputs[0]));

  if (lhs == nullptr || rhs == nullptr || output == nullptr) {
    return QHPI_ErrorFatal;
  }

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

            sum += lhs[lhs_index] * rhs[rhs_index];
          }

          const uint64_t output_index =
              (((uint64_t)b * heads + h) * m + row) * n + col;

          output[output_index] = sum;
        }
      }
    }
  }

  return QHPI_Success;
}

static float matmulqhpiscalar_float_32_CostFunc(const uint32_t num_inputs, const QHPI_Tensor *const *inputs)
{
  /*
   * Cost estimation function for MatMulQhpiScalar kernel matmulqhpiscalar_float_32_
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
 * Common stub implementations for MatMulQhpiScalar QHPI_OpInfo functions
 * These are shared across all kernels and provide default no-op implementations
 */

static const QHPI_Op* matmulqhpiscalarEarlyRewrite(const QHPI_Op *op)
{
  /*
   * Early rewrite function for MatMulQhpiScalar
   * Called during graph optimization phase
   * Return the original op if no rewriting is needed, or a new op if rewriting is required
   */
  return op;  // No rewriting by default
}

static QHPI_Shape matmulqhpiscalarShapeRequired(const QHPI_Op *op)
{
  /*
   * Shape required function for MatMulQhpiScalar
   * Specifies required input shapes for the operation
   * Return empty shape if no specific shape requirements
   */
  QHPI_Shape empty_shape = {0};  // Empty shape by default
  return empty_shape;
}

static QHPI_Shape matmulqhpiscalarShapeLegal(const QHPI_Op *op, const QHPI_Shape* shape)
{
  /*
   * Shape legal function for MatMulQhpiScalar
   * Validates if a given shape is legal for this operation
   * Return the shape if legal, or modified shape if not legal
   */
  return *shape;  // Accept the provided shape by default
}

static const QHPI_Op* matmulqhpiscalarBuildTile(const QHPI_Op *op, const QHPI_Shape* start, const QHPI_Shape* extent)
{
  /*
   * Build tile function for MatMulQhpiScalar
   * Creates a tiled version of the operation
   * Return a new op that operates on the specified tile, or op if tiling is not supported
   */
  return op;  // No tiling support by default
}

static const QHPI_Op* matmulqhpiscalarLateRewrite(const QHPI_Op *op)
{
  /*
   * Late rewrite function for MatMulQhpiScalar
   * Called during late optimization phase
   * Return the original op if no rewriting is needed, or a new op if rewriting is required
   */
  return op;  // No rewriting by default
}

// Array of all MatMulQhpiScalar operations for registration
static QHPI_OpInfo_v1 matmulqhpiscalar_ops[] = {
    matmulqhpiscalarOpInfo
};

// Registration function for MatMulQhpiScalar operations
extern "C" void register_matmulqhpiscalar_ops()
{
    qhpi_register_ops_v1(sizeof(matmulqhpiscalar_ops) / sizeof(matmulqhpiscalar_ops[0]), matmulqhpiscalar_ops, THIS_PKG_NAME_STR);
}
