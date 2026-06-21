//==============================================================================
// Auto Generated Code for MatMulCustomOpPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"


BEGIN_PKG_OP_DEFINITION(PKG_MatMulCustom);


// op execute function declarations
template<typename TensorType>
GraphStatus matmulcustomImpl(TensorType& out_0,
                             const TensorType& lhs,
                             const TensorType& rhs);

// forward declaration of sample cost function
static float matmulcustomCostFunc(const Op *op);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag (Flags::RESOURCE_HVX)
 * syntax: DEF_PACKAGE_OP(F,OP)
 * e.g. DEF_PACKAGE_OP((matmulcustomImpl<Tensor>), "MatMulCustom")
 */
DEF_PACKAGE_OP((matmulcustomImpl<Tensor>), "MatMulCustom")

/*
 * method 2 for defining op with specified cost value (one of GLACIAL, SNAIL, FAST, FREE)
 * and provided flags
 * syntax: DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...)
 * can use zero or more flags, FLAG options are IS_CONST, INHIBIT_CONST_PROP,
 * RESOURCE_HVX, RESOURCE_HMX(not supported in external op packages)
 * e.g. DEF_PACKAGE_OP_AND_COST_AND_FLAGS((matmulcustomImpl<PlainFloatTensor>), "MatMulCustom", SNAIL)
 */

DEF_PACKAGE_OP_AND_COST_AND_FLAGS(
  (matmulcustomImpl<PlainFloatTensor>),
  "MatMulCustom",
  SNAIL
)

/*
 * method 3 for defining op with cost function pointer and provided flags
 * cost function pointer type: typedef float (*cost_function) (const Op * op);
 * syntax: DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F,OP,COST_F,...)
 * e.g. DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((matmulcustomImpl<PlainFloatTensor>),
 * "MatMulCustom", matmulcustomCostFunc, Flags::RESOURCE_HVX)
 */

/*
 * optimization definitions
 * need to be global in the package
 * one definition per optimization
 * syntax: DEF_PACKAGE_OPTIMIZATION(PRIORITY,MATCHCODE,CONSTRAINTCODE,REPLACECODE)
 * PRIORITY predefined values include EARLY(2000), MIDDLE(3000), LATE(4000)
 * HTP core provides some replacement functions for op package to use
 * for more information about optimization rules, please refer to HTP core documentations
 */

/*
 * op parameter order definitions
 * need to be global in the package
 * one definition per op, and this is optional
 * syntax: DEF_PACKAGE_PARAM_ORDER(OP,PARAM1,MANDATORY1,DEFAULT1,PARAM2,MANDATORY2,DEFAULT2...)
 * one or more parameters can be specified for each op
     * order of parameters listed determines the order of parameters passed into op execution functions
 * if an op does not have a parameter order definition, parameter order passed into Qnn_addNode
 *   will be passed into op execution functions
 * if an op has a parameter order definition, any parameter passed into Qnn_addNode with unlisted
     *   name will be abandoned
 * if two or more op packages with the same package name will be registered, they cannot list
 *   conflicting parameter orders
 * PARAM refers to parameter name as a string literal
 * MANDATORY refers to whether this parameter is required to be provided at Qnn_addNode
 * DEFAULT is used when MANDATORY is false
 *     if provided as Qnn_Param_t*,
 *       DEFAULT will be used for graph construction when this parameter is not provided at
 *       Qnn_addNode
 *     if provided as nullptr,
 *       graph construction will skip this parameter when this parameter is not provided at
 *       Qnn_addNode
 */


/* execute functions for ops */

template<typename TensorType>
GraphStatus matmulcustomImpl(TensorType& out_0,
                             const TensorType& lhs,
                             const TensorType& rhs)

{
  /*
   * add code here
   * */
  /*
   * To have good performance and stability, it is required to avoid heap memory
   * allocation in this function. The heap memory allocation includes but not
   * limited to calling malloc, operator new, constructing STL container objects
   * like std::vector with default allocator, and adding items like calling
   * std::vector::push_back to STL container objects with default allocator.
   *
   * Please check in SDK documentation for more information.
   */
  if (lhs.dim(0) != rhs.dim(0) ||
      lhs.dim(1) != rhs.dim(1) ||
      lhs.dim(3) != rhs.dim(2)) {
    return GraphStatus::ErrorFatal;      
  }

  const Idx batches   = lhs.dim(0);
  const Idx heads     = lhs.dim(1);
  const Idx rows      = lhs.dim(2);
  const Idx depth     = lhs.dim(3);
  const Idx columns   = rhs.dim(3);

  size_t outputDimensions[4] = {
    static_cast<size_t>(batches),
    static_cast<size_t>(heads),
    static_cast<size_t>(rows),
    static_cast<size_t>(columns),
  };

  out_0.set_dims(outputDimensions);

  for (Idx b = 0; b < batches; ++b) {
    for (Idx h = 0; h < heads; ++h) {
      for (Idx m = 0; m < rows; ++m) {
        for (Idx n = 0; n < columns; ++n) {
          float accumulator = 0.0f;

          for (Idx k = 0; k < depth; ++k) {
            accumulator += lhs(b, h, m, k) * rhs(b, h, k, n);
          }

          out_0(b, h, m, n) = accumulator;
        }
      }
    }
  }

  return GraphStatus::Success;
}

__attribute__((unused)) static float matmulcustomCostFunc(const Op *op)
{
  /*
   * add code here
   * */

  float cost = 0.0;  // add cost computation here
  return cost;
}





/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_MatMulCustom);