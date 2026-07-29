//==============================================================================
// Auto Generated Code for ReluOpPackage
//==============================================================================
#include <algorithm>
#include <iostream>
#include <string>

#include "CpuBackendUtils.hpp"
#include "CustomOpPackage.hpp"

using namespace qnn::custom;
using namespace qnn::custom::utils;

namespace relu {

Qnn_ErrorHandle_t execute(CustomOp* operation) {
  QNN_CUSTOM_BE_ENSURE(operation != nullptr, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)

  auto input  = operation->getInput(0);
  auto output = operation->getOutput(0);

  QNN_CUSTOM_BE_ENSURE(input != nullptr, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
  QNN_CUSTOM_BE_ENSURE(output != nullptr, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
  QNN_CUSTOM_BE_ENSURE_EQ(input->dataType,
                          QNN_CPU_DATATYPE_FLOAT_32,
                          QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(output->dataType,
                          QNN_CPU_DATATYPE_FLOAT_32,
                          QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  const auto* inputData = static_cast<const float*>(input->data);
  auto* outputData      = static_cast<float*>(output->data);

  QNN_CUSTOM_BE_ENSURE(inputData != nullptr, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
  QNN_CUSTOM_BE_ENSURE(outputData != nullptr, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)

  const uint32_t elementCount = numTensorSize(input);
  for (uint32_t i = 0; i < elementCount; ++i) {
    outputData[i] = std::max(inputData[i], 0.0f);
  }

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t finalize(const CustomOp* operation) {
  QNN_CUSTOM_BE_ENSURE(operation != nullptr, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numInput(), 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numOutput(), 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  auto input  = operation->getInput(0);
  auto output = operation->getOutput(0);

  QNN_CUSTOM_BE_ENSURE(input != nullptr, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
  QNN_CUSTOM_BE_ENSURE(output != nullptr, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
  QNN_CUSTOM_BE_ENSURE_EQ(input->dataType,
                          QNN_CPU_DATATYPE_FLOAT_32,
                          QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(output->dataType,
                          QNN_CPU_DATATYPE_FLOAT_32,
                          QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(input->rank,
                          output->rank,
                          QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(numTensorSize(input),
                          numTensorSize(output),
                          QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t free(CustomOp& operation) {

  /**
   * Add code here
   **/

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t populateFromNode(const QnnOpPackage_Node_t node,
                                   QnnOpPackage_GraphInfrastructure_t graphInfrastructure,
                                   CustomOp* operation) {
  // Add input
  for (uint32_t i = 0; i < numInputs(node); i++) {
    operation->addInput(getInput(node, i));
  }

  // Add output
  for (uint32_t i = 0; i < numOutputs(node); i++) {
    operation->addOutput(getOutput(node, i));
  }


  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t validateOpConfig(Qnn_OpConfig_t opConfig) {
  QNN_CUSTOM_BE_ENSURE_EQ(
      strcmp(opConfig.v1.typeName, "Relu"), 0, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)

  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfInputs, 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfOutputs, 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  return QNN_SUCCESS;
}
}  // namespace relu

CustomOpRegistration_t* register_ReluCustomOp() {
  using namespace relu;
  static CustomOpRegistration_t ReluRegister = {execute, finalize, free, validateOpConfig, populateFromNode};
  return &ReluRegister;
}

REGISTER_OP(Relu, register_ReluCustomOp);
