//==============================================================================
// Minimal QNN model used to exercise ReluOpPackage::Relu.
//==============================================================================

#include "QnnModel.hpp"

using namespace qnn_wrapper_api;

extern "C" {
QNN_API
ModelError_t QnnModel_composeGraphs(Qnn_BackendHandle_t backendHandle,
                                    QNN_INTERFACE_VER_TYPE interface,
                                    Qnn_ContextHandle_t contextHandle,
                                    const GraphConfigInfo_t** graphsConfigInfo,
                                    const uint32_t numGraphsConfigInfo,
                                    GraphInfoPtr_t** graphsInfo,
                                    uint32_t* numGraphsInfo,
                                    bool debug,
                                    QnnLog_Callback_t logCallback,
                                    QnnLog_Level_t maxLogLevel) {
  ModelError_t err = MODEL_NO_ERROR;

  QnnModel customReluModel;
  const QnnGraph_Config_t** graphConfigs = nullptr;
  VALIDATE(getQnnGraphConfigFromInfo(
               "customReluGraph", graphsConfigInfo, numGraphsConfigInfo, graphConfigs),
           err);
  VALIDATE(customReluModel.initialize(backendHandle,
                                     interface,
                                     contextHandle,
                                     "customReluGraph",
                                     debug,
                                     true,
                                     graphConfigs),
           err);

  uint32_t inputDimensions[] = {1, 4};
  VALIDATE(customReluModel.addTensor(
               "input",
               (Qnn_Tensor_t){
                   .version = QNN_TENSOR_VERSION_1,
                   .v1      = {.id             = 0,
                          .name           = "input",
                          .type           = QNN_TENSOR_TYPE_APP_WRITE,
                          .dataFormat     = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                          .dataType       = QNN_DATATYPE_FLOAT_32,
                          .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                             QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                             {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}}},
                          .rank           = 2,
                          .dimensions     = inputDimensions,
                          .memType        = QNN_TENSORMEMTYPE_RAW,
                          .clientBuf      = {.data = nullptr, .dataSize = 0}}}),
           err);

  const char* inputNames[] = {"input"};
  uint32_t outputDimensions[] = {1, 4};
  Qnn_Tensor_t outputs[]      = {(Qnn_Tensor_t){
      .version = QNN_TENSOR_VERSION_1,
      .v1      = {.id             = 0,
             .name           = "output",
             .type           = QNN_TENSOR_TYPE_APP_READ,
             .dataFormat     = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
             .dataType       = QNN_DATATYPE_FLOAT_32,
             .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}}},
             .rank           = 2,
             .dimensions     = outputDimensions,
             .memType        = QNN_TENSORMEMTYPE_RAW,
             .clientBuf      = {.data = nullptr, .dataSize = 0}}}};

  VALIDATE(customReluModel.addNode(QNN_OPCONFIG_VERSION_1,
                                   "CustomRelu_0",
                                   "ReluOpPackage",
                                   "Relu",
                                   nullptr,
                                   0,
                                   inputNames,
                                   1,
                                   outputs,
                                   1),
           err);

  QnnModel* models[] = {&customReluModel};
  const uint32_t numModels = 1;
  VALIDATE(getGraphInfoFromModels(*models, numModels, graphsInfo), err);
  *numGraphsInfo = numModels;

  return err;
}

QNN_API
ModelError_t QnnModel_freeGraphsInfo(GraphInfoPtr_t** graphs, uint32_t numGraphsInfo) {
  return qnn_wrapper_api::freeGraphsInfo(graphs, numGraphsInfo);
}
}
