//==============================================================================
// Minimal QNN model for MatMulCustomOpPackage::MatMulCustom.
//==============================================================================

#include "QnnModel.hpp"

using namespace qnn_wrapper_api;

extern "C" {

QNN_API
ModelError_t QnnModel_composeGraphs(
    Qnn_BackendHandle_t backendHandle,
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

  QnnModel customMatMulModel;
  const QnnGraph_Config_t** graphConfigs = nullptr;

  VALIDATE(
      getQnnGraphConfigFromInfo(
          "customMatMulHtpGraph",
          graphsConfigInfo,
          numGraphsConfigInfo,
          graphConfigs),
      err);

  VALIDATE(
      customMatMulModel.initialize(
          backendHandle,
          interface,
          contextHandle,
          "customMatMulHtpGraph",
          debug,
          true,
          graphConfigs),
      err);

  // lhs: [B, H, M, K] = [1, 1, 2, 3]
  uint32_t lhsDimensions[] = {1, 1, 2, 3};

  VALIDATE(
      customMatMulModel.addTensor(
          "lhs",
          (Qnn_Tensor_t){
              .version = QNN_TENSOR_VERSION_1,
              .v1 = {
                  .id         = 0,
                  .name       = "lhs",
                  .type       = QNN_TENSOR_TYPE_APP_WRITE,
                  .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                  .dataType   = QNN_DATATYPE_FLOAT_32,
                  .quantizeParams = {
                      QNN_DEFINITION_UNDEFINED,
                      QNN_QUANTIZATION_ENCODING_UNDEFINED,
                      {.scaleOffsetEncoding = {
                          .scale  = 0.0f,
                          .offset = 0,
                      }},
                  },
                  .rank       = 4,
                  .dimensions = lhsDimensions,
                  .memType    = QNN_TENSORMEMTYPE_RAW,
                  .clientBuf  = {
                      .data     = nullptr,
                      .dataSize = 0,
                  },
              },
          }),
      err);

  // rhs: [B, H, K, N] = [1, 1, 3, 2]
  uint32_t rhsDimensions[] = {1, 1, 3, 2};

  VALIDATE(
      customMatMulModel.addTensor(
          "rhs",
          (Qnn_Tensor_t){
              .version = QNN_TENSOR_VERSION_1,
              .v1 = {
                  .id         = 0,
                  .name       = "rhs",
                  .type       = QNN_TENSOR_TYPE_APP_WRITE,
                  .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                  .dataType   = QNN_DATATYPE_FLOAT_32,
                  .quantizeParams = {
                      QNN_DEFINITION_UNDEFINED,
                      QNN_QUANTIZATION_ENCODING_UNDEFINED,
                      {.scaleOffsetEncoding = {
                          .scale  = 0.0f,
                          .offset = 0,
                      }},
                  },
                  .rank       = 4,
                  .dimensions = rhsDimensions,
                  .memType    = QNN_TENSORMEMTYPE_RAW,
                  .clientBuf  = {
                      .data     = nullptr,
                      .dataSize = 0,
                  },
              },
          }),
      err);

  const char* inputNames[] = {"lhs", "rhs"};

  // output: [B, H, M, N] = [1, 1, 2, 2]
  uint32_t outputDimensions[] = {1, 1, 2, 2};

  Qnn_Tensor_t outputs[] = {
      (Qnn_Tensor_t){
          .version = QNN_TENSOR_VERSION_1,
          .v1 = {
              .id         = 0,
              .name       = "output",
              .type       = QNN_TENSOR_TYPE_APP_READ,
              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
              .dataType   = QNN_DATATYPE_FLOAT_32,
              .quantizeParams = {
                  QNN_DEFINITION_UNDEFINED,
                  QNN_QUANTIZATION_ENCODING_UNDEFINED,
                  {.scaleOffsetEncoding = {
                      .scale  = 0.0f,
                      .offset = 0,
                  }},
              },
              .rank       = 4,
              .dimensions = outputDimensions,
              .memType    = QNN_TENSORMEMTYPE_RAW,
              .clientBuf  = {
                  .data     = nullptr,
                  .dataSize = 0,
              },
          },
      },
  };

  VALIDATE(
      customMatMulModel.addNode(
          QNN_OPCONFIG_VERSION_1,
          "MatMulCustom_0",
          "MatMulCustomOpPackage",
          "MatMulCustom",
          nullptr,
          0,
          inputNames,
          2,
          outputs,
          1),
      err);

  QnnModel* models[] = {&customMatMulModel};
  const uint32_t numModels = 1;

  VALIDATE(
      getGraphInfoFromModels(
          *models,
          numModels,
          graphsInfo),
      err);

  *numGraphsInfo = numModels;
  return err;
}

QNN_API
ModelError_t QnnModel_freeGraphsInfo(
    GraphInfoPtr_t** graphs,
    uint32_t numGraphsInfo) {
  return qnn_wrapper_api::freeGraphsInfo(
      graphs,
      numGraphsInfo);
}

}  // extern "C"