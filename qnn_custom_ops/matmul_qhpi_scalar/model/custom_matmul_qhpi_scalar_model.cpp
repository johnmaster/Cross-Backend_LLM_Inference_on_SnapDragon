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
  (void)logCallback;
  (void)maxLogLevel;

  ModelError_t err = MODEL_NO_ERROR;

  QnnModel model;
  const QnnGraph_Config_t** graphConfigs = nullptr;
  constexpr const char* graphName = "matMulQhpiScalarGraph";

  VALIDATE(
      getQnnGraphConfigFromInfo(
          graphName,
          graphsConfigInfo,
          numGraphsConfigInfo,
          graphConfigs),
      err);

  VALIDATE(
      model.initialize(
          backendHandle,
          interface,
          contextHandle,
          graphName,
          debug,
          true,
          graphConfigs),
      err);

  // lhs[B,H,M,K] = [1,1,2,3]
  uint32_t lhsDimensions[] = {1, 1, 2, 3};

  Qnn_Tensor_t lhsTensor = {
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
  };

  VALIDATE(model.addTensor("lhs", lhsTensor), err);

  // rhs[B,H,K,N] = [1,1,3,2]
  uint32_t rhsDimensions[] = {1, 1, 3, 2};

  Qnn_Tensor_t rhsTensor = {
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
  };

  VALIDATE(model.addTensor("rhs", rhsTensor), err);

  const char* inputNames[] = {
      "lhs",
      "rhs",
  };

  // output[B,H,M,N] = [1,1,2,2]
  uint32_t outputDimensions[] = {1, 1, 2, 2};

  Qnn_Tensor_t outputTensors[] = {
      {
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
      model.addNode(
          QNN_OPCONFIG_VERSION_1,
          "MatMulQhpiScalar_0",
          "MatMulQhpiScalarOpPackage",
          "MatMulQhpiScalar",
          nullptr,
          0,
          inputNames,
          2,
          outputTensors,
          1),
      err);

  QnnModel* models[] = {&model};
  constexpr uint32_t modelCount = 1;

  VALIDATE(
      getGraphInfoFromModels(
          *models,
          modelCount,
          graphsInfo),
      err);

  *numGraphsInfo = modelCount;
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
