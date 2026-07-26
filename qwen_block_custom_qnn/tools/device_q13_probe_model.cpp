#include "QnnModel.hpp"

using namespace qnn_wrapper_api;

extern "C" {

QNN_API
ModelError_t QnnModel_composeGraphs(
    Qnn_BackendHandle_t backendHandle,
    QNN_INTERFACE_VER_TYPE interface,
    Qnn_ContextHandle_t contextHandle,
    const GraphConfigInfo_t **graphsConfigInfo,
    const uint32_t numGraphsConfigInfo,
    GraphInfoPtr_t **graphsInfo,
    uint32_t *numGraphsInfo,
    bool debug,
    QnnLog_Callback_t,
    QnnLog_Level_t) {
  ModelError_t err = MODEL_NO_ERROR;
  QnnModel model;
  const QnnGraph_Config_t **graphConfigs = nullptr;
  constexpr const char *graphName = "deviceQ13ProbeGraph";
  VALIDATE(getQnnGraphConfigFromInfo(graphName, graphsConfigInfo,
                                     numGraphsConfigInfo, graphConfigs), err);
  VALIDATE(model.initialize(backendHandle, interface, contextHandle, graphName,
                            debug, true, graphConfigs), err);

  uint32_t dimensions[] = {1, 1, 896, 896};
  Qnn_Tensor_t input = {
      .version = QNN_TENSOR_VERSION_1,
      .v1 = {
          .id = 0,
          .name = "weight_fp32",
          .type = QNN_TENSOR_TYPE_APP_WRITE,
          .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
          .dataType = QNN_DATATYPE_FLOAT_32,
          .quantizeParams = {
              QNN_DEFINITION_UNDEFINED,
              QNN_QUANTIZATION_ENCODING_UNDEFINED,
              {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}},
          },
          .rank = 4,
          .dimensions = dimensions,
          .memType = QNN_TENSORMEMTYPE_RAW,
          .clientBuf = {.data = nullptr, .dataSize = 0},
      },
  };
  VALIDATE(model.addTensor("weight_fp32", input), err);

  const char *castInputs[] = {"weight_fp32"};
  Qnn_Tensor_t castOutputs[] = {{
      .version = QNN_TENSOR_VERSION_1,
      .v1 = {
          .id = 0,
          .name = "weight_fp16",
          .type = QNN_TENSOR_TYPE_NATIVE,
          .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
          .dataType = QNN_DATATYPE_FLOAT_16,
          .quantizeParams = {
              QNN_DEFINITION_UNDEFINED,
              QNN_QUANTIZATION_ENCODING_UNDEFINED,
              {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}},
          },
          .rank = 4,
          .dimensions = dimensions,
          .memType = QNN_TENSORMEMTYPE_RAW,
          .clientBuf = {.data = nullptr, .dataSize = 0},
      },
  }};
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, "CastWeightToFp16",
                         "qti.aisw", "Cast", nullptr, 0, castInputs, 1,
                         castOutputs, 1), err);

  const char *probeInputs[] = {"weight_fp16"};
  Qnn_Tensor_t probeOutputs[] = {{
      .version = QNN_TENSOR_VERSION_1,
      .v1 = {
          .id = 0,
          .name = "weight_q13",
          .type = QNN_TENSOR_TYPE_APP_READ,
          .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
          .dataType = QNN_DATATYPE_INT_16,
          .quantizeParams = {
              QNN_DEFINITION_UNDEFINED,
              QNN_QUANTIZATION_ENCODING_UNDEFINED,
              {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}},
          },
          .rank = 4,
          .dimensions = dimensions,
          .memType = QNN_TENSORMEMTYPE_RAW,
          .clientBuf = {.data = nullptr, .dataSize = 0},
      },
  }};
  VALIDATE(model.addNode(
      QNN_OPCONFIG_VERSION_1, "ConvertFp16ToDeviceQ13_0",
      "MatMulQhpiHvx8RowFp32StoreMultithreadOpPackage",
      "ConvertFp16ToDeviceQ13", nullptr, 0, probeInputs, 1, probeOutputs, 1),
      err);

  QnnModel *models[] = {&model};
  VALIDATE(getGraphInfoFromModels(*models, 1, graphsInfo), err);
  *numGraphsInfo = 1;
  return err;
}

QNN_API
ModelError_t QnnModel_freeGraphsInfo(GraphInfoPtr_t **graphs,
                                     uint32_t numGraphsInfo) {
  return qnn_wrapper_api::freeGraphsInfo(graphs, numGraphsInfo);
}

}  // extern "C"
