#include "QnnModel.hpp"

#include <array>

using namespace qnn_wrapper_api;

namespace {

Qnn_QuantizeParams_t quantParams(float scale) {
  return {
      QNN_DEFINITION_DEFINED,
      QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
      {.scaleOffsetEncoding = {.scale = scale, .offset = -128}},
  };
}

Qnn_QuantizeParams_t rhsPerAxisQuantParams() {
  static std::array<float, 256> scales = [] {
    std::array<float, 256> result{};
    float channel_range = 0.02f;
    for (uint32_t column = 0; column < result.size(); ++column) {
      result[column] = channel_range / 127.0f;
      channel_range *= 1.0127030493398506f;
    }
    return result;
  }();
  return {
      QNN_DEFINITION_DEFINED,
      QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET,
      {.bwAxisScaleOffsetEncoding = {
          .bitwidth = 8,
          .axis = 3,
          .numElements = static_cast<uint32_t>(scales.size()),
          .scales = scales.data(),
          .offsets = nullptr,
      }},
  };
}

Qnn_Tensor_t makeTensor(const char *name,
                        Qnn_TensorType_t type,
                        Qnn_DataType_t data_type,
                        uint32_t rank,
                        uint32_t *dimensions,
                        Qnn_QuantizeParams_t quantize_params) {
  return {
      .version = QNN_TENSOR_VERSION_1,
      .v1 = {
          .id = 0,
          .name = name,
          .type = type,
          .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
          .dataType = data_type,
          .quantizeParams = quantize_params,
          .rank = rank,
          .dimensions = dimensions,
          .memType = QNN_TENSORMEMTYPE_RAW,
          .clientBuf = {.data = nullptr, .dataSize = 0},
      },
  };
}

}  // namespace

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
    QnnLog_Callback_t logCallback,
    QnnLog_Level_t maxLogLevel) {
  (void)logCallback;
  (void)maxLogLevel;

  ModelError_t err = MODEL_NO_ERROR;
  QnnModel model;
  const QnnGraph_Config_t **graphConfigs = nullptr;
  constexpr const char *graphName = "int8PerAxisWeightMatMulGraph";

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

  uint32_t lhsDimensions[] = {1, 1, 128, 256};
  uint32_t rhsDimensions[] = {1, 1, 256, 256};
  uint32_t outputDimensions[] = {1, 1, 128, 256};

  Qnn_Tensor_t lhs = makeTensor(
      "lhs",
      QNN_TENSOR_TYPE_APP_WRITE,
      QNN_DATATYPE_UFIXED_POINT_8,
      4,
      lhsDimensions,
      quantParams(0.003936838211975698f));
  Qnn_Tensor_t rhs = makeTensor(
      "rhs",
      QNN_TENSOR_TYPE_STATIC,
      QNN_DATATYPE_SFIXED_POINT_8,
      4,
      rhsDimensions,
      rhsPerAxisQuantParams());
  rhs.v1.clientBuf = {
      .data = BINVARSTART(rhs_per_axis_weight),
      .dataSize = BINLEN(rhs_per_axis_weight),
  };
  VALIDATE(model.addTensor("lhs", lhs), err);
  VALIDATE(model.addTensor("rhs", rhs), err);

  const char *inputs[] = {"lhs", "rhs"};
  Qnn_Tensor_t outputs[] = {
      makeTensor(
          "output",
          QNN_TENSOR_TYPE_APP_READ,
          QNN_DATATYPE_UFIXED_POINT_8,
          4,
          outputDimensions,
          quantParams(0.03250987135519193f)),
  };
  VALIDATE(
      model.addNode(
          QNN_OPCONFIG_VERSION_1,
          "Int8PerAxisWeightMatMul_0",
          "qti.aisw",
          "MatMul",
          nullptr,
          0,
          inputs,
          2,
          outputs,
          1),
      err);

  QnnModel *models[] = {&model};
  constexpr uint32_t modelCount = 1;
  VALIDATE(getGraphInfoFromModels(*models, modelCount, graphsInfo), err);
  *numGraphsInfo = modelCount;
  return err;
}

QNN_API
ModelError_t QnnModel_freeGraphsInfo(
    GraphInfoPtr_t **graphs,
    uint32_t numGraphsInfo) {
  return qnn_wrapper_api::freeGraphsInfo(graphs, numGraphsInfo);
}

}  // extern "C"
