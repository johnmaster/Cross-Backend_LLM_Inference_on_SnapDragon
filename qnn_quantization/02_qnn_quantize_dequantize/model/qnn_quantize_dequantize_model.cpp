#include "QnnModel.hpp"

using namespace qnn_wrapper_api;

namespace {

Qnn_QuantizeParams_t quantParams(float scale, int32_t offset) {
  return {
      QNN_DEFINITION_DEFINED,
      QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
      {.scaleOffsetEncoding = {.scale = scale, .offset = offset}},
  };
}

Qnn_Tensor_t makeTensor(const char *name,
                        Qnn_TensorType_t type,
                        Qnn_DataType_t data_type,
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
          .rank = 1,
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
  constexpr const char *graphName = "qnnQuantizeDequantizeGraph";

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

  uint32_t dimensions[] = {32768};
  const Qnn_QuantizeParams_t no_quantization = {
      QNN_DEFINITION_UNDEFINED,
      QNN_QUANTIZATION_ENCODING_UNDEFINED,
      {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}},
  };
  const Qnn_QuantizeParams_t symmetric_quantization =
      quantParams(0.013284672902325007f, -128);
  const Qnn_QuantizeParams_t asymmetric_quantization =
      quantParams(0.007043378727108825f, -15);

  Qnn_Tensor_t input = makeTensor(
      "input",
      QNN_TENSOR_TYPE_APP_WRITE,
      QNN_DATATYPE_FLOAT_32,
      dimensions,
      no_quantization);
  VALIDATE(model.addTensor("input", input), err);

  const char *quantizeInputs[] = {"input"};
  Qnn_Tensor_t symmetricQuantized[] = {
      makeTensor(
          "symmetric_int8",
          QNN_TENSOR_TYPE_APP_READ,
          QNN_DATATYPE_UFIXED_POINT_8,
          dimensions,
          symmetric_quantization),
  };
  VALIDATE(
      model.addNode(
          QNN_OPCONFIG_VERSION_1,
          "QuantizeSymmetricInt8_0",
          "qti.aisw",
          "Quantize",
          nullptr,
          0,
          quantizeInputs,
          1,
          symmetricQuantized,
          1),
      err);

  const char *symmetricDequantizeInputs[] = {"symmetric_int8"};
  Qnn_Tensor_t symmetricOutput[] = {
      makeTensor(
          "symmetric_output",
          QNN_TENSOR_TYPE_APP_READ,
          QNN_DATATYPE_FLOAT_32,
          dimensions,
          no_quantization),
  };
  VALIDATE(
      model.addNode(
          QNN_OPCONFIG_VERSION_1,
          "DequantizeSymmetricInt8_0",
          "qti.aisw",
          "Dequantize",
          nullptr,
          0,
          symmetricDequantizeInputs,
          1,
          symmetricOutput,
          1),
      err);

  Qnn_Tensor_t asymmetricQuantized[] = {
      makeTensor(
          "asymmetric_uint8",
          QNN_TENSOR_TYPE_APP_READ,
          QNN_DATATYPE_UFIXED_POINT_8,
          dimensions,
          asymmetric_quantization),
  };
  VALIDATE(
      model.addNode(
          QNN_OPCONFIG_VERSION_1,
          "QuantizeAsymmetricUint8_0",
          "qti.aisw",
          "Quantize",
          nullptr,
          0,
          quantizeInputs,
          1,
          asymmetricQuantized,
          1),
      err);

  const char *asymmetricDequantizeInputs[] = {"asymmetric_uint8"};
  Qnn_Tensor_t asymmetricOutput[] = {
      makeTensor(
          "asymmetric_output",
          QNN_TENSOR_TYPE_APP_READ,
          QNN_DATATYPE_FLOAT_32,
          dimensions,
          no_quantization),
  };
  VALIDATE(
      model.addNode(
          QNN_OPCONFIG_VERSION_1,
          "DequantizeAsymmetricUint8_0",
          "qti.aisw",
          "Dequantize",
          nullptr,
          0,
          asymmetricDequantizeInputs,
          1,
          asymmetricOutput,
          1),
      err);

  QnnModel *models[] = {&model};
  constexpr uint32_t modelCount = 1;
  VALIDATE(
      getGraphInfoFromModels(*models, modelCount, graphsInfo),
      err);
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
