#include "QnnModel.hpp"

#include <array>
#include <cmath>

using namespace qnn_wrapper_api;

namespace {

constexpr uint32_t kRows = 256;
constexpr uint32_t kColumns = 64;

std::array<Qnn_ScaleOffset_t, kColumns> &perAxisScaleOffsets() {
  static std::array<Qnn_ScaleOffset_t, kColumns> values = [] {
    std::array<Qnn_ScaleOffset_t, kColumns> result{};
    for (uint32_t column = 0; column < kColumns; ++column) {
      const float fraction =
          static_cast<float>(column) / static_cast<float>(kColumns - 1);
      const float channel_range = 0.02f * std::pow(50.0f, fraction);
      result[column] = {
          .scale = channel_range / 127.0f,
          .offset = -128,
      };
    }
    return result;
  }();
  return values;
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
          .rank = 2,
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
  constexpr const char *graphName = "qnnInt8PerAxisGraph";

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

  uint32_t dimensions[] = {kRows, kColumns};
  const Qnn_QuantizeParams_t no_quantization = {
      QNN_DEFINITION_UNDEFINED,
      QNN_QUANTIZATION_ENCODING_UNDEFINED,
      {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}},
  };
  const Qnn_QuantizeParams_t per_tensor_quantization = {
      QNN_DEFINITION_DEFINED,
      QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
      {.scaleOffsetEncoding = {
          .scale = 1.0f / 127.0f,
          .offset = -128,
      }},
  };
  auto &axis_scales = perAxisScaleOffsets();
  const Qnn_QuantizeParams_t per_axis_quantization = {
      QNN_DEFINITION_DEFINED,
      QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET,
      {.axisScaleOffsetEncoding = {
          .axis = 1,
          .numScaleOffsets = kColumns,
          .scaleOffset = axis_scales.data(),
      }},
  };

  Qnn_Tensor_t input = makeTensor(
      "weights",
      QNN_TENSOR_TYPE_APP_WRITE,
      QNN_DATATYPE_FLOAT_32,
      dimensions,
      no_quantization);
  VALIDATE(model.addTensor("weights", input), err);

  const char *inputs[] = {"weights"};
  Qnn_Tensor_t perTensorOutput[] = {
      makeTensor(
          "per_tensor_uint8",
          QNN_TENSOR_TYPE_NATIVE,
          QNN_DATATYPE_UFIXED_POINT_8,
          dimensions,
          per_tensor_quantization),
  };
  VALIDATE(
      model.addNode(
          QNN_OPCONFIG_VERSION_1,
          "QuantizePerTensor_0",
          "qti.aisw",
          "Quantize",
          nullptr,
          0,
          inputs,
          1,
          perTensorOutput,
          1),
      err);

  const char *perTensorDequantizeInputs[] = {"per_tensor_uint8"};
  Qnn_Tensor_t perTensorDequantized[] = {
      makeTensor(
          "per_tensor_output",
          QNN_TENSOR_TYPE_APP_READ,
          QNN_DATATYPE_FLOAT_32,
          dimensions,
          no_quantization),
  };
  VALIDATE(
      model.addNode(
          QNN_OPCONFIG_VERSION_1,
          "DequantizePerTensor_0",
          "qti.aisw",
          "Dequantize",
          nullptr,
          0,
          perTensorDequantizeInputs,
          1,
          perTensorDequantized,
          1),
      err);

  Qnn_Tensor_t perAxisOutput[] = {
      makeTensor(
          "per_axis_uint8",
          QNN_TENSOR_TYPE_NATIVE,
          QNN_DATATYPE_UFIXED_POINT_8,
          dimensions,
          per_axis_quantization),
  };
  VALIDATE(
      model.addNode(
          QNN_OPCONFIG_VERSION_1,
          "QuantizePerAxis_0",
          "qti.aisw",
          "Quantize",
          nullptr,
          0,
          inputs,
          1,
          perAxisOutput,
          1),
      err);

  const char *perAxisDequantizeInputs[] = {"per_axis_uint8"};
  Qnn_Tensor_t perAxisDequantized[] = {
      makeTensor(
          "per_axis_output",
          QNN_TENSOR_TYPE_APP_READ,
          QNN_DATATYPE_FLOAT_32,
          dimensions,
          no_quantization),
  };
  VALIDATE(
      model.addNode(
          QNN_OPCONFIG_VERSION_1,
          "DequantizePerAxis_0",
          "qti.aisw",
          "Dequantize",
          nullptr,
          0,
          perAxisDequantizeInputs,
          1,
          perAxisDequantized,
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
