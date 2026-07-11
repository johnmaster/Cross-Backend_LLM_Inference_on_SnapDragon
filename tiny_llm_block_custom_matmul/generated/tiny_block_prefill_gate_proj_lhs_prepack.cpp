/* COPYRIGHT HEADER GOES HERE: No CopyRight Header String Passed During Model Conversion */

/* Patched by tiny_llm_block_custom_matmul/tools/patch_prefill_gate_proj_lhs_prepack.py
 * Replaces gate_proj FullyConnected with MatMulQhpiHvx8RowLhsPrepackFp32Store.
 */

/* Command Line used:
qnn-onnx-converter; act_bitwidth=8; act_quantizer=tf; act_quantizer_calibration=min-max; act_quantizer_schema=asymmetric; adjust_nms_features_dims=True; algorithms=[]; align_matmul_ranks=True; apply_masked_softmax=uncompressed; arch_checker=False; backend=None; batch=None; bias_bitwidth=8; calc_static_encodings=False; converter_op_package_lib=; copyright_file=None; custom_io=; custom_op_config_paths=None; debug=-1; defer_loading=False; define_symbol=None; disable_batchnorm_folding=False; disable_defer_loading=False; disable_node_validation=False; disable_qnn_op_config_validation=False; disable_relu_squashing=False; dry_run=None; dumpIR=False; dump_custom_io_config_template=; dump_encoding_json=False; dump_inferred_model=False; dump_ir=; dump_ir_optimizer_config_template=False; dump_optimization_pass_mode_config=False; dump_pass_trace_info=False; dump_qairt_io_config_yaml=; dump_qairt_quantizer_command=None; dump_value_info=False; enable_framework_trace=False; enable_match_gathernd=False; enable_match_topk=False; enable_per_row_quantized_bias=False; exclude_named_tensors=False; expand_gru_op_structure=True; expand_lstm_op_structure=False; expand_sparse_op_structure=False; export_format=cpp; extract_color_transform=True; float_bias_bitwidth=0; float_bias_bw=0; float_bitwidth=32; float_bw=32; float_fallback=False; force_prune_cast_ops=False; handle_gather_negative_indices=True; ignore_encodings=False; include_data_invariant_ops=False; inject_cast_for_gather=True; input_dim=None; input_dtype=[]; input_encoding=[]; input_layout=[]; input_list=None; input_type=[]; ir_optimizer_config=; keep_disconnected_nodes=False; keep_int64_inputs=False; keep_quant_nodes=False; keep_weights_quantized=False; match_caffe_ssd_to_tf=True; model_version=None; multi_time_steps_gru=False; multi_time_steps_lstm=False; no_simplification=False; op_package_lib=; optimization_pass_mode=ir_optimizer_mainline; optimization_pass_mode_config=; out_names=['hidden_out', 'present_key', 'present_value']; overwrite_model_prefix=False; pack_4_bit_weights=False; package_name=None; packed_masked_softmax_inputs=[]; packed_max_seq=1; param_quantizer=None; param_quantizer_calibration=min-max; param_quantizer_schema=asymmetric; percentile_calibration_value=99.99; perform_axes_to_spatial_first_order=True; perform_layout_transformation=False; prepare_inputs_as_params=False; preprocess_roi_pool_inputs=True; preserve_io=[['layout']]; preserve_onnx_output_order=False; quantization_overrides=; quantizer_log=None; quantizer_log_level=LogLevel.NONE; restrict_quantization_steps=[]; squash_box_decoder=True; unroll_gru_time_steps=True; unroll_lstm_time_steps=True; use_aimet_quantizer=False; use_convert_quantization_nodes=False; use_dynamic_16_bit_weights=False; use_native_dtype=False; use_native_input_files=False; use_native_output_files=False; use_per_channel_quantization=False; use_per_row_quantization=False; use_quantize_v2=False; validate_models=False; weights_bitwidth=8
*/

#include "QnnOpDef.h"
#include "QnnModel.hpp"

// Flag to determine if Backend should do node validation for each opNode added
#define DO_GRAPH_NODE_VALIDATIONS 1

using namespace qnn_wrapper_api;
const __attribute__((visibility("default"))) char* QNN_SDK_VERSION = "qaisw-v2.47.0.260601114230";
extern "C" {
static ModelError_t addTensor_hidden_states(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_hidden_states[] = {1, 32, 256};
  VALIDATE(model.addTensor("hidden_states", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "hidden_states",
                                 .type= QNN_TENSOR_TYPE_APP_WRITE,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 3,
                                 .dimensions=dimensions_hidden_states,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=nullptr,
                                                .dataSize=0}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode_hidden_states_nfc(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR hidden_states_nfc */
  uint32_t dimensions_hidden_states_nfc_perm[] = {3};
  uint32_t hidden_states_nfc_perm[] = {0, 2, 1};
  Qnn_Param_t params_hidden_states_nfc[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "hidden_states_nfc_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_hidden_states_nfc_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)hidden_states_nfc_perm,
                           .dataSize=12}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs_hidden_states_nfc[] = {
    "hidden_states"
  };
  uint32_t dimensions_hidden_states_nfc[] = {1, 256, 32};
  Qnn_Tensor_t outputs_hidden_states_nfc[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "hidden_states_nfc",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions_hidden_states_nfc,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "hidden_states_nfc", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params_hidden_states_nfc, // Node Params
                         1, // Num Node Params
                         inputs_hidden_states_nfc, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_hidden_states_nfc, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_input_norm_weight(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_input_norm_weight[] = {256};
  VALIDATE(model.addTensor("input_norm_weight", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "input_norm_weight",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_input_norm_weight,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(input_norm_weight),
                                                .dataSize=BINLEN(input_norm_weight)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_input_norm_weight_bias(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_input_norm_weight_bias[] = {256};
  VALIDATE(model.addTensor("input_norm_weight_bias", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "input_norm_weight_bias",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_input_norm_weight_bias,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(input_norm_weight_bias),
                                                .dataSize=BINLEN(input_norm_weight_bias)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode_rms_norm__Mul_2_output_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR rms_norm__Mul_2_output_0 */
  uint32_t dimensions_rms_norm__Mul_2_output_0_axes[] = {1};
  uint32_t rms_norm__Mul_2_output_0_axes[] = {2};
  Qnn_Param_t params_rms_norm__Mul_2_output_0[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="axes",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "rms_norm__Mul_2_output_0_axes",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_rms_norm__Mul_2_output_0_axes,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)rms_norm__Mul_2_output_0_axes,
                           .dataSize=4}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="epsilon",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 0.0000009999999974752427078783512115478516f}}}}
  };
  const char*  inputs_rms_norm__Mul_2_output_0[] = {
    "hidden_states",
    "input_norm_weight",
    "input_norm_weight_bias"
  };
  uint32_t dimensions__Mul_2_output_0[] = {1, 32, 256};
  Qnn_Tensor_t outputs_rms_norm__Mul_2_output_0[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Mul_2_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__Mul_2_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "rms_norm__Mul_2_output_0", // Node Name
                         "qti.aisw", // Package Name
                         "RmsNorm", // Qnn Node Type
                         params_rms_norm__Mul_2_output_0, // Node Params
                         2, // Num Node Params
                         inputs_rms_norm__Mul_2_output_0, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs_rms_norm__Mul_2_output_0, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__MatMul_2_pre_reshape(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _MatMul_2_pre_reshape */
  const char*  inputs__MatMul_2_pre_reshape[] = {
    "_Mul_2_output_0"
  };
  uint32_t dimensions__MatMul_2_pre_reshape[] = {32, 256};
  Qnn_Tensor_t outputs__MatMul_2_pre_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_2_pre_reshape",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__MatMul_2_pre_reshape,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_MatMul_2_pre_reshape", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__MatMul_2_pre_reshape, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__MatMul_2_pre_reshape, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__MatMul_1_pre_reshape(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _MatMul_1_pre_reshape */
  const char*  inputs__MatMul_1_pre_reshape[] = {
    "_Mul_2_output_0"
  };
  uint32_t dimensions__MatMul_1_pre_reshape[] = {32, 256};
  Qnn_Tensor_t outputs__MatMul_1_pre_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_1_pre_reshape",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__MatMul_1_pre_reshape,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_MatMul_1_pre_reshape", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__MatMul_1_pre_reshape, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__MatMul_1_pre_reshape, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__MatMul_pre_reshape(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _MatMul_pre_reshape */
  const char*  inputs__MatMul_pre_reshape[] = {
    "_Mul_2_output_0"
  };
  uint32_t dimensions__MatMul_pre_reshape[] = {32, 256};
  Qnn_Tensor_t outputs__MatMul_pre_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_pre_reshape",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__MatMul_pre_reshape,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_MatMul_pre_reshape", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__MatMul_pre_reshape, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__MatMul_pre_reshape, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_q_proj_weight(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_q_proj_weight[] = {256, 256};
  VALIDATE(model.addTensor("q_proj_weight", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "q_proj_weight",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 2,
                                 .dimensions=dimensions_q_proj_weight,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(q_proj_weight),
                                                .dataSize=BINLEN(q_proj_weight)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_q_proj_bias(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_q_proj_bias[] = {256};
  VALIDATE(model.addTensor("q_proj_bias", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "q_proj_bias",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_q_proj_bias,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(q_proj_bias),
                                                .dataSize=BINLEN(q_proj_bias)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__MatMul(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _MatMul */
  const char*  inputs__MatMul[] = {
    "_MatMul_pre_reshape",
    "q_proj_weight",
    "q_proj_bias"
  };
  uint32_t dimensions__Add_1_output_0_fc[] = {32, 256};
  Qnn_Tensor_t outputs__MatMul[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Add_1_output_0_fc",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Add_1_output_0_fc,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_MatMul", // Node Name
                         "qti.aisw", // Package Name
                         "FullyConnected", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__MatMul, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__MatMul, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__MatMul_post_reshape(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _MatMul_post_reshape */
  const char*  inputs__MatMul_post_reshape[] = {
    "_Add_1_output_0_fc"
  };
  uint32_t dimensions__Reshape_output_0[] = {1, 32, 8, 32};
  Qnn_Tensor_t outputs__MatMul_post_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Reshape_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Reshape_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_MatMul_post_reshape", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__MatMul_post_reshape, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__MatMul_post_reshape, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Transpose(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Transpose */
  uint32_t dimensions__Transpose_perm[] = {4};
  uint32_t _Transpose_perm[] = {0, 2, 1, 3};
  Qnn_Param_t params__Transpose[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Transpose_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__Transpose_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_Transpose_perm,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__Transpose[] = {
    "_Reshape_output_0"
  };
  uint32_t dimensions__Transpose_output_0[] = {1, 8, 32, 32};
  Qnn_Tensor_t outputs__Transpose[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Transpose_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Transpose_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Transpose", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__Transpose, // Node Params
                         1, // Num Node Params
                         inputs__Transpose, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Transpose, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_k_proj_weight(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_k_proj_weight[] = {64, 256};
  VALIDATE(model.addTensor("k_proj_weight", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "k_proj_weight",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 2,
                                 .dimensions=dimensions_k_proj_weight,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(k_proj_weight),
                                                .dataSize=BINLEN(k_proj_weight)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_k_proj_bias(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_k_proj_bias[] = {64};
  VALIDATE(model.addTensor("k_proj_bias", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "k_proj_bias",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_k_proj_bias,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(k_proj_bias),
                                                .dataSize=BINLEN(k_proj_bias)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__MatMul_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _MatMul_1 */
  const char*  inputs__MatMul_1[] = {
    "_MatMul_1_pre_reshape",
    "k_proj_weight",
    "k_proj_bias"
  };
  uint32_t dimensions__Add_2_output_0_fc[] = {32, 64};
  Qnn_Tensor_t outputs__MatMul_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Add_2_output_0_fc",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Add_2_output_0_fc,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_MatMul_1", // Node Name
                         "qti.aisw", // Package Name
                         "FullyConnected", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__MatMul_1, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__MatMul_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__MatMul_1_post_reshape(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _MatMul_1_post_reshape */
  const char*  inputs__MatMul_1_post_reshape[] = {
    "_Add_2_output_0_fc"
  };
  uint32_t dimensions__Reshape_1_output_0[] = {1, 32, 2, 32};
  Qnn_Tensor_t outputs__MatMul_1_post_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Reshape_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Reshape_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_MatMul_1_post_reshape", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__MatMul_1_post_reshape, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__MatMul_1_post_reshape, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Transpose_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Transpose_1 */
  uint32_t dimensions__Transpose_1_perm[] = {4};
  uint32_t _Transpose_1_perm[] = {0, 2, 1, 3};
  Qnn_Param_t params__Transpose_1[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Transpose_1_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__Transpose_1_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_Transpose_1_perm,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__Transpose_1[] = {
    "_Reshape_1_output_0"
  };
  uint32_t dimensions__Transpose_1_output_0[] = {1, 2, 32, 32};
  Qnn_Tensor_t outputs__Transpose_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Transpose_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Transpose_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Transpose_1", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__Transpose_1, // Node Params
                         1, // Num Node Params
                         inputs__Transpose_1, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Transpose_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_v_proj_weight(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_v_proj_weight[] = {64, 256};
  VALIDATE(model.addTensor("v_proj_weight", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "v_proj_weight",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 2,
                                 .dimensions=dimensions_v_proj_weight,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(v_proj_weight),
                                                .dataSize=BINLEN(v_proj_weight)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_v_proj_bias(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_v_proj_bias[] = {64};
  VALIDATE(model.addTensor("v_proj_bias", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "v_proj_bias",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_v_proj_bias,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(v_proj_bias),
                                                .dataSize=BINLEN(v_proj_bias)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__MatMul_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _MatMul_2 */
  const char*  inputs__MatMul_2[] = {
    "_MatMul_2_pre_reshape",
    "v_proj_weight",
    "v_proj_bias"
  };
  uint32_t dimensions__Add_3_output_0_fc[] = {32, 64};
  Qnn_Tensor_t outputs__MatMul_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Add_3_output_0_fc",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Add_3_output_0_fc,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_MatMul_2", // Node Name
                         "qti.aisw", // Package Name
                         "FullyConnected", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__MatMul_2, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__MatMul_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__MatMul_2_post_reshape(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _MatMul_2_post_reshape */
  const char*  inputs__MatMul_2_post_reshape[] = {
    "_Add_3_output_0_fc"
  };
  uint32_t dimensions__Reshape_2_output_0[] = {1, 32, 2, 32};
  Qnn_Tensor_t outputs__MatMul_2_post_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Reshape_2_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Reshape_2_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_MatMul_2_post_reshape", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__MatMul_2_post_reshape, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__MatMul_2_post_reshape, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Transpose_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Transpose_2 */
  uint32_t dimensions__Transpose_2_perm[] = {4};
  uint32_t _Transpose_2_perm[] = {0, 2, 1, 3};
  Qnn_Param_t params__Transpose_2[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Transpose_2_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__Transpose_2_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_Transpose_2_perm,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__Transpose_2[] = {
    "_Reshape_2_output_0"
  };
  uint32_t dimensions_present_value[] = {1, 2, 32, 32};
  Qnn_Tensor_t outputs__Transpose_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "present_value",
            .type= QNN_TENSOR_TYPE_APP_READ,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions_present_value,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Transpose_2", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__Transpose_2, // Node Params
                         1, // Num Node Params
                         inputs__Transpose_2, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Transpose_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Slice(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Slice */
  uint32_t dimensions__Slice_ranges[] = {4, 3};
  int32_t _Slice_ranges[] = {0, 1, 1, 0, 8, 1, 0, 32, 1, 0, 16, 1};
  Qnn_Param_t params__Slice[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="ranges",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Slice_ranges",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_INT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Slice_ranges,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_Slice_ranges,
                           .dataSize=48}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="begin_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="end_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="new_axes_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="shrink_axes",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__Slice[] = {
    "_Transpose_output_0"
  };
  uint32_t dimensions__Slice_output_0[] = {1, 8, 32, 16};
  Qnn_Tensor_t outputs__Slice[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Slice_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Slice_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Slice", // Node Name
                         "qti.aisw", // Package Name
                         "StridedSlice", // Qnn Node Type
                         params__Slice, // Node Params
                         5, // Num Node Params
                         inputs__Slice, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Slice, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Slice_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Slice_1 */
  uint32_t dimensions__Slice_1_ranges[] = {4, 3};
  int32_t _Slice_1_ranges[] = {0, 1, 1, 0, 8, 1, 0, 32, 1, 16, 32, 1};
  Qnn_Param_t params__Slice_1[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="ranges",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Slice_1_ranges",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_INT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Slice_1_ranges,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_Slice_1_ranges,
                           .dataSize=48}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="begin_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="end_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="new_axes_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="shrink_axes",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__Slice_1[] = {
    "_Transpose_output_0"
  };
  uint32_t dimensions__Slice_1_output_0[] = {1, 8, 32, 16};
  Qnn_Tensor_t outputs__Slice_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Slice_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Slice_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Slice_1", // Node Name
                         "qti.aisw", // Package Name
                         "StridedSlice", // Qnn Node Type
                         params__Slice_1, // Node Params
                         5, // Num Node Params
                         inputs__Slice_1, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Slice_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__Unsqueeze_1_output_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__Unsqueeze_1_output_0[] = {1, 1, 32, 16};
  VALIDATE(model.addTensor("_Unsqueeze_1_output_0", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_Unsqueeze_1_output_0",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions__Unsqueeze_1_output_0,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_Unsqueeze_1_output_0),
                                                .dataSize=BINLEN(_Unsqueeze_1_output_0)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__Mul_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Mul_3 */
  Qnn_Param_t params__Mul_3[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 13}}}}
  };
  const char*  inputs__Mul_3[] = {
    "_Slice_output_0",
    "_Unsqueeze_1_output_0"
  };
  uint32_t dimensions__Mul_3_output_0[] = {1, 8, 32, 16};
  Qnn_Tensor_t outputs__Mul_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Mul_3_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Mul_3_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Mul_3", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Mul_3, // Node Params
                         1, // Num Node Params
                         inputs__Mul_3, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Mul_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__Unsqueeze_3_output_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__Unsqueeze_3_output_0[] = {1, 1, 32, 16};
  VALIDATE(model.addTensor("_Unsqueeze_3_output_0", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_Unsqueeze_3_output_0",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions__Unsqueeze_3_output_0,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_Unsqueeze_3_output_0),
                                                .dataSize=BINLEN(_Unsqueeze_3_output_0)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__Mul_4(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Mul_4 */
  Qnn_Param_t params__Mul_4[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 13}}}}
  };
  const char*  inputs__Mul_4[] = {
    "_Slice_1_output_0",
    "_Unsqueeze_3_output_0"
  };
  uint32_t dimensions__Mul_4_output_0[] = {1, 8, 32, 16};
  Qnn_Tensor_t outputs__Mul_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Mul_4_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Mul_4_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Mul_4", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Mul_4, // Node Params
                         1, // Num Node Params
                         inputs__Mul_4, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Mul_4, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Sub(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Sub */
  Qnn_Param_t params__Sub[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 18}}}}
  };
  const char*  inputs__Sub[] = {
    "_Mul_3_output_0",
    "_Mul_4_output_0"
  };
  uint32_t dimensions__Sub_output_0[] = {1, 8, 32, 16};
  Qnn_Tensor_t outputs__Sub[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Sub_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Sub_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Sub", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Sub, // Node Params
                         1, // Num Node Params
                         inputs__Sub, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Sub, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Mul_5(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Mul_5 */
  Qnn_Param_t params__Mul_5[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 13}}}}
  };
  const char*  inputs__Mul_5[] = {
    "_Slice_1_output_0",
    "_Unsqueeze_1_output_0"
  };
  uint32_t dimensions__Mul_5_output_0[] = {1, 8, 32, 16};
  Qnn_Tensor_t outputs__Mul_5[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Mul_5_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Mul_5_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Mul_5", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Mul_5, // Node Params
                         1, // Num Node Params
                         inputs__Mul_5, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Mul_5, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Mul_6(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Mul_6 */
  Qnn_Param_t params__Mul_6[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 13}}}}
  };
  const char*  inputs__Mul_6[] = {
    "_Slice_output_0",
    "_Unsqueeze_3_output_0"
  };
  uint32_t dimensions__Mul_6_output_0[] = {1, 8, 32, 16};
  Qnn_Tensor_t outputs__Mul_6[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Mul_6_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Mul_6_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Mul_6", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Mul_6, // Node Params
                         1, // Num Node Params
                         inputs__Mul_6, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Mul_6, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Add_4(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Add_4 */
  Qnn_Param_t params__Add_4[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__Add_4[] = {
    "_Mul_5_output_0",
    "_Mul_6_output_0"
  };
  uint32_t dimensions__Add_4_output_0[] = {1, 8, 32, 16};
  Qnn_Tensor_t outputs__Add_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Add_4_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Add_4_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Add_4", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Add_4, // Node Params
                         1, // Num Node Params
                         inputs__Add_4, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Add_4, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Concat(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Concat */
  Qnn_Param_t params__Concat[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 3}}}}
  };
  const char*  inputs__Concat[] = {
    "_Sub_output_0",
    "_Add_4_output_0"
  };
  uint32_t dimensions__Concat_output_0[] = {1, 8, 32, 32};
  Qnn_Tensor_t outputs__Concat[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Concat_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Concat_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Concat", // Node Name
                         "qti.aisw", // Package Name
                         "Concat", // Qnn Node Type
                         params__Concat, // Node Params
                         1, // Num Node Params
                         inputs__Concat, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Concat, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Slice_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Slice_2 */
  uint32_t dimensions__Slice_2_ranges[] = {4, 3};
  int32_t _Slice_2_ranges[] = {0, 1, 1, 0, 2, 1, 0, 32, 1, 0, 16, 1};
  Qnn_Param_t params__Slice_2[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="ranges",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Slice_2_ranges",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_INT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Slice_2_ranges,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_Slice_2_ranges,
                           .dataSize=48}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="begin_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="end_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="new_axes_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="shrink_axes",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__Slice_2[] = {
    "_Transpose_1_output_0"
  };
  uint32_t dimensions__Slice_2_output_0[] = {1, 2, 32, 16};
  Qnn_Tensor_t outputs__Slice_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Slice_2_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Slice_2_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Slice_2", // Node Name
                         "qti.aisw", // Package Name
                         "StridedSlice", // Qnn Node Type
                         params__Slice_2, // Node Params
                         5, // Num Node Params
                         inputs__Slice_2, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Slice_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Slice_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Slice_3 */
  uint32_t dimensions__Slice_3_ranges[] = {4, 3};
  int32_t _Slice_3_ranges[] = {0, 1, 1, 0, 2, 1, 0, 32, 1, 16, 32, 1};
  Qnn_Param_t params__Slice_3[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="ranges",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Slice_3_ranges",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_INT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Slice_3_ranges,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_Slice_3_ranges,
                           .dataSize=48}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="begin_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="end_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="new_axes_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="shrink_axes",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__Slice_3[] = {
    "_Transpose_1_output_0"
  };
  uint32_t dimensions__Slice_3_output_0[] = {1, 2, 32, 16};
  Qnn_Tensor_t outputs__Slice_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Slice_3_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Slice_3_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Slice_3", // Node Name
                         "qti.aisw", // Package Name
                         "StridedSlice", // Qnn Node Type
                         params__Slice_3, // Node Params
                         5, // Num Node Params
                         inputs__Slice_3, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Slice_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Mul_7(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Mul_7 */
  Qnn_Param_t params__Mul_7[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 13}}}}
  };
  const char*  inputs__Mul_7[] = {
    "_Slice_2_output_0",
    "_Unsqueeze_1_output_0"
  };
  uint32_t dimensions__Mul_7_output_0[] = {1, 2, 32, 16};
  Qnn_Tensor_t outputs__Mul_7[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Mul_7_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Mul_7_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Mul_7", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Mul_7, // Node Params
                         1, // Num Node Params
                         inputs__Mul_7, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Mul_7, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Mul_8(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Mul_8 */
  Qnn_Param_t params__Mul_8[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 13}}}}
  };
  const char*  inputs__Mul_8[] = {
    "_Slice_3_output_0",
    "_Unsqueeze_3_output_0"
  };
  uint32_t dimensions__Mul_8_output_0[] = {1, 2, 32, 16};
  Qnn_Tensor_t outputs__Mul_8[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Mul_8_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Mul_8_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Mul_8", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Mul_8, // Node Params
                         1, // Num Node Params
                         inputs__Mul_8, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Mul_8, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Sub_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Sub_1 */
  Qnn_Param_t params__Sub_1[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 18}}}}
  };
  const char*  inputs__Sub_1[] = {
    "_Mul_7_output_0",
    "_Mul_8_output_0"
  };
  uint32_t dimensions__Sub_1_output_0[] = {1, 2, 32, 16};
  Qnn_Tensor_t outputs__Sub_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Sub_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Sub_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Sub_1", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Sub_1, // Node Params
                         1, // Num Node Params
                         inputs__Sub_1, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Sub_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Mul_9(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Mul_9 */
  Qnn_Param_t params__Mul_9[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 13}}}}
  };
  const char*  inputs__Mul_9[] = {
    "_Slice_3_output_0",
    "_Unsqueeze_1_output_0"
  };
  uint32_t dimensions__Mul_9_output_0[] = {1, 2, 32, 16};
  Qnn_Tensor_t outputs__Mul_9[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Mul_9_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Mul_9_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Mul_9", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Mul_9, // Node Params
                         1, // Num Node Params
                         inputs__Mul_9, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Mul_9, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Mul_10(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Mul_10 */
  Qnn_Param_t params__Mul_10[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 13}}}}
  };
  const char*  inputs__Mul_10[] = {
    "_Slice_2_output_0",
    "_Unsqueeze_3_output_0"
  };
  uint32_t dimensions__Mul_10_output_0[] = {1, 2, 32, 16};
  Qnn_Tensor_t outputs__Mul_10[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Mul_10_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Mul_10_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Mul_10", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Mul_10, // Node Params
                         1, // Num Node Params
                         inputs__Mul_10, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Mul_10, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Add_5(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Add_5 */
  Qnn_Param_t params__Add_5[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__Add_5[] = {
    "_Mul_9_output_0",
    "_Mul_10_output_0"
  };
  uint32_t dimensions__Add_5_output_0[] = {1, 2, 32, 16};
  Qnn_Tensor_t outputs__Add_5[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Add_5_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Add_5_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Add_5", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Add_5, // Node Params
                         1, // Num Node Params
                         inputs__Add_5, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Add_5, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Concat_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Concat_1 */
  Qnn_Param_t params__Concat_1[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 3}}}}
  };
  const char*  inputs__Concat_1[] = {
    "_Sub_1_output_0",
    "_Add_5_output_0"
  };
  uint32_t dimensions_present_key[] = {1, 2, 32, 32};
  Qnn_Tensor_t outputs__Concat_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "present_key",
            .type= QNN_TENSOR_TYPE_APP_READ,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions_present_key,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Concat_1", // Node Name
                         "qti.aisw", // Package Name
                         "Concat", // Qnn Node Type
                         params__Concat_1, // Node Params
                         1, // Num Node Params
                         inputs__Concat_1, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Concat_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Unsqueeze_4(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Unsqueeze_4 */
  const char*  inputs__Unsqueeze_4[] = {
    "present_key"
  };
  uint32_t dimensions__Unsqueeze_4_output_0[] = {1, 2, 1, 32, 32};
  Qnn_Tensor_t outputs__Unsqueeze_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Unsqueeze_4_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__Unsqueeze_4_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Unsqueeze_4", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__Unsqueeze_4, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Unsqueeze_4, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Tile(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Tile */
  uint32_t dimensions__Tile_multiples[] = {5};
  uint32_t _Tile_multiples[] = {1, 1, 4, 1, 1};
  Qnn_Param_t params__Tile[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="multiples",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Tile_multiples",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__Tile_multiples,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_Tile_multiples,
                           .dataSize=20}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__Tile[] = {
    "_Unsqueeze_4_output_0"
  };
  uint32_t dimensions__Tile_output_0[] = {1, 2, 4, 32, 32};
  Qnn_Tensor_t outputs__Tile[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Tile_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__Tile_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Tile", // Node Name
                         "qti.aisw", // Package Name
                         "Tile", // Qnn Node Type
                         params__Tile, // Node Params
                         1, // Num Node Params
                         inputs__Tile, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Tile, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Reshape_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Reshape_3 */
  const char*  inputs__Reshape_3[] = {
    "_Tile_output_0"
  };
  uint32_t dimensions__Reshape_3_output_0[] = {1, 8, 32, 32};
  Qnn_Tensor_t outputs__Reshape_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Reshape_3_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Reshape_3_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Reshape_3", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__Reshape_3, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Reshape_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Unsqueeze_5(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Unsqueeze_5 */
  const char*  inputs__Unsqueeze_5[] = {
    "present_value"
  };
  uint32_t dimensions__Unsqueeze_5_output_0[] = {1, 2, 1, 32, 32};
  Qnn_Tensor_t outputs__Unsqueeze_5[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Unsqueeze_5_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__Unsqueeze_5_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Unsqueeze_5", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__Unsqueeze_5, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Unsqueeze_5, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Tile_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Tile_1 */
  uint32_t dimensions__Tile_1_multiples[] = {5};
  uint32_t _Tile_1_multiples[] = {1, 1, 4, 1, 1};
  Qnn_Param_t params__Tile_1[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="multiples",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Tile_1_multiples",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__Tile_1_multiples,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_Tile_1_multiples,
                           .dataSize=20}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__Tile_1[] = {
    "_Unsqueeze_5_output_0"
  };
  uint32_t dimensions__Tile_1_output_0[] = {1, 2, 4, 32, 32};
  Qnn_Tensor_t outputs__Tile_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Tile_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__Tile_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Tile_1", // Node Name
                         "qti.aisw", // Package Name
                         "Tile", // Qnn Node Type
                         params__Tile_1, // Node Params
                         1, // Num Node Params
                         inputs__Tile_1, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Tile_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Reshape_4(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Reshape_4 */
  const char*  inputs__Reshape_4[] = {
    "_Tile_1_output_0"
  };
  uint32_t dimensions__Reshape_4_output_0[] = {1, 8, 32, 32};
  Qnn_Tensor_t outputs__Reshape_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Reshape_4_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Reshape_4_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Reshape_4", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__Reshape_4, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Reshape_4, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Transpose_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Transpose_3 */
  uint32_t dimensions__Transpose_3_perm[] = {4};
  uint32_t _Transpose_3_perm[] = {0, 1, 3, 2};
  Qnn_Param_t params__Transpose_3[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Transpose_3_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__Transpose_3_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_Transpose_3_perm,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__Transpose_3[] = {
    "_Reshape_3_output_0"
  };
  uint32_t dimensions__Transpose_3_output_0[] = {1, 8, 32, 32};
  Qnn_Tensor_t outputs__Transpose_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Transpose_3_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Transpose_3_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Transpose_3", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__Transpose_3, // Node Params
                         1, // Num Node Params
                         inputs__Transpose_3, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Transpose_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__MatMul_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _MatMul_3 */
  Qnn_Param_t params__MatMul_3[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs__MatMul_3[] = {
    "_Concat_output_0",
    "_Transpose_3_output_0"
  };
  uint32_t dimensions__MatMul_3_output_0[] = {1, 8, 32, 32};
  Qnn_Tensor_t outputs__MatMul_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_3_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__MatMul_3_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_MatMul_3", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params__MatMul_3, // Node Params
                         2, // Num Node Params
                         inputs__MatMul_3, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__MatMul_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__Constant_44_output_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__Constant_44_output_0[] = {1};
  VALIDATE(model.addTensor("_Constant_44_output_0", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_Constant_44_output_0",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions__Constant_44_output_0,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_Constant_44_output_0),
                                                .dataSize=BINLEN(_Constant_44_output_0)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__Mul_11(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Mul_11 */
  Qnn_Param_t params__Mul_11[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 13}}}}
  };
  const char*  inputs__Mul_11[] = {
    "_MatMul_3_output_0",
    "_Constant_44_output_0"
  };
  uint32_t dimensions__Mul_11_output_0[] = {1, 8, 32, 32};
  Qnn_Tensor_t outputs__Mul_11[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Mul_11_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Mul_11_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Mul_11", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Mul_11, // Node Params
                         1, // Num Node Params
                         inputs__Mul_11, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Mul_11, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__Unsqueeze_7_output_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__Unsqueeze_7_output_0[] = {1, 1, 32, 32};
  VALIDATE(model.addTensor("_Unsqueeze_7_output_0", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_Unsqueeze_7_output_0",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions__Unsqueeze_7_output_0,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_Unsqueeze_7_output_0),
                                                .dataSize=BINLEN(_Unsqueeze_7_output_0)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__Unsqueeze_7_output_0_cast_to_bool(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Unsqueeze_7_output_0_cast_to_bool */
  const char*  inputs__Unsqueeze_7_output_0_cast_to_bool[] = {
    "_Unsqueeze_7_output_0"
  };
  uint32_t dimensions__Unsqueeze_7_output_0_cast_to_bool[] = {1, 1, 32, 32};
  Qnn_Tensor_t outputs__Unsqueeze_7_output_0_cast_to_bool[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Unsqueeze_7_output_0_cast_to_bool",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_BOOL_8,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Unsqueeze_7_output_0_cast_to_bool,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Unsqueeze_7_output_0_cast_to_bool", // Node Name
                         "qti.aisw", // Package Name
                         "Cast", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__Unsqueeze_7_output_0_cast_to_bool, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Unsqueeze_7_output_0_cast_to_bool, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__Constant_49_output_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__Constant_49_output_0[] = {1, 8, 32, 32};
  VALIDATE(model.addTensor("_Constant_49_output_0", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_Constant_49_output_0",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions__Constant_49_output_0,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_Constant_49_output_0),
                                                .dataSize=BINLEN(_Constant_49_output_0)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__Where(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Where */
  const char*  inputs__Where[] = {
    "_Unsqueeze_7_output_0_cast_to_bool",
    "_Mul_11_output_0",
    "_Constant_49_output_0"
  };
  uint32_t dimensions__Where_output_0[] = {1, 8, 32, 32};
  Qnn_Tensor_t outputs__Where[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Where_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Where_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Where", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseSelect", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__Where, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__Where, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Softmax(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Softmax */
  Qnn_Param_t params__Softmax[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 3}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="beta",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 1.0000000000000000000000000000000000000000f}}}}
  };
  const char*  inputs__Softmax[] = {
    "_Where_output_0"
  };
  uint32_t dimensions__Softmax_output_0[] = {1, 8, 32, 32};
  Qnn_Tensor_t outputs__Softmax[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Softmax_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Softmax_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Softmax", // Node Name
                         "qti.aisw", // Package Name
                         "Softmax", // Qnn Node Type
                         params__Softmax, // Node Params
                         2, // Num Node Params
                         inputs__Softmax, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Softmax, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__MatMul_4(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _MatMul_4 */
  Qnn_Param_t params__MatMul_4[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs__MatMul_4[] = {
    "_Softmax_output_0",
    "_Reshape_4_output_0"
  };
  uint32_t dimensions__MatMul_4_output_0[] = {1, 8, 32, 32};
  Qnn_Tensor_t outputs__MatMul_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_4_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__MatMul_4_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_MatMul_4", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params__MatMul_4, // Node Params
                         2, // Num Node Params
                         inputs__MatMul_4, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__MatMul_4, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Transpose_4(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Transpose_4 */
  uint32_t dimensions__Transpose_4_perm[] = {4};
  uint32_t _Transpose_4_perm[] = {0, 2, 1, 3};
  Qnn_Param_t params__Transpose_4[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Transpose_4_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__Transpose_4_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_Transpose_4_perm,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__Transpose_4[] = {
    "_MatMul_4_output_0"
  };
  uint32_t dimensions__Transpose_4_output_0[] = {1, 32, 8, 32};
  Qnn_Tensor_t outputs__Transpose_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Transpose_4_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__Transpose_4_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Transpose_4", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__Transpose_4, // Node Params
                         1, // Num Node Params
                         inputs__Transpose_4, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Transpose_4, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Reshape_5(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Reshape_5 */
  const char*  inputs__Reshape_5[] = {
    "_Transpose_4_output_0"
  };
  uint32_t dimensions__MatMul_5_pre_reshape[] = {32, 256};
  Qnn_Tensor_t outputs__Reshape_5[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_5_pre_reshape",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__MatMul_5_pre_reshape,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Reshape_5", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__Reshape_5, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Reshape_5, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_o_proj_weight(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_o_proj_weight[] = {256, 256};
  VALIDATE(model.addTensor("o_proj_weight", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "o_proj_weight",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 2,
                                 .dimensions=dimensions_o_proj_weight,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(o_proj_weight),
                                                .dataSize=BINLEN(o_proj_weight)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__MatMul_5(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _MatMul_5 */
  const char*  inputs__MatMul_5[] = {
    "_MatMul_5_pre_reshape",
    "o_proj_weight"
  };
  uint32_t dimensions__MatMul_5_output_0_fc[] = {32, 256};
  Qnn_Tensor_t outputs__MatMul_5[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_5_output_0_fc",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__MatMul_5_output_0_fc,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_MatMul_5", // Node Name
                         "qti.aisw", // Package Name
                         "FullyConnected", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__MatMul_5, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__MatMul_5, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__MatMul_5_post_reshape(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _MatMul_5_post_reshape */
  const char*  inputs__MatMul_5_post_reshape[] = {
    "_MatMul_5_output_0_fc"
  };
  uint32_t dimensions__MatMul_5_output_0[] = {1, 32, 256};
  Qnn_Tensor_t outputs__MatMul_5_post_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_5_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__MatMul_5_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_MatMul_5_post_reshape", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__MatMul_5_post_reshape, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__MatMul_5_post_reshape, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__MatMul_5_output_0_nfc(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _MatMul_5_output_0_nfc */
  uint32_t dimensions__MatMul_5_output_0_nfc_perm[] = {3};
  uint32_t _MatMul_5_output_0_nfc_perm[] = {0, 2, 1};
  Qnn_Param_t params__MatMul_5_output_0_nfc[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_5_output_0_nfc_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__MatMul_5_output_0_nfc_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_MatMul_5_output_0_nfc_perm,
                           .dataSize=12}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__MatMul_5_output_0_nfc[] = {
    "_MatMul_5_output_0"
  };
  uint32_t dimensions__MatMul_5_output_0_nfc[] = {1, 256, 32};
  Qnn_Tensor_t outputs__MatMul_5_output_0_nfc[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_5_output_0_nfc",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__MatMul_5_output_0_nfc,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_MatMul_5_output_0_nfc", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__MatMul_5_output_0_nfc, // Node Params
                         1, // Num Node Params
                         inputs__MatMul_5_output_0_nfc, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__MatMul_5_output_0_nfc, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Add_6(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Add_6 */
  Qnn_Param_t params__Add_6[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__Add_6[] = {
    "hidden_states_nfc",
    "_MatMul_5_output_0_nfc"
  };
  uint32_t dimensions__Add_6_output_0[] = {1, 256, 32};
  Qnn_Tensor_t outputs__Add_6[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Add_6_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__Add_6_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Add_6", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Add_6, // Node Params
                         1, // Num Node Params
                         inputs__Add_6, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Add_6, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Add_6_output_0_ncf(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Add_6_output_0_ncf */
  uint32_t dimensions__Add_6_output_0_ncf_perm[] = {3};
  uint32_t _Add_6_output_0_ncf_perm[] = {0, 2, 1};
  Qnn_Param_t params__Add_6_output_0_ncf[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Add_6_output_0_ncf_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__Add_6_output_0_ncf_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_Add_6_output_0_ncf_perm,
                           .dataSize=12}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__Add_6_output_0_ncf[] = {
    "_Add_6_output_0"
  };
  uint32_t dimensions__Add_6_output_0_ncf[] = {1, 32, 256};
  Qnn_Tensor_t outputs__Add_6_output_0_ncf[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Add_6_output_0_ncf",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__Add_6_output_0_ncf,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Add_6_output_0_ncf", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__Add_6_output_0_ncf, // Node Params
                         1, // Num Node Params
                         inputs__Add_6_output_0_ncf, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Add_6_output_0_ncf, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_post_attention_norm_weight(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_post_attention_norm_weight[] = {256};
  VALIDATE(model.addTensor("post_attention_norm_weight", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "post_attention_norm_weight",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_post_attention_norm_weight,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(post_attention_norm_weight),
                                                .dataSize=BINLEN(post_attention_norm_weight)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_post_attention_norm_weight_bias(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_post_attention_norm_weight_bias[] = {256};
  VALIDATE(model.addTensor("post_attention_norm_weight_bias", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "post_attention_norm_weight_bias",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_post_attention_norm_weight_bias,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(post_attention_norm_weight_bias),
                                                .dataSize=BINLEN(post_attention_norm_weight_bias)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode_rms_norm__Mul_14_output_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR rms_norm__Mul_14_output_0 */
  uint32_t dimensions_rms_norm__Mul_14_output_0_axes[] = {1};
  uint32_t rms_norm__Mul_14_output_0_axes[] = {2};
  Qnn_Param_t params_rms_norm__Mul_14_output_0[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="axes",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "rms_norm__Mul_14_output_0_axes",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_rms_norm__Mul_14_output_0_axes,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)rms_norm__Mul_14_output_0_axes,
                           .dataSize=4}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="epsilon",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 0.0000009999999974752427078783512115478516f}}}}
  };
  const char*  inputs_rms_norm__Mul_14_output_0[] = {
    "_Add_6_output_0_ncf",
    "post_attention_norm_weight",
    "post_attention_norm_weight_bias"
  };
  uint32_t dimensions__Mul_14_output_0[] = {1, 32, 256};
  Qnn_Tensor_t outputs_rms_norm__Mul_14_output_0[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Mul_14_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__Mul_14_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "rms_norm__Mul_14_output_0", // Node Name
                         "qti.aisw", // Package Name
                         "RmsNorm", // Qnn Node Type
                         params_rms_norm__Mul_14_output_0, // Node Params
                         2, // Num Node Params
                         inputs_rms_norm__Mul_14_output_0, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs_rms_norm__Mul_14_output_0, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__MatMul_7_pre_reshape(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _MatMul_7_pre_reshape */
  const char*  inputs__MatMul_7_pre_reshape[] = {
    "_Mul_14_output_0"
  };
  uint32_t dimensions__MatMul_7_pre_reshape[] = {32, 256};
  Qnn_Tensor_t outputs__MatMul_7_pre_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_7_pre_reshape",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__MatMul_7_pre_reshape,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_MatMul_7_pre_reshape", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__MatMul_7_pre_reshape, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__MatMul_7_pre_reshape, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__MatMul_6_pre_reshape(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _MatMul_6_pre_reshape */
  const char*  inputs__MatMul_6_pre_reshape[] = {
    "_Mul_14_output_0"
  };
  uint32_t dimensions__MatMul_6_pre_reshape[] = {32, 256};
  Qnn_Tensor_t outputs__MatMul_6_pre_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_6_pre_reshape",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__MatMul_6_pre_reshape,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_MatMul_6_pre_reshape", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__MatMul_6_pre_reshape, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__MatMul_6_pre_reshape, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_gate_proj_weight(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_gate_proj_weight[] = {1, 1, 768, 256};
  VALIDATE(model.addTensor("gate_proj_weight", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "gate_proj_weight",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_gate_proj_weight,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(gate_proj_weight),
                                                .dataSize=BINLEN(gate_proj_weight)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__MatMul_6(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* Replaces gate_proj FullyConnected with LHS-prepack custom HTP MatMul:
     input [32,256] -> [1,1,32,256],
     gate_proj_weight [768,256] -> [1,1,768,256] -> Transpose -> [1,1,256,768]. */
  const char* inputs__MatMul_6_lhs_reshape[] = {
    "_MatMul_6_pre_reshape"
  };
  uint32_t dimensions__MatMul_6_lhs_4d[] = {1, 1, 32, 256};
  Qnn_Tensor_t outputs__MatMul_6_lhs_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_6_lhs_4d",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__MatMul_6_lhs_4d,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,
                         "_MatMul_6_lhs_reshape",
                         "qti.aisw",
                         "Reshape",
                         nullptr,
                         0,
                         inputs__MatMul_6_lhs_reshape,
                         1,
                         outputs__MatMul_6_lhs_reshape,
                         1), err);

  const char* inputs__MatMul_6_lhs_cast[] = {
    "_MatMul_6_lhs_4d"
  };
  Qnn_Tensor_t outputs__MatMul_6_lhs_cast[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_6_lhs_fp16",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_16,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__MatMul_6_lhs_4d,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,
                         "_MatMul_6_lhs_cast_fp16",
                         "qti.aisw",
                         "Cast",
                         nullptr,
                         0,
                         inputs__MatMul_6_lhs_cast,
                         1,
                         outputs__MatMul_6_lhs_cast,
                         1), err);

  const char* inputs__MatMul_6_pack_lhs[] = {
    "_MatMul_6_lhs_fp16"
  };
  Qnn_Tensor_t outputs__MatMul_6_pack_lhs[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_6_lhs_q13_bits",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_16,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__MatMul_6_lhs_4d,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,
                         "_MatMul_6_pack_lhs_q13",
                         "MatMulQhpiHvx8RowLhsPrepackFp32StoreOpPackage",
                         "MatMulQhpiHvxPackLhs",
                         nullptr,
                         0,
                         inputs__MatMul_6_pack_lhs,
                         1,
                         outputs__MatMul_6_pack_lhs,
                         1), err);

  uint32_t dimensions__MatMul_6_weight_transpose_perm[] = {4};
  uint32_t _MatMul_6_weight_transpose_perm[] = {0, 1, 3, 2};
  Qnn_Param_t params__MatMul_6_weight_transpose[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_6_weight_transpose_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__MatMul_6_weight_transpose_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_MatMul_6_weight_transpose_perm,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char* inputs__MatMul_6_weight_transpose[] = {
    "gate_proj_weight"
  };
  uint32_t dimensions__MatMul_6_weight_kn[] = {1, 1, 256, 768};
  Qnn_Tensor_t outputs__MatMul_6_weight_transpose[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_6_weight_kn",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__MatMul_6_weight_kn,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,
                         "_MatMul_6_weight_transpose",
                         "qti.aisw",
                         "Transpose",
                         params__MatMul_6_weight_transpose,
                         1,
                         inputs__MatMul_6_weight_transpose,
                         1,
                         outputs__MatMul_6_weight_transpose,
                         1), err);

  const char* inputs__MatMul_6_rhs_cast[] = {
    "_MatMul_6_weight_kn"
  };
  Qnn_Tensor_t outputs__MatMul_6_rhs_cast[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_6_rhs_fp16",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_16,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__MatMul_6_weight_kn,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,
                         "_MatMul_6_rhs_cast_fp16",
                         "qti.aisw",
                         "Cast",
                         nullptr,
                         0,
                         inputs__MatMul_6_rhs_cast,
                         1,
                         outputs__MatMul_6_rhs_cast,
                         1), err);

  const char* inputs__MatMul_6_custom[] = {
    "_MatMul_6_lhs_q13_bits",
    "_MatMul_6_rhs_fp16"
  };
  uint32_t dimensions__MatMul_6_custom_output_4d[] = {1, 1, 32, 768};
  Qnn_Tensor_t outputs__MatMul_6_custom[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_6_custom_output_4d",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__MatMul_6_custom_output_4d,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,
                         "_MatMul_6",
                         "MatMulQhpiHvx8RowLhsPrepackFp32StoreOpPackage",
                         "MatMulQhpiHvx8RowLhsPrepackFp32Store",
                         nullptr,
                         0,
                         inputs__MatMul_6_custom,
                         2,
                         outputs__MatMul_6_custom,
                         1), err);

  const char* inputs__MatMul_6_output_reshape[] = {
    "_MatMul_6_custom_output_4d"
  };
  uint32_t dimensions__MatMul_6_output_0_fc[] = {32, 768};
  Qnn_Tensor_t outputs__MatMul_6_output_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_6_output_0_fc",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__MatMul_6_output_0_fc,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,
                         "_MatMul_6_output_reshape",
                         "qti.aisw",
                         "Reshape",
                         nullptr,
                         0,
                         inputs__MatMul_6_output_reshape,
                         1,
                         outputs__MatMul_6_output_reshape,
                         1), err);
  return err;
}

static ModelError_t addNode__MatMul_6_post_reshape(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _MatMul_6_post_reshape */
  const char*  inputs__MatMul_6_post_reshape[] = {
    "_MatMul_6_output_0_fc"
  };
  uint32_t dimensions__MatMul_6_output_0[] = {1, 32, 768};
  Qnn_Tensor_t outputs__MatMul_6_post_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_6_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__MatMul_6_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_MatMul_6_post_reshape", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__MatMul_6_post_reshape, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__MatMul_6_post_reshape, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Sigmoid(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Sigmoid */
  Qnn_Param_t params__Sigmoid[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 6}}}}
  };
  const char*  inputs__Sigmoid[] = {
    "_MatMul_6_output_0"
  };
  uint32_t dimensions__Sigmoid_output_0[] = {1, 32, 768};
  Qnn_Tensor_t outputs__Sigmoid[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Sigmoid_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__Sigmoid_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Sigmoid", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__Sigmoid, // Node Params
                         1, // Num Node Params
                         inputs__Sigmoid, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Sigmoid, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Mul_15(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Mul_15 */
  Qnn_Param_t params__Mul_15[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 13}}}}
  };
  const char*  inputs__Mul_15[] = {
    "_MatMul_6_output_0",
    "_Sigmoid_output_0"
  };
  uint32_t dimensions__Mul_15_output_0[] = {1, 32, 768};
  Qnn_Tensor_t outputs__Mul_15[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Mul_15_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__Mul_15_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Mul_15", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Mul_15, // Node Params
                         1, // Num Node Params
                         inputs__Mul_15, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Mul_15, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_up_proj_weight(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_up_proj_weight[] = {768, 256};
  VALIDATE(model.addTensor("up_proj_weight", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "up_proj_weight",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 2,
                                 .dimensions=dimensions_up_proj_weight,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(up_proj_weight),
                                                .dataSize=BINLEN(up_proj_weight)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__MatMul_7(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _MatMul_7 */
  const char*  inputs__MatMul_7[] = {
    "_MatMul_7_pre_reshape",
    "up_proj_weight"
  };
  uint32_t dimensions__MatMul_7_output_0_fc[] = {32, 768};
  Qnn_Tensor_t outputs__MatMul_7[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_7_output_0_fc",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__MatMul_7_output_0_fc,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_MatMul_7", // Node Name
                         "qti.aisw", // Package Name
                         "FullyConnected", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__MatMul_7, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__MatMul_7, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__MatMul_7_post_reshape(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _MatMul_7_post_reshape */
  const char*  inputs__MatMul_7_post_reshape[] = {
    "_MatMul_7_output_0_fc"
  };
  uint32_t dimensions__MatMul_7_output_0[] = {1, 32, 768};
  Qnn_Tensor_t outputs__MatMul_7_post_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_7_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__MatMul_7_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_MatMul_7_post_reshape", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__MatMul_7_post_reshape, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__MatMul_7_post_reshape, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Mul_16(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Mul_16 */
  Qnn_Param_t params__Mul_16[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 13}}}}
  };
  const char*  inputs__Mul_16[] = {
    "_Mul_15_output_0",
    "_MatMul_7_output_0"
  };
  uint32_t dimensions__Mul_16_output_0[] = {1, 32, 768};
  Qnn_Tensor_t outputs__Mul_16[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Mul_16_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__Mul_16_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Mul_16", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Mul_16, // Node Params
                         1, // Num Node Params
                         inputs__Mul_16, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Mul_16, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__MatMul_8_pre_reshape(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _MatMul_8_pre_reshape */
  const char*  inputs__MatMul_8_pre_reshape[] = {
    "_Mul_16_output_0"
  };
  uint32_t dimensions__MatMul_8_pre_reshape[] = {32, 768};
  Qnn_Tensor_t outputs__MatMul_8_pre_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_8_pre_reshape",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__MatMul_8_pre_reshape,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_MatMul_8_pre_reshape", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__MatMul_8_pre_reshape, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__MatMul_8_pre_reshape, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_down_proj_weight(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_down_proj_weight[] = {256, 768};
  VALIDATE(model.addTensor("down_proj_weight", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "down_proj_weight",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 2,
                                 .dimensions=dimensions_down_proj_weight,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(down_proj_weight),
                                                .dataSize=BINLEN(down_proj_weight)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__MatMul_8(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _MatMul_8 */
  const char*  inputs__MatMul_8[] = {
    "_MatMul_8_pre_reshape",
    "down_proj_weight"
  };
  uint32_t dimensions__MatMul_8_output_0_fc[] = {32, 256};
  Qnn_Tensor_t outputs__MatMul_8[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_8_output_0_fc",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__MatMul_8_output_0_fc,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_MatMul_8", // Node Name
                         "qti.aisw", // Package Name
                         "FullyConnected", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__MatMul_8, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__MatMul_8, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__MatMul_8_post_reshape(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _MatMul_8_post_reshape */
  const char*  inputs__MatMul_8_post_reshape[] = {
    "_MatMul_8_output_0_fc"
  };
  uint32_t dimensions__MatMul_8_output_0[] = {1, 32, 256};
  Qnn_Tensor_t outputs__MatMul_8_post_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_8_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__MatMul_8_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_MatMul_8_post_reshape", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__MatMul_8_post_reshape, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__MatMul_8_post_reshape, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__MatMul_8_output_0_nfc(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _MatMul_8_output_0_nfc */
  uint32_t dimensions__MatMul_8_output_0_nfc_perm[] = {3};
  uint32_t _MatMul_8_output_0_nfc_perm[] = {0, 2, 1};
  Qnn_Param_t params__MatMul_8_output_0_nfc[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_8_output_0_nfc_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__MatMul_8_output_0_nfc_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_MatMul_8_output_0_nfc_perm,
                           .dataSize=12}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__MatMul_8_output_0_nfc[] = {
    "_MatMul_8_output_0"
  };
  uint32_t dimensions__MatMul_8_output_0_nfc[] = {1, 256, 32};
  Qnn_Tensor_t outputs__MatMul_8_output_0_nfc[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_8_output_0_nfc",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__MatMul_8_output_0_nfc,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_MatMul_8_output_0_nfc", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__MatMul_8_output_0_nfc, // Node Params
                         1, // Num Node Params
                         inputs__MatMul_8_output_0_nfc, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__MatMul_8_output_0_nfc, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Add_8(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Add_8 */
  Qnn_Param_t params__Add_8[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__Add_8[] = {
    "_Add_6_output_0",
    "_MatMul_8_output_0_nfc"
  };
  uint32_t dimensions_hidden_out_nfc[] = {1, 256, 32};
  Qnn_Tensor_t outputs__Add_8[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "hidden_out_nfc",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions_hidden_out_nfc,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Add_8", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Add_8, // Node Params
                         1, // Num Node Params
                         inputs__Add_8, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Add_8, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_hidden_out_ncf(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR hidden_out_ncf */
  uint32_t dimensions_hidden_out_ncf_perm[] = {3};
  uint32_t hidden_out_ncf_perm[] = {0, 2, 1};
  Qnn_Param_t params_hidden_out_ncf[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "hidden_out_ncf_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_hidden_out_ncf_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)hidden_out_ncf_perm,
                           .dataSize=12}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs_hidden_out_ncf[] = {
    "hidden_out_nfc"
  };
  uint32_t dimensions_hidden_out[] = {1, 32, 256};
  Qnn_Tensor_t outputs_hidden_out_ncf[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "hidden_out",
            .type= QNN_TENSOR_TYPE_APP_READ,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions_hidden_out,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "hidden_out_ncf", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params_hidden_out_ncf, // Node Params
                         1, // Num Node Params
                         inputs_hidden_out_ncf, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_hidden_out_ncf, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

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

  /* model/graph for tiny_block_prefill*/
  QnnModel tiny_block_prefill;
  const QnnGraph_Config_t** graphConfigs = nullptr;
  VALIDATE(getQnnGraphConfigFromInfo("tiny_block_prefill", graphsConfigInfo, numGraphsConfigInfo, graphConfigs), err);
  VALIDATE(tiny_block_prefill.initialize(backendHandle, interface, contextHandle, "tiny_block_prefill", debug, DO_GRAPH_NODE_VALIDATIONS, graphConfigs), err);
  VALIDATE(addTensor_hidden_states(tiny_block_prefill), err);
  VALIDATE(addNode_hidden_states_nfc(tiny_block_prefill), err);
  VALIDATE(addTensor_input_norm_weight(tiny_block_prefill), err);
  VALIDATE(addTensor_input_norm_weight_bias(tiny_block_prefill), err);
  VALIDATE(addNode_rms_norm__Mul_2_output_0(tiny_block_prefill), err);
  VALIDATE(addNode__MatMul_2_pre_reshape(tiny_block_prefill), err);
  VALIDATE(addNode__MatMul_1_pre_reshape(tiny_block_prefill), err);
  VALIDATE(addNode__MatMul_pre_reshape(tiny_block_prefill), err);
  VALIDATE(addTensor_q_proj_weight(tiny_block_prefill), err);
  VALIDATE(addTensor_q_proj_bias(tiny_block_prefill), err);
  VALIDATE(addNode__MatMul(tiny_block_prefill), err);
  VALIDATE(addNode__MatMul_post_reshape(tiny_block_prefill), err);
  VALIDATE(addNode__Transpose(tiny_block_prefill), err);
  VALIDATE(addTensor_k_proj_weight(tiny_block_prefill), err);
  VALIDATE(addTensor_k_proj_bias(tiny_block_prefill), err);
  VALIDATE(addNode__MatMul_1(tiny_block_prefill), err);
  VALIDATE(addNode__MatMul_1_post_reshape(tiny_block_prefill), err);
  VALIDATE(addNode__Transpose_1(tiny_block_prefill), err);
  VALIDATE(addTensor_v_proj_weight(tiny_block_prefill), err);
  VALIDATE(addTensor_v_proj_bias(tiny_block_prefill), err);
  VALIDATE(addNode__MatMul_2(tiny_block_prefill), err);
  VALIDATE(addNode__MatMul_2_post_reshape(tiny_block_prefill), err);
  VALIDATE(addNode__Transpose_2(tiny_block_prefill), err);
  VALIDATE(addNode__Slice(tiny_block_prefill), err);
  VALIDATE(addNode__Slice_1(tiny_block_prefill), err);
  VALIDATE(addTensor__Unsqueeze_1_output_0(tiny_block_prefill), err);
  VALIDATE(addNode__Mul_3(tiny_block_prefill), err);
  VALIDATE(addTensor__Unsqueeze_3_output_0(tiny_block_prefill), err);
  VALIDATE(addNode__Mul_4(tiny_block_prefill), err);
  VALIDATE(addNode__Sub(tiny_block_prefill), err);
  VALIDATE(addNode__Mul_5(tiny_block_prefill), err);
  VALIDATE(addNode__Mul_6(tiny_block_prefill), err);
  VALIDATE(addNode__Add_4(tiny_block_prefill), err);
  VALIDATE(addNode__Concat(tiny_block_prefill), err);
  VALIDATE(addNode__Slice_2(tiny_block_prefill), err);
  VALIDATE(addNode__Slice_3(tiny_block_prefill), err);
  VALIDATE(addNode__Mul_7(tiny_block_prefill), err);
  VALIDATE(addNode__Mul_8(tiny_block_prefill), err);
  VALIDATE(addNode__Sub_1(tiny_block_prefill), err);
  VALIDATE(addNode__Mul_9(tiny_block_prefill), err);
  VALIDATE(addNode__Mul_10(tiny_block_prefill), err);
  VALIDATE(addNode__Add_5(tiny_block_prefill), err);
  VALIDATE(addNode__Concat_1(tiny_block_prefill), err);
  VALIDATE(addNode__Unsqueeze_4(tiny_block_prefill), err);
  VALIDATE(addNode__Tile(tiny_block_prefill), err);
  VALIDATE(addNode__Reshape_3(tiny_block_prefill), err);
  VALIDATE(addNode__Unsqueeze_5(tiny_block_prefill), err);
  VALIDATE(addNode__Tile_1(tiny_block_prefill), err);
  VALIDATE(addNode__Reshape_4(tiny_block_prefill), err);
  VALIDATE(addNode__Transpose_3(tiny_block_prefill), err);
  VALIDATE(addNode__MatMul_3(tiny_block_prefill), err);
  VALIDATE(addTensor__Constant_44_output_0(tiny_block_prefill), err);
  VALIDATE(addNode__Mul_11(tiny_block_prefill), err);
  VALIDATE(addTensor__Unsqueeze_7_output_0(tiny_block_prefill), err);
  VALIDATE(addNode__Unsqueeze_7_output_0_cast_to_bool(tiny_block_prefill), err);
  VALIDATE(addTensor__Constant_49_output_0(tiny_block_prefill), err);
  VALIDATE(addNode__Where(tiny_block_prefill), err);
  VALIDATE(addNode__Softmax(tiny_block_prefill), err);
  VALIDATE(addNode__MatMul_4(tiny_block_prefill), err);
  VALIDATE(addNode__Transpose_4(tiny_block_prefill), err);
  VALIDATE(addNode__Reshape_5(tiny_block_prefill), err);
  VALIDATE(addTensor_o_proj_weight(tiny_block_prefill), err);
  VALIDATE(addNode__MatMul_5(tiny_block_prefill), err);
  VALIDATE(addNode__MatMul_5_post_reshape(tiny_block_prefill), err);
  VALIDATE(addNode__MatMul_5_output_0_nfc(tiny_block_prefill), err);
  VALIDATE(addNode__Add_6(tiny_block_prefill), err);
  VALIDATE(addNode__Add_6_output_0_ncf(tiny_block_prefill), err);
  VALIDATE(addTensor_post_attention_norm_weight(tiny_block_prefill), err);
  VALIDATE(addTensor_post_attention_norm_weight_bias(tiny_block_prefill), err);
  VALIDATE(addNode_rms_norm__Mul_14_output_0(tiny_block_prefill), err);
  VALIDATE(addNode__MatMul_7_pre_reshape(tiny_block_prefill), err);
  VALIDATE(addNode__MatMul_6_pre_reshape(tiny_block_prefill), err);
  VALIDATE(addTensor_gate_proj_weight(tiny_block_prefill), err);
  VALIDATE(addNode__MatMul_6(tiny_block_prefill), err);
  VALIDATE(addNode__MatMul_6_post_reshape(tiny_block_prefill), err);
  VALIDATE(addNode__Sigmoid(tiny_block_prefill), err);
  VALIDATE(addNode__Mul_15(tiny_block_prefill), err);
  VALIDATE(addTensor_up_proj_weight(tiny_block_prefill), err);
  VALIDATE(addNode__MatMul_7(tiny_block_prefill), err);
  VALIDATE(addNode__MatMul_7_post_reshape(tiny_block_prefill), err);
  VALIDATE(addNode__Mul_16(tiny_block_prefill), err);
  VALIDATE(addNode__MatMul_8_pre_reshape(tiny_block_prefill), err);
  VALIDATE(addTensor_down_proj_weight(tiny_block_prefill), err);
  VALIDATE(addNode__MatMul_8(tiny_block_prefill), err);
  VALIDATE(addNode__MatMul_8_post_reshape(tiny_block_prefill), err);
  VALIDATE(addNode__MatMul_8_output_0_nfc(tiny_block_prefill), err);
  VALIDATE(addNode__Add_8(tiny_block_prefill), err);
  VALIDATE(addNode_hidden_out_ncf(tiny_block_prefill), err);

  // Add all models to array to get graphsInfo
  QnnModel* models [] = {&tiny_block_prefill};
  uint32_t numModels = 1;

  // Populate the constructed graphs in provided output variables
  VALIDATE(getGraphInfoFromModels(*models, numModels, graphsInfo), err);
  *numGraphsInfo = numModels;

  return err;

} // PREPARE_GRAPHS

QNN_API
ModelError_t QnnModel_freeGraphsInfo(GraphInfoPtr_t** graphsInfo, uint32_t numGraphsInfo){
  return qnn_wrapper_api::freeGraphsInfo(graphsInfo, numGraphsInfo);
} // FREEGRAPHINFO

}