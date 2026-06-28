/* COPYRIGHT HEADER GOES HERE: No CopyRight Header String Passed During Model Conversion */

/* Command Line used:
qnn-onnx-converter; act_bitwidth=8; act_quantizer=tf; act_quantizer_calibration=min-max; act_quantizer_schema=asymmetric; adjust_nms_features_dims=True; algorithms=[]; align_matmul_ranks=True; apply_masked_softmax=uncompressed; arch_checker=False; backend=None; batch=None; bias_bitwidth=8; calc_static_encodings=False; converter_op_package_lib=; copyright_file=None; custom_io=; custom_op_config_paths=None; debug=-1; defer_loading=False; define_symbol=None; disable_batchnorm_folding=False; disable_defer_loading=False; disable_node_validation=False; disable_qnn_op_config_validation=False; disable_relu_squashing=False; dry_run=None; dumpIR=False; dump_custom_io_config_template=; dump_encoding_json=False; dump_inferred_model=False; dump_ir=; dump_ir_optimizer_config_template=False; dump_optimization_pass_mode_config=False; dump_pass_trace_info=False; dump_qairt_io_config_yaml=; dump_qairt_quantizer_command=None; dump_value_info=False; enable_framework_trace=False; enable_match_gathernd=False; enable_match_topk=False; enable_per_row_quantized_bias=False; exclude_named_tensors=False; expand_gru_op_structure=True; expand_lstm_op_structure=False; expand_sparse_op_structure=False; export_format=cpp; extract_color_transform=True; float_bias_bitwidth=0; float_bias_bw=0; float_bitwidth=32; float_bw=32; float_fallback=False; force_prune_cast_ops=False; handle_gather_negative_indices=True; ignore_encodings=False; include_data_invariant_ops=False; inject_cast_for_gather=True; input_dim=None; input_dtype=[]; input_encoding=[]; input_layout=[]; input_list=qnn_quantization/09_calibration_sensitivity/calibration/narrow/converter_input_list.txt; input_type=[]; ir_optimizer_config=; keep_disconnected_nodes=False; keep_int64_inputs=False; keep_quant_nodes=False; keep_weights_quantized=False; match_caffe_ssd_to_tf=True; model_version=None; multi_time_steps_gru=False; multi_time_steps_lstm=False; no_simplification=False; op_package_lib=; optimization_pass_mode=ir_optimizer_mainline; optimization_pass_mode_config=; out_names=['output']; overwrite_model_prefix=False; pack_4_bit_weights=False; package_name=None; packed_masked_softmax_inputs=[]; packed_max_seq=1; param_quantizer=None; param_quantizer_calibration=min-max; param_quantizer_schema=symmetric; percentile_calibration_value=99.99; perform_axes_to_spatial_first_order=True; perform_layout_transformation=False; prepare_inputs_as_params=False; preprocess_roi_pool_inputs=True; preserve_io=[]; preserve_onnx_output_order=False; quantization_overrides=qnn_quantization/06_onnx_qdq_matmul/model/quantization_overrides.json; quantizer_log=None; quantizer_log_level=LogLevel.NONE; restrict_quantization_steps=[]; squash_box_decoder=True; unroll_gru_time_steps=True; unroll_lstm_time_steps=True; use_aimet_quantizer=False; use_convert_quantization_nodes=False; use_dynamic_16_bit_weights=False; use_native_dtype=False; use_native_input_files=False; use_native_output_files=False; use_per_channel_quantization=False; use_per_row_quantization=False; use_quantize_v2=False; validate_models=False; weights_bitwidth=8
*/

#include "QnnOpDef.h"
#include "QnnModel.hpp"

// Flag to determine if Backend should do node validation for each opNode added
#define DO_GRAPH_NODE_VALIDATIONS 1

using namespace qnn_wrapper_api;
const __attribute__((visibility("default"))) char* QNN_SDK_VERSION = "qaisw-v2.47.0.260601114230";
extern "C" {
static ModelError_t addTensor_lhs(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_lhs[] = {1, 128, 256, 1};
  VALIDATE(model.addTensor("lhs", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "lhs",
                                 .type= QNN_TENSOR_TYPE_APP_WRITE,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                 .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                    QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                    {.scaleOffsetEncoding= {.scale= 0.0009803444845601916313171386718750000000f, .offset= -128}}},
                                 .rank= 4,
                                 .dimensions=dimensions_lhs,
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

static ModelError_t addNode_lhs_nchw(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR lhs_nchw */
  uint32_t dimensions_lhs_nchw_perm[] = {4};
  uint32_t lhs_nchw_perm[] = {0, 3, 1, 2};
  Qnn_Param_t params_lhs_nchw[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "lhs_nchw_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_lhs_nchw_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)lhs_nchw_perm,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs_lhs_nchw[] = {
    "lhs"
  };
  uint32_t dimensions_lhs_nchw[] = {1, 1, 128, 256};
  Qnn_Tensor_t outputs_lhs_nchw[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "lhs_nchw",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0009803444845601916313171386718750000000f, .offset= -128}}},
            .rank= 4,
            .dimensions=dimensions_lhs_nchw,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "lhs_nchw", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params_lhs_nchw, // Node Params
                         1, // Num Node Params
                         inputs_lhs_nchw, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_lhs_nchw, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_rhs_nchw(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_rhs_nchw[] = {1, 1, 256, 256};
  Qnn_ScaleOffset_t scaleOffset_rhs_nchw[] = {
    {.scale= 0.0001574803172843530774116516113281250000f, .offset= 0}, {.scale= 0.0001594807981746271252632141113281250000f, .offset= 0}, {.scale= 0.0001615066867088899016380310058593750000f, .offset= 0}, 
    {.scale= 0.0001635583175811916589736938476562500000f, .offset= 0}, {.scale= 0.0001656359963817521929740905761718750000f, .offset= 0}, {.scale= 0.0001677400869084522128105163574218750000f, .offset= 0}, 
    {.scale= 0.0001698708947515115141868591308593750000f, .offset= 0}, {.scale= 0.0001720287691568955779075622558593750000f, .offset= 0}, {.scale= 0.0001742140593705698847770690917968750000f, .offset= 0}, 
    {.scale= 0.0001764271146384999155998229980468750000f, .offset= 0}, {.scale= 0.0001786682842066511511802673339843750000f, .offset= 0}, {.scale= 0.0001809379027690738439559936523437500000f, .offset= 0}, 
    {.scale= 0.0001832363632274791598320007324218750000f, .offset= 0}, {.scale= 0.0001855640293797478079795837402343750000f, .offset= 0}, {.scale= 0.0001879212504718452692031860351562500000f, .offset= 0}, 
    {.scale= 0.0001903084339573979377746582031250000000f, .offset= 0}, {.scale= 0.0001927259290823712944984436035156250000f, .offset= 0}, {.scale= 0.0001951741287484765052795410156250000000f, .offset= 0}, 
    {.scale= 0.0001976534404093399643898010253906250000f, .offset= 0}, {.scale= 0.0002001642424147576093673706054687500000f, .offset= 0}, {.scale= 0.0002027069422183558344841003417968750000f, .offset= 0}, 
    {.scale= 0.0002052819472737610340118408203125000000f, .offset= 0}, {.scale= 0.0002078896504826843738555908203125000000f, .offset= 0}, {.scale= 0.0002105304884025827050209045410156250000f, .offset= 0}, 
    {.scale= 0.0002132048539351671934127807617187500000f, .offset= 0}, {.scale= 0.0002159132127417251467704772949218750000f, .offset= 0}, {.scale= 0.0002186559722758829593658447265625000000f, .offset= 0}, 
    {.scale= 0.0002214335545431822538375854492187500000f, .offset= 0}, {.scale= 0.0002242464397568255662918090820312500000f, .offset= 0}, {.scale= 0.0002270950644742697477340698242187500000f, .offset= 0}, 
    {.scale= 0.0002299798507010564208030700683593750000f, .offset= 0}, {.scale= 0.0002329013077542185783386230468750000000f, .offset= 0}, {.scale= 0.0002358598576392978429794311523437500000f, .offset= 0}, 
    {.scale= 0.0002388560096733272075653076171875000000f, .offset= 0}, {.scale= 0.0002418902004137635231018066406250000000f, .offset= 0}, {.scale= 0.0002449629537295550107955932617187500000f, .offset= 0}, 
    {.scale= 0.0002480747061781585216522216796875000000f, .offset= 0}, {.scale= 0.0002512260398361831903457641601562500000f, .offset= 0}, {.scale= 0.0002544173621572554111480712890625000000f, .offset= 0}, 
    {.scale= 0.0002576492261141538619995117187500000000f, .offset= 0}, {.scale= 0.0002609221846796572208404541015625000000f, .offset= 0}, {.scale= 0.0002642366744112223386764526367187500000f, .offset= 0}, 
    {.scale= 0.0002675933064892888069152832031250000000f, .offset= 0}, {.scale= 0.0002709925465751439332962036132812500000f, .offset= 0}, {.scale= 0.0002744349767453968524932861328125000000f, .offset= 0}, 
    {.scale= 0.0002779211499728262424468994140625000000f, .offset= 0}, {.scale= 0.0002814515901263803243637084960937500000f, .offset= 0}, {.scale= 0.0002850268501788377761840820312500000000f, .offset= 0}, 
    {.scale= 0.0002886475704144686460494995117187500000f, .offset= 0}, {.scale= 0.0002923143038060516119003295898437500000f, .offset= 0}, {.scale= 0.0002960275742225348949432373046875000000f, .offset= 0}, 
    {.scale= 0.0002997880219481885433197021484375000000f, .offset= 0}, {.scale= 0.0003035962581634521484375000000000000000f, .offset= 0}, {.scale= 0.0003074528358411043882369995117187500000f, .offset= 0}, 
    {.scale= 0.0003113584534730762243270874023437500000f, .offset= 0}, {.scale= 0.0003153136349283158779144287109375000000f, .offset= 0}, {.scale= 0.0003193190786987543106079101562500000000f, .offset= 0}, 
    {.scale= 0.0003233753959648311138153076171875000000f, .offset= 0}, {.scale= 0.0003274832561146467924118041992187500000f, .offset= 0}, {.scale= 0.0003316432994324713945388793945312500000f, .offset= 0}, 
    {.scale= 0.0003358561953064054250717163085937500000f, .offset= 0}, {.scale= 0.0003401225840207189321517944335937500000f, .offset= 0}, {.scale= 0.0003444431640673428773880004882812500000f, .offset= 0}, 
    {.scale= 0.0003488186339382082223892211914062500000f, .offset= 0}, {.scale= 0.0003532497212290763854980468750000000000f, .offset= 0}, {.scale= 0.0003577370371203869581222534179687500000f, .offset= 0}, 
    {.scale= 0.0003622813965193927288055419921875000000f, .offset= 0}, {.scale= 0.0003668834979180246591567993164062500000f, .offset= 0}, {.scale= 0.0003715440398082137107849121093750000000f, .offset= 0}, 
    {.scale= 0.0003762637788895517587661743164062500000f, .offset= 0}, {.scale= 0.0003810434718616306781768798828125000000f, .offset= 0}, {.scale= 0.0003858838754240423440933227539062500000f, .offset= 0}, 
    {.scale= 0.0003907857753802090883255004882812500000f, .offset= 0}, {.scale= 0.0003957499284297227859497070312500000000f, .offset= 0}, {.scale= 0.0004007771785836666822433471679687500000f, .offset= 0}, 
    {.scale= 0.0004058682534378021955490112304687500000f, .offset= 0}, {.scale= 0.0004110240261070430278778076171875000000f, .offset= 0}, {.scale= 0.0004162452823948115110397338867187500000f, .offset= 0}, 
    {.scale= 0.0004215328663121908903121948242187500000f, .offset= 0}, {.scale= 0.0004268876509740948677062988281250000000f, .offset= 0}, {.scale= 0.0004323103930801153182983398437500000000f, .offset= 0}, 
    {.scale= 0.0004378020821604877710342407226562500000f, .offset= 0}, {.scale= 0.0004433634749148041009902954101562500000f, .offset= 0}, {.scale= 0.0004489955317694693803787231445312500000f, .offset= 0}, 
    {.scale= 0.0004546991549432277679443359375000000000f, .offset= 0}, {.scale= 0.0004604752466548234224319458007812500000f, .offset= 0}, {.scale= 0.0004663246800191700458526611328125000000f, .offset= 0}, 
    {.scale= 0.0004722484154626727104187011718750000000f, .offset= 0}, {.scale= 0.0004782474134117364883422851562500000000f, .offset= 0}, {.scale= 0.0004843226051889359951019287109375000000f, .offset= 0}, 
    {.scale= 0.0004904749803245067596435546875000000000f, .offset= 0}, {.scale= 0.0004967055283486843109130859375000000000f, .offset= 0}, {.scale= 0.0005030152387917041778564453125000000000f, .offset= 0}, 
    {.scale= 0.0005094049847684800624847412109375000000f, .offset= 0}, {.scale= 0.0005158760468475520610809326171875000000f, .offset= 0}, {.scale= 0.0005224291817285120487213134765625000000f, .offset= 0}, 
    {.scale= 0.0005290656699799001216888427734375000000f, .offset= 0}, {.scale= 0.0005357863847166299819946289062500000000f, .offset= 0}, {.scale= 0.0005425925482995808124542236328125000000f, .offset= 0}, 
    {.scale= 0.0005494851502589881420135498046875000000f, .offset= 0}, {.scale= 0.0005564652965404093265533447265625000000f, .offset= 0}, {.scale= 0.0005635340348817408084869384765625000000f, .offset= 0}, 
    {.scale= 0.0005706926458515226840972900390625000000f, .offset= 0}, {.scale= 0.0005779421771876513957977294921875000000f, .offset= 0}, {.scale= 0.0005852837930433452129364013671875000000f, .offset= 0}, 
    {.scale= 0.0005927186575718224048614501953125000000f, .offset= 0}, {.scale= 0.0006002480513416230678558349609375000000f, .offset= 0}, {.scale= 0.0006078730220906436443328857421875000000f, .offset= 0}, 
    {.scale= 0.0006155948503874242305755615234375000000f, .offset= 0}, {.scale= 0.0006234147585928440093994140625000000000f, .offset= 0}, {.scale= 0.0006313340272754430770874023437500000000f, .offset= 0}, 
    {.scale= 0.0006393539370037615299224853515625000000f, .offset= 0}, {.scale= 0.0006474757101386785507202148437500000000f, .offset= 0}, {.scale= 0.0006557006272487342357635498046875000000f, .offset= 0}, 
    {.scale= 0.0006640300271101295948028564453125000000f, .offset= 0}, {.scale= 0.0006724651902914047241210937500000000000f, .offset= 0}, {.scale= 0.0006810075137764215469360351562500000000f, .offset= 0}, 
    {.scale= 0.0006896583945490419864654541015625000000f, .offset= 0}, {.scale= 0.0006984191713854670524597167968750000000f, .offset= 0}, {.scale= 0.0007072912412695586681365966796875000000f, .offset= 0}, 
    {.scale= 0.0007162760011851787567138671875000000000f, .offset= 0}, {.scale= 0.0007253748481161892414093017578125000000f, .offset= 0}, {.scale= 0.0007345893536694347858428955078125000000f, .offset= 0}, 
    {.scale= 0.0007439208566211163997650146484375000000f, .offset= 0}, {.scale= 0.0007533709867857396602630615234375000000f, .offset= 0}, {.scale= 0.0007629410247318446636199951171875000000f, .offset= 0}, 
    {.scale= 0.0007726327166892588138580322265625000000f, .offset= 0}, {.scale= 0.0007824475178495049476623535156250000000f, .offset= 0}, {.scale= 0.0007923869416117668151855468750000000000f, .offset= 0}, 
    {.scale= 0.0008024526759982109069824218750000000000f, .offset= 0}, {.scale= 0.0008126463508233428001403808593750000000f, .offset= 0}, {.scale= 0.0008229694212786853313446044921875000000f, .offset= 0}, 
    {.scale= 0.0008334236335940659046173095703125000000f, .offset= 0}, {.scale= 0.0008440106175839900970458984375000000000f, .offset= 0}, {.scale= 0.0008547321194782853126525878906250000000f, .offset= 0}, 
    {.scale= 0.0008655898855067789554595947265625000000f, .offset= 0}, {.scale= 0.0008765854872763156890869140625000000000f, .offset= 0}, {.scale= 0.0008877207874320447444915771484375000000f, .offset= 0}, 
    {.scale= 0.0008989975322037935256958007812500000000f, .offset= 0}, {.scale= 0.0009104175842367112636566162109375000000f, .offset= 0}, {.scale= 0.0009219826315529644489288330078125000000f, .offset= 0}, 
    {.scale= 0.0009336945950053632259368896484375000000f, .offset= 0}, {.scale= 0.0009455553954467177391052246093750000000f, .offset= 0}, {.scale= 0.0009575668373145163059234619140625000000f, .offset= 0}, 
    {.scale= 0.0009697308414615690708160400390625000000f, .offset= 0}, {.scale= 0.0009820493869483470916748046875000000000f, .offset= 0}, {.scale= 0.0009945243364199995994567871093750000000f, .offset= 0}, 
    {.scale= 0.0010071578435599803924560546875000000000f, .offset= 0}, {.scale= 0.0010199518874287605285644531250000000000f, .offset= 0}, {.scale= 0.0010329083306714892387390136718750000000f, .offset= 0}, 
    {.scale= 0.0010460295015946030616760253906250000000f, .offset= 0}, {.scale= 0.0010593172628432512283325195312500000000f, .offset= 0}, {.scale= 0.0010727738263085484504699707031250000000f, .offset= 0}, 
    {.scale= 0.0010864012874662876129150390625000000000f, .offset= 0}, {.scale= 0.0011002018582075834274291992187500000000f, .offset= 0}, {.scale= 0.0011141778668388724327087402343750000000f, .offset= 0}, 
    {.scale= 0.0011283312924206256866455078125000000000f, .offset= 0}, {.scale= 0.0011426645796746015548706054687500000000f, .offset= 0}, {.scale= 0.0011571798240765929222106933593750000000f, .offset= 0}, 
    {.scale= 0.0011718795867636799812316894531250000000f, .offset= 0}, {.scale= 0.0011867660796269774436950683593750000000f, .offset= 0}, {.scale= 0.0012018415145576000213623046875000000000f, .offset= 0}, 
    {.scale= 0.0012171086855232715606689453125000000000f, .offset= 0}, {.scale= 0.0012325695715844631195068359375000000000f, .offset= 0}, {.scale= 0.0012482269667088985443115234375000000000f, .offset= 0}, 
    {.scale= 0.0012640833156183362007141113281250000000f, .offset= 0}, {.scale= 0.0012801410630345344543457031250000000000f, .offset= 0}, {.scale= 0.0012964026536792516708374023437500000000f, .offset= 0}, 
    {.scale= 0.0013128708815202116966247558593750000000f, .offset= 0}, {.scale= 0.0013295484241098165512084960937500000000f, .offset= 0}, {.scale= 0.0013464377261698246002197265625000000000f, .offset= 0}, 
    {.scale= 0.0013635416980832815170288085937500000000f, .offset= 0}, {.scale= 0.0013808627845719456672668457031250000000f, .offset= 0}, {.scale= 0.0013984038960188627243041992187500000000f, .offset= 0}, 
    {.scale= 0.0014161678263917565345764160156250000000f, .offset= 0}, {.scale= 0.0014341576024889945983886718750000000000f, .offset= 0}, {.scale= 0.0014523756690323352813720703125000000000f, .offset= 0}, 
    {.scale= 0.0014708252856507897377014160156250000000f, .offset= 0}, {.scale= 0.0014895092463120818138122558593750000000f, .offset= 0}, {.scale= 0.0015084305778145790100097656250000000000f, .offset= 0}, 
    {.scale= 0.0015275923069566488265991210937500000000f, .offset= 0}, {.scale= 0.0015469973441213369369506835937500000000f, .offset= 0}, {.scale= 0.0015666489489376544952392578125000000000f, .offset= 0}, 
    {.scale= 0.0015865502646192908287048339843750000000f, .offset= 0}, {.scale= 0.0016067042015492916107177734375000000000f, .offset= 0}, {.scale= 0.0016271142521873116493225097656250000000f, .offset= 0}, 
    {.scale= 0.0016477835597470402717590332031250000000f, .offset= 0}, {.scale= 0.0016687155002728104591369628906250000000f, .offset= 0}, {.scale= 0.0016899132169783115386962890625000000000f, .offset= 0}, 
    {.scale= 0.0017113803187385201454162597656250000000f, .offset= 0}, {.scale= 0.0017331200651824474334716796875000000000f, .offset= 0}, {.scale= 0.0017551358323544263839721679687500000000f, .offset= 0}, 
    {.scale= 0.0017774314619600772857666015625000000000f, .offset= 0}, {.scale= 0.0018000103300437331199645996093750000000f, .offset= 0}, {.scale= 0.0018228759290650486946105957031250000000f, .offset= 0}, 
    {.scale= 0.0018460319843143224716186523437500000000f, .offset= 0}, {.scale= 0.0018694822210818529129028320312500000000f, .offset= 0}, {.scale= 0.0018932303646579384803771972656250000000f, .offset= 0}, 
    {.scale= 0.0019172801403328776359558105468750000000f, .offset= 0}, {.scale= 0.0019416355062276124954223632812500000000f, .offset= 0}, {.scale= 0.0019663001876324415206909179687500000000f, .offset= 0}, 
    {.scale= 0.0019912780262529850006103515625000000000f, .offset= 0}, {.scale= 0.0020165734458714723587036132812500000000f, .offset= 0}, {.scale= 0.0020421901717782020568847656250000000000f, .offset= 0}, 
    {.scale= 0.0020681321620941162109375000000000000000f, .offset= 0}, {.scale= 0.0020944038406014442443847656250000000000f, .offset= 0}, {.scale= 0.0021210089325904846191406250000000000000f, .offset= 0}, 
    {.scale= 0.0021479523275047540664672851562500000000f, .offset= 0}, {.scale= 0.0021752379834651947021484375000000000000f, .offset= 0}, {.scale= 0.0022028700914233922958374023437500000000f, .offset= 0}, 
    {.scale= 0.0022308530751615762710571289062500000000f, .offset= 0}, {.scale= 0.0022591918241232633590698242187500000000f, .offset= 0}, {.scale= 0.0022878905292600393295288085937500000000f, .offset= 0}, 
    {.scale= 0.0023169536143541336059570312500000000000f, .offset= 0}, {.scale= 0.0023463859688490629196166992187500000000f, .offset= 0}, {.scale= 0.0023761922493577003479003906250000000000f, .offset= 0}, 
    {.scale= 0.0024063773453235626220703125000000000000f, .offset= 0}, {.scale= 0.0024369454476982355117797851562500000000f, .offset= 0}, {.scale= 0.0024679021444171667098999023437500000000f, .offset= 0}, 
    {.scale= 0.0024992520920932292938232421875000000000f, .offset= 0}, {.scale= 0.0025309999473392963409423828125000000000f, .offset= 0}, {.scale= 0.0025631515309214591979980468750000000000f, .offset= 0}, 
    {.scale= 0.0025957114994525909423828125000000000000f, .offset= 0}, {.scale= 0.0026286847423762083053588867187500000000f, .offset= 0}, {.scale= 0.0026620770804584026336669921875000000000f, .offset= 0}, 
    {.scale= 0.0026958936359733343124389648437500000000f, .offset= 0}, {.scale= 0.0027301395311951637268066406250000000000f, .offset= 0}, {.scale= 0.0027648208197206258773803710937500000000f, .offset= 0}, 
    {.scale= 0.0027999423909932374954223632812500000000f, .offset= 0}, {.scale= 0.0028355102986097335815429687500000000000f, .offset= 0}, {.scale= 0.0028715298976749181747436523437500000000f, .offset= 0}, 
    {.scale= 0.0029080072417855262756347656250000000000f, .offset= 0}, {.scale= 0.0029449476860463619232177734375000000000f, .offset= 0}, {.scale= 0.0029823572840541601181030273437500000000f, .offset= 0}, 
    {.scale= 0.0030202425550669431686401367187500000000f, .offset= 0}, {.scale= 0.0030586088541895151138305664062500000000f, .offset= 0}, {.scale= 0.0030974624678492546081542968750000000000f, .offset= 0}, 
    {.scale= 0.0031368096824735403060913085937500000000f, .offset= 0}, {.scale= 0.0031766567844897508621215820312500000000f, .offset= 0}, {.scale= 0.0032170098274946212768554687500000000000f, .offset= 0}, 
    {.scale= 0.0032578757964074611663818359375000000000f, .offset= 0}, {.scale= 0.0032992607448250055313110351562500000000f, .offset= 0}, {.scale= 0.0033411714248359203338623046875000000000f, .offset= 0}, 
    {.scale= 0.0033836143556982278823852539062500000000f, .offset= 0}, {.scale= 0.0034265967551618814468383789062500000000f, .offset= 0}, {.scale= 0.0034701249096542596817016601562500000000f, .offset= 0}, 
    {.scale= 0.0035142060369253158569335937500000000000f, .offset= 0}, {.scale= 0.0035588471218943595886230468750000000000f, .offset= 0}, {.scale= 0.0036040553823113441467285156250000000000f, .offset= 0}, 
    {.scale= 0.0036498378030955791473388671875000000000f, .offset= 0}, {.scale= 0.0036962020676583051681518554687500000000f, .offset= 0}, {.scale= 0.0037431549280881881713867187500000000000f, .offset= 0}, 
    {.scale= 0.0037907045334577560424804687500000000000f, .offset= 0}, {.scale= 0.0038388581015169620513916015625000000000f, .offset= 0}, {.scale= 0.0038876233156770467758178710937500000000f, .offset= 0}, 
    {.scale= 0.0039370078593492507934570312500000000000f, .offset= 0}};
  VALIDATE(model.addTensor("rhs_nchw", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "rhs_nchw",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_SFIXED_POINT_8,
                                 .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                    QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET,
                                                    {.axisScaleOffsetEncoding= {.axis= 3, .numScaleOffsets= 256, .scaleOffset=scaleOffset_rhs_nchw}}},
                                 .rank= 4,
                                 .dimensions=dimensions_rhs_nchw,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(rhs_nchw),
                                                .dataSize=BINLEN(rhs_nchw)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode_MatMul(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR MatMul */
  Qnn_Param_t params_MatMul[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_MatMul[] = {
    "lhs_nchw",
    "rhs_nchw"
  };
  uint32_t dimensions_output[] = {1, 1, 128, 256};
  Qnn_Tensor_t outputs_MatMul[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "output",
            .type= QNN_TENSOR_TYPE_APP_READ,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0078855128958821296691894531250000000000f, .offset= -124}}},
            .rank= 4,
            .dimensions=dimensions_output,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "MatMul", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params_MatMul, // Node Params
                         2, // Num Node Params
                         inputs_MatMul, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_MatMul, // Output Tensors 
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

  /* model/graph for narrow*/
  QnnModel narrow;
  const QnnGraph_Config_t** graphConfigs = nullptr;
  VALIDATE(getQnnGraphConfigFromInfo("narrow", graphsConfigInfo, numGraphsConfigInfo, graphConfigs), err);
  VALIDATE(narrow.initialize(backendHandle, interface, contextHandle, "narrow", debug, DO_GRAPH_NODE_VALIDATIONS, graphConfigs), err);
  VALIDATE(addTensor_lhs(narrow), err);
  VALIDATE(addNode_lhs_nchw(narrow), err);
  VALIDATE(addTensor_rhs_nchw(narrow), err);
  VALIDATE(addNode_MatMul(narrow), err);

  // Add all models to array to get graphsInfo
  QnnModel* models [] = {&narrow};
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