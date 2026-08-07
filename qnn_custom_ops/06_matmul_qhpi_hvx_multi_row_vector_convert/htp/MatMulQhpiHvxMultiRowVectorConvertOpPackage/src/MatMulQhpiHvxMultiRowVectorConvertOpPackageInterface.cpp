//==============================================================================
// Auto Generated Code for MatMulQhpiHvxMultiRowVectorConvertOpPackage - QHPI Interface
//==============================================================================

#include "HTP/QnnHtpCommon.h"
#include "QnnOpPackage.h"
#include "QnnSdkBuildId.h"
#include "HTP/core/qhpi.h"
#include <array>
#include <string>


#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration for MatMulQhpiHvxMultiRowVectorConvert registration
extern void register_matmulqhpihvxmultirowvectorconvert_ops();

// External declarations for operator infos from individual operator files
extern QHPI_OpInfo_v1 matmulqhpihvxmultirowvectorconvertOpInfo;

// op package info
const char* const sg_packageName = "MatMulQhpiHvxMultiRowVectorConvertOpPackage";  // package name passed in as compile flag

static std::array<const char*, 1> sg_opNames{{"MatMulQhpiHvxMultiRowVectorConvert"}};

static Qnn_ApiVersion_t sg_sdkApiVersion  = QNN_HTP_API_VERSION_INIT;
static QnnOpPackage_Info_t sg_packageInfo = QNN_OP_PACKAGE_INFO_INIT;

// global data
static QnnOpPackage_GlobalInfrastructure_t sg_globalInfra =
nullptr;  // global infrastructure not in use for now
static bool sg_packageInitialized = false;

/*
 * user provided logging call back function
 * currently only supported on linux x86-64 and nonrpc versions
 * typedef void (*QnnLog_Callback_t)(const char* fmt,
 *                                   QnnLog_Level_t level,
 *                                   uint64_t timestamp,
 *                                   va_list args);
 * usage: if(sg_logInitialized && level <= sg_maxLogLevel)
 *            sg_logCallback(fmt, level, timestamp, args);
 *
 * for cross rpc versions, skel side user provided logging call back function
 * can be defined as part of op packages. maximal log level sg_maxLogLevel
 * can be set by Qnn_ErrorHandle_t MatMulQhpiHvxMultiRowVectorConvertOpPackageLogSetLevel(QnnLog_Level_t maxLogLevel)
 */
/*
 * for alternative logging method provided by HTP core, please refer to log.h
 */
static QnnLog_Callback_t sg_logCallback =
    nullptr;  // user provided call back function pointer for logging
static QnnLog_Level_t sg_maxLogLevel =
    (QnnLog_Level_t)0;  // maximal log level used in user provided logging
static bool sg_logInitialized =
    false;  // tracks whether user provided logging method has been initialized

/* op package API's */

Qnn_ErrorHandle_t MatMulQhpiHvxMultiRowVectorConvertOpPackageInit(QnnOpPackage_GlobalInfrastructure_t infrastructure) {
    if (sg_packageInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;

    /*
     * QHPI packages don't use traditional DEF_OP registration macros
     * Plugin registration is handled through  qhpi_register_ops_vxx in the source files
     */

    sg_globalInfra        = infrastructure;
    sg_packageInitialized = true;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t MatMulQhpiHvxMultiRowVectorConvertOpPackageGetInfo(const QnnOpPackage_Info_t** info) {
    if (!sg_packageInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    if (!info) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;

    sg_packageInfo                = QNN_OP_PACKAGE_INFO_INIT;
    sg_packageInfo.packageName    = sg_packageName;
    sg_packageInfo.operationNames = sg_opNames.data();
    sg_packageInfo.numOperations  = sg_opNames.size();
    sg_packageInfo.sdkBuildId     = QNN_SDK_BUILD_ID;
    sg_packageInfo.sdkApiVersion  = &sg_sdkApiVersion;

    *info = &sg_packageInfo;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t MatMulQhpiHvxMultiRowVectorConvertOpPackageLogInitialize(QnnLog_Callback_t callback, QnnLog_Level_t maxLogLevel) {
    if (sg_logInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
    if (!callback) return QNN_LOG_ERROR_INVALID_ARGUMENT;
    if (maxLogLevel < QNN_LOG_LEVEL_ERROR) return QNN_LOG_ERROR_INVALID_ARGUMENT;
    sg_logCallback    = callback;
    sg_maxLogLevel    = maxLogLevel;
    sg_logInitialized = true;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t MatMulQhpiHvxMultiRowVectorConvertOpPackageLogSetLevel(QnnLog_Level_t maxLogLevel) {
    if (maxLogLevel < QNN_LOG_LEVEL_ERROR) return QNN_LOG_ERROR_INVALID_ARGUMENT;
    sg_maxLogLevel = maxLogLevel;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t MatMulQhpiHvxMultiRowVectorConvertOpPackageLogTerminate() {
    if (!sg_logInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    sg_logCallback    = nullptr;
    sg_maxLogLevel    = (QnnLog_Level_t)0;
    sg_logInitialized = false;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t MatMulQhpiHvxMultiRowVectorConvertOpPackageValidateOpConfig (Qnn_OpConfig_t opConfig){
    if (std::string(sg_packageName) != opConfig.v1.packageName) {
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }

    /* auto-generated validation code below
     * Check if op config type matches any registered ops
     * If a match is found, check number of inputs, outputs and params
     */
    if (std::string(opConfig.v1.typeName) == "MatMulQhpiHvxMultiRowVectorConvert"){
        if (opConfig.v1.numOfParams != 0 || opConfig.v1.numOfInputs != 2 || opConfig.v1.numOfOutputs != 1){
          return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
        }
    }
    else{
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }

    /*
    * additional validation code here
    * */

    return QNN_SUCCESS;
}

/* The following three functions in this comment are not called by HTP backend for now,
 * no auto-generated implementations are created. Users should see example for full function signatures.
 * (version 1.3.0) Qnn_ErrorHandle_t MatMulQhpiHvxMultiRowVectorConvertOpPackageCreateKernels (QnnOpPackage_GraphInfrastructure_t
 * graphInfrastructure, QnnOpPackage_Node_t node, QnnOpPackage_Kernel_t** kernels, uint32_t*
 * numKernels)
 * (version 1.3.0) Qnn_ErrorHandle_t MatMulQhpiHvxMultiRowVectorConvertOpPackageFreeKernels (QnnOpPackage_Kernel_t* kernels)
 *
 * (version 1.4.0) Qnn_ErrorHandle_t MatMulQhpiHvxMultiRowVectorConvertOpPackageCreateOpImpl (QnnOpPackage_GraphInfrastructure_t
 * graphInfrastructure, QnnOpPackage_Node_t node, QnnOpPackage_OpImpl_t* opImpl)
 *(version 1.4.0) Qnn_ErrorHandle_t MatMulQhpiHvxMultiRowVectorConvertOpPackageFreeOpImpl (QnnOpPackage_OpImpl_t opImpl)
 */

Qnn_ErrorHandle_t MatMulQhpiHvxMultiRowVectorConvertOpPackageTerminate() {
if (!sg_packageInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;

sg_globalInfra        = nullptr;
sg_packageInitialized = false;
return QNN_SUCCESS;
}



/* latest version */
Qnn_ErrorHandle_t MatMulQhpiHvxMultiRowVectorConvertOpPackageInterfaceProvider(QnnOpPackage_Interface_t* interface) {
  if (!interface) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
  interface->interfaceVersion      = {1, 4, 0};
  interface->v1_4.init             = MatMulQhpiHvxMultiRowVectorConvertOpPackageInit;
  interface->v1_4.terminate        = MatMulQhpiHvxMultiRowVectorConvertOpPackageTerminate;
  interface->v1_4.getInfo          = MatMulQhpiHvxMultiRowVectorConvertOpPackageGetInfo;
  interface->v1_4.validateOpConfig = MatMulQhpiHvxMultiRowVectorConvertOpPackageValidateOpConfig;
  interface->v1_4.createOpImpl     = nullptr;
  interface->v1_4.freeOpImpl       = nullptr;
  interface->v1_4.logInitialize    = MatMulQhpiHvxMultiRowVectorConvertOpPackageLogInitialize;
  interface->v1_4.logSetLevel      = MatMulQhpiHvxMultiRowVectorConvertOpPackageLogSetLevel;
  interface->v1_4.logTerminate     = MatMulQhpiHvxMultiRowVectorConvertOpPackageLogTerminate;
  return QNN_SUCCESS;
}

// Implementation of qhpi_init function
const char* qhpi_init() {
    register_matmulqhpihvxmultirowvectorconvert_ops();
    return sg_packageName;
}
#ifdef __cplusplus
}
#endif

