//==============================================================================
// Auto Generated Code for ReluOpPackage
//==============================================================================
#include "QnnCpuOpPackage.h"
#include "CustomOpPackage.hpp"

using namespace qnn::custom;
using namespace qnn::custom::macros;

static Qnn_ErrorHandle_t ReluOpPackageInitialize(
  QnnOpPackage_GlobalInfrastructure_t globalInfrastructure) {

  QNN_CUSTOM_BE_ENSURE(!(CustomOpPackage::getIsInitialized()),QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED);

  INIT_BE_OP_PACKAGE(ReluOpPackage)

  REGISTER_PACKAGE_OP(Relu)

  // INIT_BE_PACKAGE_OPTIMIZATIONS();

  CustomOpPackage::setIsInitialized(true);

  return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t ReluOpPackageGetInfo(const QnnOpPackage_Info_t** info) {
  auto opPkg = CustomOpPackage::getInstance();

  QNN_CUSTOM_BE_ENSURE(opPkg, QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED);

  QNN_CUSTOM_BE_ENSURE_STATUS(opPkg->getPackageInfo(info));

  return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t ReluOpPackageValidateOpConfig(Qnn_OpConfig_t opConfig) {
  auto opPkg = CustomOpPackage::getInstance();

  QNN_CUSTOM_BE_ENSURE(opPkg, QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED);

  auto opRegistration = opPkg->getOpRegistration(opConfig.v1.typeName);

  QNN_CUSTOM_BE_ENSURE(opRegistration, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  QNN_CUSTOM_BE_ENSURE_STATUS(opRegistration->validateOpConfig(opConfig));

return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t ReluOpPackageCreateOpImpl(
   QnnOpPackage_GraphInfrastructure_t graphInfrastructure,
   QnnOpPackage_Node_t node,
   QnnOpPackage_OpImpl_t* opImpl) {
  auto opPkg = CustomOpPackage::getInstance();

  QNN_CUSTOM_BE_ENSURE(opPkg, QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED);

  QNN_CUSTOM_BE_ENSURE_STATUS(
    opPkg->createOpImpl(graphInfrastructure, node, opImpl));

  return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t ReluOpPackageFreeOpImpl(
   QnnCpuOpPackage_OpImpl_t* opImpl) {
  auto opPkg = CustomOpPackage::getInstance();

  QNN_CUSTOM_BE_ENSURE(opPkg, QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED);

  QNN_CUSTOM_BE_ENSURE_STATUS(opPkg->freeOpImpl(opImpl));

  return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t ReluOpPackageTerminate() {

  CustomOpPackage::destroyInstance();

  return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t ReluOpPackageLogInitialize(
QnnLog_Callback_t callback, QnnLog_Level_t maxLogLevel) {
// function should be used if at least two backends support it
// USER SHOULD NOTE THIS FUNCTION IS UNUSED BY BE

  return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t ReluOpPackageLogSetLevel(
QnnLog_Level_t maxLogLevel) {
// USER SHOULD NOTE THIS FUNCTION IS UNUSED BY CPU BE

return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t ReluOpPackageLogTerminate() {
// USER SHOULD NOTE THIS FUNCTION IS UNUSED BY CPU BE

  return QNN_SUCCESS;
}


extern "C" QNN_API Qnn_ErrorHandle_t ReluOpPackageInterfaceProvider(
   QnnOpPackage_Interface_t* interface) {
  interface->interfaceVersion.major = 1;
  interface->interfaceVersion.minor = 4;
  interface->interfaceVersion.patch = 0;
  interface->v1_4.init              = ReluOpPackageInitialize;
  interface->v1_4.terminate         = ReluOpPackageTerminate;
  interface->v1_4.getInfo           = ReluOpPackageGetInfo;
  interface->v1_4.validateOpConfig  = ReluOpPackageValidateOpConfig;
  interface->v1_4.createOpImpl     =  ReluOpPackageCreateOpImpl;
  interface->v1_4.freeOpImpl        = ReluOpPackageFreeOpImpl;
  interface->v1_4.logInitialize     = ReluOpPackageLogInitialize;
  interface->v1_4.logSetLevel       = ReluOpPackageLogSetLevel;
  interface->v1_4.logTerminate      = ReluOpPackageLogTerminate;
  return QNN_SUCCESS;
}

