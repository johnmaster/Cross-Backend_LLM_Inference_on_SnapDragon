#!/usr/bin/env bash
#
# Reproduce the Qwen2.5-0.5B layer0 q_proj multithread + LHS tile-cache run.
#
# The script deliberately keeps generated C++, model libraries, device outputs,
# profiling CSV and numerical-comparison output. Override the environment
# variables below when SDK/device paths differ from the recorded environment.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QWEN_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${QWEN_DIR}/.." && pwd)"

QAIRT_ROOT="${QAIRT_ROOT:-/home/lingbok/Qualcomm/qairt/2.47.0.260601}"
HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT:-/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0}"
HEXAGON_TOOLS_VERSION="${HEXAGON_TOOLS_VERSION:-8.7.06}"
ANDROID_NDK_ROOT="${ANDROID_NDK_ROOT:-/home/lingbok/android/android-ndk-r28}"
DEVICE_QNN_ROOT="${DEVICE_QNN_ROOT:-/data/local/tmp/qnn}"
DEVICE_CASE_DIR="${DEVICE_CASE_DIR:-qwen_block_custom_qnn}"
NUM_INFERENCES="${NUM_INFERENCES:-10}"
RUN_TAG="${RUN_TAG:-q_proj_custom_multithread_4row_layer0_prefill_seq16_repro}"

PACKAGE_DIR="${REPO_ROOT}/qnn_custom_ops/matmul_qhpi_hvx_8row_fp32_store_multithread/htp/MatMulQhpiHvx8RowFp32StoreMultithreadOpPackage"
GENERATED_CPP="${QWEN_DIR}/generated/qwen2_0_5b_layer0_prefill_seq16_q_proj_custom_multithread_tile_cache.cpp"
MODEL_BIN="${QWEN_DIR}/generated/qwen2_0_5b_layer0_prefill_seq16.bin"
MODEL_NAME="qwen2_0_5b_layer0_prefill_seq16_q_proj_custom_multithread_tile_cache"
MODEL_SO="${QWEN_DIR}/model_libs/aarch64-android/lib${MODEL_NAME}.so"
PACKAGE_NAME="QnnMatMulQhpiHvx8RowFp32StoreMultithreadOpPackage"
PACKAGE_PROVIDER="MatMulQhpiHvx8RowFp32StoreMultithreadOpPackageInterfaceProvider"
ARM_PACKAGE_SO="${PACKAGE_DIR}/build/aarch64-android/lib${PACKAGE_NAME}.so"
HTP_PACKAGE_SO="${PACKAGE_DIR}/build/hexagon-v75/lib${PACKAGE_NAME}.so"
LOCAL_RESULT="${QWEN_DIR}/device_output/${RUN_TAG}"
DEVICE_CASE_ROOT="${DEVICE_QNN_ROOT}/${DEVICE_CASE_DIR}"
DEVICE_OUTPUT="${DEVICE_CASE_ROOT}/output_${RUN_TAG}"
INPUT_LIST="${DEVICE_CASE_ROOT}/input/layer0_prefill_seq16/input_list.txt"

QNN_MODEL_LIB_GENERATOR="${QAIRT_ROOT}/bin/x86_64-linux-clang/qnn-model-lib-generator"
QNN_PROFILE_VIEWER="${QAIRT_ROOT}/bin/x86_64-linux-clang/qnn-profile-viewer"
QNN_NET_RUN="${DEVICE_QNN_ROOT}/bin/qnn-net-run"
QNN_HTP_SO="${DEVICE_QNN_ROOT}/lib/libQnnHtp.so"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_file "${MODEL_BIN}"
require_file "${SCRIPT_DIR}/device_input_list_layer0_prefill_seq16.txt"

echo "[1/7] Regenerate the patched QNN C++ source"
python3 "${SCRIPT_DIR}/patch_qwen_prefill_q_proj_multithread_tile_cache.py"

echo "[2/7] Build the HTP and Android OpPackage libraries"
make -C "${PACKAGE_DIR}" htp_v75 \
  QNN_INCLUDE="${QAIRT_ROOT}/include/QNN" \
  HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT}" \
  HEXAGON_SDK_ROOT_V75="${HEXAGON_SDK_ROOT}" \
  HEXAGON_TOOLS_VERSION_V75="${HEXAGON_TOOLS_VERSION}"
make -C "${PACKAGE_DIR}" htp_aarch64 \
  QNN_INCLUDE="${QAIRT_ROOT}/include/QNN" \
  QNN_TARGET_LIB="${QAIRT_ROOT}/lib/aarch64-android" \
  HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT}" \
  X86_LIBNATIVE_RELEASE_DIR="${HEXAGON_SDK_ROOT}/tools/HEXAGON_Tools/${HEXAGON_TOOLS_VERSION}/Tools" \
  ANDROID_NDK_ROOT="${ANDROID_NDK_ROOT}"

echo "[3/7] Build the Android QNN model library"
PATH="${ANDROID_NDK_ROOT}:${PATH}" "${QNN_MODEL_LIB_GENERATOR}" \
  -c "${GENERATED_CPP}" \
  -b "${MODEL_BIN}" \
  -t aarch64-android \
  -l "${MODEL_NAME}" \
  -o "${QWEN_DIR}/model_libs"

require_file "${MODEL_SO}"
require_file "${ARM_PACKAGE_SO}"
require_file "${HTP_PACKAGE_SO}"

echo "[4/7] Push immutable inputs and freshly built libraries"
adb shell mkdir -p "${DEVICE_CASE_ROOT}/lib" \
  "${DEVICE_CASE_ROOT}/input/layer0_prefill_seq16" "${DEVICE_QNN_ROOT}/dsp"
adb push "${MODEL_SO}" "${DEVICE_CASE_ROOT}/lib/"
adb push "${ARM_PACKAGE_SO}" "${DEVICE_CASE_ROOT}/lib/"
adb push "${HTP_PACKAGE_SO}" "${DEVICE_QNN_ROOT}/dsp/"
adb push "${QWEN_DIR}/test_data/layer0_prefill_seq16/hidden_states.raw" \
  "${DEVICE_CASE_ROOT}/input/layer0_prefill_seq16/"
adb push "${SCRIPT_DIR}/device_input_list_layer0_prefill_seq16.txt" "${INPUT_LIST}"

echo "[5/7] Run detailed profiling on the device"
adb shell rm -rf "${DEVICE_OUTPUT}"
adb shell "cd '${DEVICE_QNN_ROOT}' && \
  export LD_LIBRARY_PATH='${DEVICE_QNN_ROOT}/lib:${DEVICE_CASE_ROOT}/lib' && \
  export ADSP_LIBRARY_PATH='${DEVICE_QNN_ROOT}/dsp' && \
  '${QNN_NET_RUN}' \
    --backend '${QNN_HTP_SO}' \
    --model '${DEVICE_CASE_ROOT}/lib/lib${MODEL_NAME}.so' \
    --input_list '${INPUT_LIST}' \
    --output_dir '${DEVICE_OUTPUT}' \
    --op_packages '${DEVICE_CASE_ROOT}/lib/lib${PACKAGE_NAME}.so:${PACKAGE_PROVIDER}:CPU,lib${PACKAGE_NAME}.so:${PACKAGE_PROVIDER}:HTP' \
    --profiling_level detailed \
    --num_inferences '${NUM_INFERENCES}'"

echo "[6/7] Pull and decode all reproducibility artifacts"
mkdir -p "${LOCAL_RESULT}"
adb pull "${DEVICE_OUTPUT}/Result_0" "${LOCAL_RESULT}/"
adb pull "${DEVICE_OUTPUT}/qnn-profiling-data_0.log" "${LOCAL_RESULT}/qnn-profiling-data_0.log"
"${QNN_PROFILE_VIEWER}" \
  --input_log "${LOCAL_RESULT}/qnn-profiling-data_0.log" \
  --output "${LOCAL_RESULT}/profile.csv"

echo "[7/7] Record numerical and performance summaries"
python3 "${SCRIPT_DIR}/compare_qnn_output.py" \
  --reference "${QWEN_DIR}/test_data/layer0_prefill_seq16" \
  --output "${LOCAL_RESULT}/Result_0" | tee "${LOCAL_RESULT}/correctness.txt"
python3 "${SCRIPT_DIR}/summarize_profile.py" \
  "${LOCAL_RESULT}/profile.csv" | tee "${LOCAL_RESULT}/summary.txt"

echo "Reproduction artifacts: ${LOCAL_RESULT}"
