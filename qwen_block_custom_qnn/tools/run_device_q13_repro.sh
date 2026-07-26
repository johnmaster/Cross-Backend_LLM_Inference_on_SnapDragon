#!/usr/bin/env bash
# Reproduce device-exact q_proj RHS conversion, embedding and Qwen profiling.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QWEN_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${QWEN_DIR}/.." && pwd)"
QAIRT_ROOT="${QAIRT_ROOT:-/home/lingbok/Qualcomm/qairt/2.47.0.260601}"
HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT:-/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0}"
HEXAGON_TOOLS_VERSION="${HEXAGON_TOOLS_VERSION:-8.7.06}"
ANDROID_NDK_ROOT="${ANDROID_NDK_ROOT:-/home/lingbok/android/android-ndk-r28}"
DEVICE_QNN_ROOT="${DEVICE_QNN_ROOT:-/data/local/tmp/qnn}"
NUM_INFERENCES="${NUM_INFERENCES:-10}"

PACKAGE_DIR="${REPO_ROOT}/qnn_custom_ops/matmul_qhpi_hvx_8row_fp32_store_multithread/htp/MatMulQhpiHvx8RowFp32StoreMultithreadOpPackage"
PACKAGE_NAME="QnnMatMulQhpiHvx8RowFp32StoreMultithreadOpPackage"
PROVIDER="MatMulQhpiHvx8RowFp32StoreMultithreadOpPackageInterfaceProvider"
DEVICE_CASE="${DEVICE_QNN_ROOT}/qwen_block_custom_qnn"
PROBE_RESULT="${QWEN_DIR}/device_output/device_q13_conversion_probe"
FINAL_RESULT="${QWEN_DIR}/device_output/q_proj_device_q13_4x128_slim_layer0_prefill_seq16"
DEVICE_Q13="${PROBE_RESULT}/Result_0/weight_q13_native.raw"
DEVICE_BIN="${QWEN_DIR}/generated/qwen2_0_5b_layer0_prefill_seq16_q_proj_device_q13.bin"
DEVICE_JSON="${QWEN_DIR}/generated/qwen2_0_5b_layer0_prefill_seq16_q_proj_device_q13.json"
PATCHED_CPP="${QWEN_DIR}/generated/qwen2_0_5b_layer0_prefill_seq16_q_proj_offline_q13.cpp"
MODEL_NAME="qwen2_0_5b_layer0_prefill_seq16_q_proj_device_q13"
MODEL_SO="${QWEN_DIR}/model_libs/aarch64-android/lib${MODEL_NAME}.so"
ARM_PACKAGE="${PACKAGE_DIR}/build/aarch64-android/lib${PACKAGE_NAME}.so"
HTP_PACKAGE="${PACKAGE_DIR}/build/hexagon-v75/lib${PACKAGE_NAME}.so"
MODEL_GENERATOR="${QAIRT_ROOT}/bin/x86_64-linux-clang/qnn-model-lib-generator"
PROFILE_VIEWER="${QAIRT_ROOT}/bin/x86_64-linux-clang/qnn-profile-viewer"

echo "[1/8] Build converter/custom OpPackage"
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

echo "[2/8] Build probe model and extract original FP32 q_proj weight"
PATH="${ANDROID_NDK_ROOT}:${PATH}" "${MODEL_GENERATOR}" \
  -c "${SCRIPT_DIR}/device_q13_probe_model.cpp" -t aarch64-android \
  -l device_q13_probe_model -o "${QWEN_DIR}/model_libs"
python3 "${SCRIPT_DIR}/generate_q_proj_q13_model_bin.py"

echo "[3/8] Run native device conversion probe"
adb shell mkdir -p "${DEVICE_CASE}/lib" "${DEVICE_CASE}/input" \
  "${DEVICE_QNN_ROOT}/dsp"
adb push "${QWEN_DIR}/model_libs/aarch64-android/libdevice_q13_probe_model.so" \
  "${DEVICE_CASE}/lib/"
adb push "${ARM_PACKAGE}" "${DEVICE_CASE}/lib/"
adb push "${HTP_PACKAGE}" "${DEVICE_QNN_ROOT}/dsp/"
adb push "${QWEN_DIR}/generated/q_proj_weight_fp32.raw" "${DEVICE_CASE}/input/"
adb push "${SCRIPT_DIR}/device_q13_probe_input_list.txt" "${DEVICE_CASE}/input/"
adb shell mkdir -p "${DEVICE_CASE}/input/layer0_prefill_seq16"
adb push "${QWEN_DIR}/test_data/layer0_prefill_seq16/hidden_states.raw" \
  "${DEVICE_CASE}/input/layer0_prefill_seq16/"
adb push "${SCRIPT_DIR}/device_input_list_layer0_prefill_seq16.txt" \
  "${DEVICE_CASE}/input/layer0_prefill_seq16/input_list.txt"
adb shell "cd '${DEVICE_QNN_ROOT}' && rm -rf '${DEVICE_CASE}/output_device_q13_probe' && \
  export LD_LIBRARY_PATH='${DEVICE_CASE}/lib:${DEVICE_QNN_ROOT}/lib' && \
  export ADSP_LIBRARY_PATH='${DEVICE_QNN_ROOT}/dsp' && \
  ./bin/qnn-net-run --backend lib/libQnnHtp.so \
  --model '${DEVICE_CASE}/lib/libdevice_q13_probe_model.so' \
  --input_list '${DEVICE_CASE}/input/device_q13_probe_input_list.txt' \
  --output_dir '${DEVICE_CASE}/output_device_q13_probe' \
  --op_packages '${DEVICE_CASE}/lib/lib${PACKAGE_NAME}.so:${PROVIDER}:CPU,lib${PACKAGE_NAME}.so:${PROVIDER}:HTP' \
  --output_data_type native_only"
mkdir -p "${PROBE_RESULT}"
adb pull "${DEVICE_CASE}/output_device_q13_probe/Result_0" "${PROBE_RESULT}/"

echo "[4/8] Embed device-exact INT16 and build Qwen model"
python3 "${SCRIPT_DIR}/generate_q_proj_q13_model_bin.py" \
  --device-q13 "${DEVICE_Q13}" --output "${DEVICE_BIN}" \
  --metadata "${DEVICE_JSON}" --drop-source-member
python3 "${SCRIPT_DIR}/patch_qwen_prefill_q_proj_offline_q13.py"
PATH="${ANDROID_NDK_ROOT}:${PATH}" "${MODEL_GENERATOR}" \
  -c "${PATCHED_CPP}" -b "${DEVICE_BIN}" -t aarch64-android \
  -l "${MODEL_NAME}" -o "${QWEN_DIR}/model_libs"

echo "[5/8] Deploy device-Q13 Qwen model"
adb push "${MODEL_SO}" "${DEVICE_CASE}/lib/"

echo "[6/8] Run 10-inference detailed profiling"
adb shell "cd '${DEVICE_QNN_ROOT}' && rm -rf '${DEVICE_CASE}/output_q_proj_device_q13' && \
  export LD_LIBRARY_PATH='${DEVICE_CASE}/lib:${DEVICE_QNN_ROOT}/lib' && \
  export ADSP_LIBRARY_PATH='${DEVICE_QNN_ROOT}/dsp' && \
  ./bin/qnn-net-run --backend lib/libQnnHtp.so \
  --model '${DEVICE_CASE}/lib/lib${MODEL_NAME}.so' \
  --input_list '${DEVICE_CASE}/input/layer0_prefill_seq16/input_list.txt' \
  --output_dir '${DEVICE_CASE}/output_q_proj_device_q13' \
  --op_packages '${DEVICE_CASE}/lib/lib${PACKAGE_NAME}.so:${PROVIDER}:CPU,lib${PACKAGE_NAME}.so:${PROVIDER}:HTP' \
  --perf_profile burst --profiling_level detailed \
  --num_inferences '${NUM_INFERENCES}'"

echo "[7/8] Pull and decode artifacts"
mkdir -p "${FINAL_RESULT}"
adb pull "${DEVICE_CASE}/output_q_proj_device_q13/Result_0" "${FINAL_RESULT}/"
adb pull "${DEVICE_CASE}/output_q_proj_device_q13/qnn-profiling-data_0.log" \
  "${FINAL_RESULT}/qnn-profiling-data_0.log"
"${PROFILE_VIEWER}" --input_log "${FINAL_RESULT}/qnn-profiling-data_0.log" \
  --output "${FINAL_RESULT}/profile.csv"

echo "[8/8] Verify reference accuracy, bit equality and performance"
python3 "${SCRIPT_DIR}/compare_qnn_output.py" \
  --output "${FINAL_RESULT}/Result_0"
python3 "${SCRIPT_DIR}/compare_qnn_output_pair.py" \
  "${FINAL_RESULT}/Result_0" \
  "${QWEN_DIR}/device_output/q_proj_custom_multithread_4row_layer0_prefill_seq16/Result_0"
python3 "${SCRIPT_DIR}/summarize_profile.py" "${FINAL_RESULT}/profile.csv"
