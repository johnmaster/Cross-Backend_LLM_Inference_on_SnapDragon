#!/usr/bin/env bash
#
# Build and run the fixed-window past128 persistent KV-cache runner.
# The graph/context/tensors are created once. After each execution, current
# K/V are converted from [1,2,1,64] to NHWC and appended to the same sliding
# [1,128,64,2] input buffers.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QWEN_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${QWEN_DIR}/.." && pwd)"

QAIRT_ROOT="${QAIRT_ROOT:-/home/lingbok/Qualcomm/qairt/2.47.0.260601}"
ANDROID_NDK_ROOT="${ANDROID_NDK_ROOT:-/home/lingbok/android/android-ndk-r28}"
DEVICE_QNN_ROOT="${DEVICE_QNN_ROOT:-/data/local/tmp/qnn}"
STEPS="${STEPS:-100}"
SHARED_BUFFER="${SHARED_BUFFER:-1}"

APP_DIR="${REPO_ROOT}/qnn_custom_ops/tools/qnn_sample_app_profile"
APP_LOCAL="${APP_DIR}/libs/arm64-v8a/qnn-sample-app"
APP_DEVICE="${DEVICE_QNN_ROOT}/bin/qnn-persistent-kv-runner"
CASE_DEVICE="${DEVICE_QNN_ROOT}/qwen_block_custom_qnn"
MODEL_NAME="qwen2_0_5b_layer0_decode_past128_grouped_gqa_delta_kv"
MODEL_LOCAL="${QWEN_DIR}/model_libs/aarch64-android/lib${MODEL_NAME}.so"
MODEL_DEVICE="${CASE_DEVICE}/lib/lib${MODEL_NAME}.so"
INPUT_LOCAL="${QWEN_DIR}/test_data/layer0_decode_past128_grouped_gqa_delta_kv"
INPUT_DEVICE="${CASE_DEVICE}/input/layer0_decode_past128_grouped_gqa_delta_kv"
RUN_TAG="persistent_runner_layer0_decode_past128_repeat${STEPS}"
OUTPUT_DEVICE="${CASE_DEVICE}/output_${RUN_TAG}"
OUTPUT_LOCAL="${QWEN_DIR}/device_output/${RUN_TAG}"

SHARED_ARG=()
if [[ "${SHARED_BUFFER}" == "1" ]]; then
  SHARED_ARG=(--persistent_shared_buffer)
  RUN_TAG="${RUN_TAG}_shared"
  OUTPUT_DEVICE="${CASE_DEVICE}/output_${RUN_TAG}"
  OUTPUT_LOCAL="${QWEN_DIR}/device_output/${RUN_TAG}"
fi

"${ANDROID_NDK_ROOT}/ndk-build" \
  APP_ALLOW_MISSING_DEPS=true \
  APP_ABI=arm64-v8a \
  NDK_PROJECT_PATH="${APP_DIR}" \
  NDK_APPLICATION_MK="${APP_DIR}/make/Application.mk" \
  APP_BUILD_SCRIPT="${APP_DIR}/make/Android.mk" \
  QNN_SDK_ROOT="${QAIRT_ROOT}"

adb shell mkdir -p "${DEVICE_QNN_ROOT}/bin" "${CASE_DEVICE}/lib" "${INPUT_DEVICE}"
adb push "${APP_LOCAL}" "${APP_DEVICE}"
adb shell chmod 755 "${APP_DEVICE}"
adb push "${MODEL_LOCAL}" "${MODEL_DEVICE}"
adb push \
  "${INPUT_LOCAL}/hidden_states.raw" \
  "${INPUT_LOCAL}/past_key_qnn_nhwc.raw" \
  "${INPUT_LOCAL}/past_value_qnn_nhwc.raw" \
  "${INPUT_DEVICE}/"
adb push \
  "${SCRIPT_DIR}/device_input_list_layer0_decode_past128_grouped_gqa_delta_kv.txt" \
  "${INPUT_DEVICE}/input_list.txt"

adb shell rm -rf "${OUTPUT_DEVICE}"
adb shell mkdir -p "${OUTPUT_DEVICE}"
adb shell "cd '${DEVICE_QNN_ROOT}' && \
  export LD_LIBRARY_PATH='${DEVICE_QNN_ROOT}/lib:${CASE_DEVICE}/lib' && \
  export ADSP_LIBRARY_PATH='${DEVICE_QNN_ROOT}/dsp' && \
  '${APP_DEVICE}' \
    --backend '${DEVICE_QNN_ROOT}/lib/libQnnHtp.so' \
    --model '${MODEL_DEVICE}' \
    --input_list '${INPUT_DEVICE}/input_list.txt' \
    --output_dir '${OUTPUT_DEVICE}' \
    --num_inferences '${STEPS}' \
    --persistent_decode_past128 \
    ${SHARED_ARG[*]} \
    --log_level error"

mkdir -p "${OUTPUT_LOCAL}"
adb pull "${OUTPUT_DEVICE}/Result_0" "${OUTPUT_LOCAL}/"
adb pull \
  "${OUTPUT_DEVICE}/persistent_decode_timings.csv" \
  "${OUTPUT_LOCAL}/persistent_decode_timings.csv"

python3 "${SCRIPT_DIR}/compare_decode_delta_output.py" \
  --reference "${QWEN_DIR}/device_output/builtin_layer0_decode_past128_grouped_gqa_delta_kv/Result_0" \
  --output "${OUTPUT_LOCAL}/Result_0"
python3 "${SCRIPT_DIR}/summarize_persistent_runner.py" \
  "${OUTPUT_LOCAL}/persistent_decode_timings.csv" \
  --drop 1

echo "Persistent runner artifacts: ${OUTPUT_LOCAL}"
