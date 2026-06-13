#!/system/bin/sh

QNN_DIR="${QNN_DIR:-/data/local/tmp/qnn}"
MODEL_NAME="${MODEL_NAME:-mobilenet_v2}"
MODEL_DIR="${MODEL_DIR:-$QNN_DIR/$MODEL_NAME}"
MODEL_BIN="${MODEL_BIN:-$MODEL_DIR/mobilenet_v2.bin}"
INPUT_LIST="${INPUT_LIST:-$MODEL_DIR/input/input_list.txt}"
LOG="${LOG:-$MODEL_DIR/qnn_throughput_all.log}"

THROUGHPUT_BIN="${THROUGHPUT_BIN:-$QNN_DIR/bin/qnn-throughput-net-run}"
LOOP_SECONDS="${LOOP_SECONDS:-10}"
LOG_LEVEL="${LOG_LEVEL:-info}"

DSP_PATH="${DSP_PATH:-$QNN_DIR/dsp;$QNN_DIR/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp}"
LIB_PATH="${LIB_PATH:-$QNN_DIR/lib:$LD_LIBRARY_PATH}"

mkdir -p "$MODEL_DIR"
: > "$LOG"

log() {
  echo "$@" | tee -a "$LOG"
}

write_config() {
  name="$1"
  backend_name="$2"
  backend_path="$3"
  context_name="$4"
  thread_name="$5"
  output_path="$6"
  extra_backend_lines="$7"
  config_path="$MODEL_DIR/throughput_${name}.json"

  cat > "$config_path" <<EOF
{
  "backends": [
    {
      "backendName": "$backend_name",
      "backendPath": "$backend_path",
      "profilingLevel": "OFF"$extra_backend_lines
    }
  ],
  "models": [
    {
      "modelName": "$MODEL_NAME",
      "modelPath": "$MODEL_BIN",
      "loadFromCachedBinary": true,
      "inputPath": "$INPUT_LIST",
      "inputDataType": "FLOAT",
      "outputPath": "$output_path",
      "outputDataType": "FLOAT_ONLY",
      "saveOutput": "NONE"
    }
  ],
  "contexts": [
    {
      "contextName": "$context_name"
    }
  ],
  "testCase": {
    "iteration": 1,
    "logLevel": "$LOG_LEVEL",
    "threads": [
      {
        "threadName": "$thread_name",
        "backend": "$backend_name",
        "context": "$context_name",
        "model": "$MODEL_NAME",
        "interval": 0,
        "loopUnit": "second",
        "loop": $LOOP_SECONDS
      }
    ]
  }
}
EOF

  echo "$config_path"
}

run_case() {
  name="$1"
  config="$2"
  tmp_out="$MODEL_DIR/throughput_${name}.tmp.log"

  log ""
  log "=== QNN throughput: $name ==="
  log "time: $(date)"
  log "config: $config"
  log "cmd: $THROUGHPUT_BIN --config $config"

  "$THROUGHPUT_BIN" --config "$config" > "$tmp_out" 2>&1
  rc=$?
  cat "$tmp_out" | tee -a "$LOG"

  log "return_code: $rc"
  log "=== done: $name ==="
}

if [ ! -x "$THROUGHPUT_BIN" ]; then
  log "ERROR: qnn-throughput-net-run not found or not executable: $THROUGHPUT_BIN"
  exit 1
fi

if [ ! -f "$MODEL_BIN" ]; then
  log "ERROR: model context binary not found: $MODEL_BIN"
  exit 1
fi

if [ ! -f "$INPUT_LIST" ]; then
  log "ERROR: input list not found: $INPUT_LIST"
  exit 1
fi

export LD_LIBRARY_PATH="$LIB_PATH"
export ADSP_LIBRARY_PATH="$DSP_PATH"

log "=== QNN throughput all started ==="
log "time: $(date)"
log "qnn_dir: $QNN_DIR"
log "model_bin: $MODEL_BIN"
log "input_list: $INPUT_LIST"
log "loop_seconds: $LOOP_SECONDS"
log "log: $LOG"
log "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
log "ADSP_LIBRARY_PATH: $ADSP_LIBRARY_PATH"

CPU_CONFIG="$(write_config \
  cpu \
  cpu_backend \
  "$QNN_DIR/lib/libQnnCpu.so" \
  cpu_context \
  cpu_thread_1 \
  "$MODEL_DIR/throughput_output_cpu" \
  "")"

GPU_CONFIG="$(write_config \
  gpu \
  gpu_backend \
  "$QNN_DIR/lib/libQnnGpu.so" \
  gpu_context \
  gpu_thread_1 \
  "$MODEL_DIR/throughput_output_gpu" \
  "")"

HTP_CONFIG="$(write_config \
  htp \
  htp_backend \
  "$QNN_DIR/lib/libQnnHtp.so" \
  htp_context \
  htp_thread_1 \
  "$MODEL_DIR/throughput_output_htp" \
  ",
      \"backendExtensions\": \"$QNN_DIR/lib/libQnnHtpNetRunExtensions.so\",
      \"perfProfile\": \"high_performance\"")"

run_case cpu "$CPU_CONFIG"
run_case gpu "$GPU_CONFIG"
run_case htp "$HTP_CONFIG"

log ""
log "=== QNN throughput all completed ==="
log "time: $(date)"
log "log saved to: $LOG"
log ""
log "NOTE: qnn_context_binary is usually backend-specific. If CPU or GPU fails while HTP works, export CPU/GPU-compatible QNN artifacts separately."
