#!/system/bin/sh

BENCH="${BENCH:-/data/local/tmp/llama-opencl/llama-bench}"
MODEL_DIR="${MODEL_DIR:-/sdcard/Download/models}"
LOG="${LOG:-/data/local/tmp/llama-opencl/opencl_ngl_sweep.log}"

PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
GEN_TOKENS="${GEN_TOKENS:-128}"
THREADS="${THREADS:-4}"
NGLS="${NGLS:-0 8 16 24 32 99}"
MODELS="${MODELS:-qwen2.5-3b-instruct-q4_0.gguf qwen2.5-3b-instruct-q6_k.gguf qwen2.5-3b-instruct-q8_0.gguf}"

OPENCL_LIB_PATH="${OPENCL_LIB_PATH:-/vendor/lib64:/system/vendor/lib64:/system/lib64:/data/local/tmp/llama-opencl}"

mkdir -p "$(dirname "$LOG")"
: > "$LOG"

log() {
  echo "$@" | tee -a "$LOG"
}

if [ ! -x "$BENCH" ]; then
  log "ERROR: llama-bench not found or not executable: $BENCH"
  log "Set BENCH=/path/to/llama-bench if your OpenCL build is in another directory."
  exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
  log "ERROR: model directory not found: $MODEL_DIR"
  exit 1
fi

NGL_COUNT=0
for ngl in $NGLS; do
  NGL_COUNT=$((NGL_COUNT + 1))
done

MODEL_COUNT=0
FOUND_MODELS=""
MISSING_MODELS=""

for model_name in $MODELS; do
  model_path="$MODEL_DIR/$model_name"
  if [ -f "$model_path" ]; then
    MODEL_COUNT=$((MODEL_COUNT + 1))
    FOUND_MODELS="$FOUND_MODELS $model_name"
  else
    MISSING_MODELS="$MISSING_MODELS $model_name"
  fi
done

if [ "$MODEL_COUNT" -eq 0 ]; then
  log "ERROR: none of the selected models were found in: $MODEL_DIR"
  log "selected models: $MODELS"
  exit 1
fi

TOTAL=$((MODEL_COUNT * NGL_COUNT))
COUNT=0

log "=== OpenCL ngl sweep started ==="
log "time: $(date)"
log "bench: $BENCH"
log "model_dir: $MODEL_DIR"
log "log: $LOG"
log "prompt tokens: $PROMPT_TOKENS"
log "generation tokens: $GEN_TOKENS"
log "threads: $THREADS"
log "ngl list: $NGLS"
log "selected models:$FOUND_MODELS"
if [ -n "$MISSING_MODELS" ]; then
  log "missing models:$MISSING_MODELS"
fi
log "opencl library path: $OPENCL_LIB_PATH"
log ""

for model_name in $MODELS; do
  model="$MODEL_DIR/$model_name"
  if [ ! -f "$model" ]; then
    continue
  fi
  model_name="$(basename "$model")"

  for ngl in $NGLS; do
    COUNT=$((COUNT + 1))

    log "=== [$COUNT/$TOTAL] MODEL: $model_name NGL: $ngl ==="
    log "CMD: LD_LIBRARY_PATH=$OPENCL_LIB_PATH:\$LD_LIBRARY_PATH $BENCH -m $model -p $PROMPT_TOKENS -n $GEN_TOKENS -t $THREADS -ngl $ngl"
    log "start time: $(date)"

    LD_LIBRARY_PATH="$OPENCL_LIB_PATH:$LD_LIBRARY_PATH" \
      "$BENCH" \
        -m "$model" \
        -p "$PROMPT_TOKENS" \
        -n "$GEN_TOKENS" \
        -t "$THREADS" \
        -ngl "$ngl" 2>&1 | tee -a "$LOG"

    log "end time: $(date)"
    log "=== [$COUNT/$TOTAL] done: MODEL: $model_name NGL: $ngl ==="
    log ""
  done
done

log "=== all completed ==="
log "time: $(date)"
log "log saved to: $LOG"
