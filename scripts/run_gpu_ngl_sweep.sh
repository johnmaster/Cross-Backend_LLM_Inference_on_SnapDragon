#!/bin/bash

BENCH="${BENCH:-$HOME/llama.cpp/build-vulkan/bin/llama-bench}"
MODEL_DIR="${MODEL_DIR:-/sdcard/Download/models}"
LOG="${LOG:-$HOME/tripath/results/gpu_ngl_sweep.log}"

PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
GEN_TOKENS="${GEN_TOKENS:-128}"
THREADS="${THREADS:-4}"

NGLS=(0 8 16 24 32 99)

mkdir -p "$(dirname "$LOG")"
: > "$LOG"

log() {
  echo "$@" | tee -a "$LOG"
}

if [ ! -x "$BENCH" ]; then
  log "ERROR: llama-bench not found or not executable: $BENCH"
  log "Set BENCH=/path/to/llama-bench if your Vulkan build is in another directory."
  exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
  log "ERROR: model directory not found: $MODEL_DIR"
  exit 1
fi

shopt -s nullglob
MODELS=("$MODEL_DIR"/*.gguf)
shopt -u nullglob

if [ ${#MODELS[@]} -eq 0 ]; then
  log "ERROR: no .gguf models found in: $MODEL_DIR"
  exit 1
fi

TOTAL=$(( ${#MODELS[@]} * ${#NGLS[@]} ))
COUNT=0

log "=== GPU ngl sweep started ==="
log "time: $(date)"
log "bench: $BENCH"
log "model_dir: $MODEL_DIR"
log "log: $LOG"
log "prompt tokens: $PROMPT_TOKENS"
log "generation tokens: $GEN_TOKENS"
log "threads: $THREADS"
log "ngl list: ${NGLS[*]}"
log ""

for model in "${MODELS[@]}"; do
  model_name="$(basename "$model")"

  for ngl in "${NGLS[@]}"; do
    COUNT=$((COUNT + 1))

    CMD=(
      "$BENCH"
      -m "$model"
      -p "$PROMPT_TOKENS"
      -n "$GEN_TOKENS"
      -t "$THREADS"
      -ngl "$ngl"
    )

    log "=== [$COUNT/$TOTAL] MODEL: $model_name NGL: $ngl ==="
    log "CMD: ${CMD[*]}"
    log "start time: $(date)"

    "${CMD[@]}" 2>&1 | tee -a "$LOG"

    log "end time: $(date)"
    log "=== [$COUNT/$TOTAL] done: MODEL: $model_name NGL: $ngl ==="
    log ""
  done
done

log "=== all completed ==="
log "time: $(date)"
log "log saved to: $LOG"
