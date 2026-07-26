#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
qairt_root="${QAIRT_ROOT:-/home/lingbok/Qualcomm/qairt/2.47.0.260601}"
ndk_root="${ANDROID_NDK_ROOT:-/home/lingbok/android/android-ndk-r28}"
model_name="qwen2_0_5b_layer0_decode_past128_flash_attention_head_contiguous"
source_cpp="$repo_dir/qwen_block_custom_qnn/generated/${model_name}.cpp"
source_bin="$repo_dir/qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_decode_past128_grouped_gqa_delta_kv.bin"
output_dir="$repo_dir/qnn_custom_ops/qwen_flash_attention_hvx/model_libs"

python3 "$repo_dir/qnn_custom_ops/qwen_flash_attention_hvx/tools/patch_qwen_decode_head_contiguous.py"
PATH="$ndk_root:$PATH" "$qairt_root/bin/x86_64-linux-clang/qnn-model-lib-generator" \
  -c "$source_cpp" \
  -b "$source_bin" \
  -t aarch64-android \
  -l "$model_name" \
  -o "$output_dir"
