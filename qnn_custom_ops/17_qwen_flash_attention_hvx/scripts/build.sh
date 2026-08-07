#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
package_dir="$repo_dir/qnn_custom_ops/17_qwen_flash_attention_hvx/htp/QwenFlashAttentionHvxOpPackage"
qairt_root="${QAIRT_ROOT:-/home/lingbok/Qualcomm/qairt/2.47.0.260601}"
hexagon_root="${HEXAGON_SDK_ROOT:-/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0}"
ndk_root="${ANDROID_NDK_ROOT:-/home/lingbok/android/android-ndk-r28}"
tools_root="$hexagon_root/tools/HEXAGON_Tools/8.7.06/Tools"

make -C "$package_dir" htp_v75 \
  QNN_INCLUDE="$qairt_root/include/QNN" \
  HEXAGON_SDK_ROOT="$hexagon_root" \
  HEXAGON_SDK_ROOT_V75="$hexagon_root" \
  HEXAGON_TOOLS_VERSION_V75=8.7.06 \
  X86_LIBNATIVE_RELEASE_DIR="$tools_root"

make -C "$package_dir" htp_aarch64 \
  QNN_INCLUDE="$qairt_root/include/QNN" \
  QNN_TARGET_LIB="$qairt_root/lib/aarch64-android" \
  HEXAGON_SDK_ROOT="$hexagon_root" \
  X86_LIBNATIVE_RELEASE_DIR="$tools_root" \
  ANDROID_NDK_ROOT="$ndk_root"
