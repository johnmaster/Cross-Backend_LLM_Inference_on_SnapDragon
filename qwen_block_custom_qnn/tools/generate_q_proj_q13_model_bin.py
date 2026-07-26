#!/usr/bin/env python3
"""Add an offline FP16-rounded Q13 q_proj weight to the QNN model tar."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from io import BytesIO
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "generated" / "qwen2_0_5b_layer0_prefill_seq16.bin"
DEFAULT_OUTPUT = (
    ROOT / "generated" / "qwen2_0_5b_layer0_prefill_seq16_q_proj_q13.bin"
)
DEFAULT_METADATA = (
    ROOT / "generated" / "qwen2_0_5b_layer0_prefill_seq16_q_proj_q13.json"
)
DEFAULT_SOURCE_OUTPUT = ROOT / "generated" / "q_proj_weight_fp32.raw"
SOURCE_MEMBER = "onnx__MatMul_227.raw"
Q13_MEMBER = "onnx__MatMul_227_q13.raw"
SHAPE = (896, 896)


def q13_from_fp32(
    payload: bytes, source_member: str = SOURCE_MEMBER,
    q13_member: str = Q13_MEMBER
) -> tuple[bytes, dict[str, object]]:
    fp32 = np.frombuffer(payload, dtype=np.float32)
    if fp32.size != int(np.prod(SHAPE)):
        raise SystemExit(f"unexpected q_proj element count: {fp32.size}")

    # Match the current graph exactly: FP32 static weight -> QNN FP16 Cast ->
    # hnnx::s16_from_hf_rnd_sat<13>.
    fp16 = fp32.astype(np.float16)
    scaled = fp16.astype(np.float32) * np.float32(8192.0)
    # hvx_mathops.h documents half-way cases as rounding away from zero.
    # Note that QAIRT 2.47 only guarantees s16_from_hf_rnd_sat for FBITS
    # -2..9. The current kernel's FBITS=13 result therefore cannot be assumed
    # to equal this mathematical Q13 conversion; this artifact is retained as
    # a diagnostic experiment, not as the adopted model.
    rounded = np.copysign(
        np.floor(np.abs(scaled) + np.float32(0.5)), scaled
    ).astype(np.int32)
    q13 = np.clip(rounded, -32768, 32767).astype(np.int16)
    output = q13.tobytes(order="C")
    metadata = {
        "source_member": source_member,
        "output_member": q13_member,
        "shape": list(SHAPE),
        "dtype": "int16",
        "fractional_bits": 13,
        "scale": 1.0 / 8192.0,
        "conversion": "float32 -> IEEE float16 -> round-to-nearest-away-from-zero Q13",
        "elements": int(q13.size),
        "bytes": len(output),
        "minimum": int(q13.min()),
        "maximum": int(q13.max()),
        "saturated_min": int(np.count_nonzero(q13 == -32768)),
        "saturated_max": int(np.count_nonzero(q13 == 32767)),
        "sha256": hashlib.sha256(output).hexdigest(),
    }
    return output, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--source-member", default=SOURCE_MEMBER)
    parser.add_argument("--q13-member", default=Q13_MEMBER)
    parser.add_argument(
        "--source-output",
        type=Path,
        default=DEFAULT_SOURCE_OUTPUT,
        help="FP32 q_proj payload used as the device conversion-probe input",
    )
    parser.add_argument(
        "--device-q13",
        type=Path,
        help="optional native INT16 payload exported by the HTP conversion probe",
    )
    parser.add_argument(
        "--drop-source-member",
        action="store_true",
        help="omit the unused FP32 q_proj member from the output model tar",
    )
    args = parser.parse_args()

    with tarfile.open(args.input, "r") as source:
        member = source.getmember(args.source_member)
        stream = source.extractfile(member)
        if stream is None:
            raise SystemExit(f"could not read {args.source_member}")
        source_payload = stream.read()
        q13_payload, metadata = q13_from_fp32(
            source_payload, args.source_member, args.q13_member
        )
        if args.device_q13 is not None:
            q13_payload = args.device_q13.read_bytes()
            if len(q13_payload) != int(np.prod(SHAPE)) * 2:
                raise SystemExit(
                    f"unexpected device Q13 byte count: {len(q13_payload)}"
                )
            device_q13 = np.frombuffer(q13_payload, dtype=np.int16)
            metadata.update(
                {
                    "conversion": (
                        "device-exported hnnx::s16_from_hf_rnd_sat<13>"
                    ),
                    "minimum": int(device_q13.min()),
                    "maximum": int(device_q13.max()),
                    "saturated_min": int(
                        np.count_nonzero(device_q13 == -32768)
                    ),
                    "saturated_max": int(
                        np.count_nonzero(device_q13 == 32767)
                    ),
                    "sha256": hashlib.sha256(q13_payload).hexdigest(),
                    "device_payload": str(args.device_q13),
                }
            )
        args.source_output.parent.mkdir(parents=True, exist_ok=True)
        args.source_output.write_bytes(source_payload)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.output, "w") as destination:
            for original in source.getmembers():
                if (
                    args.drop_source_member
                    and original.name == args.source_member
                ):
                    continue
                original_stream = source.extractfile(original)
                destination.addfile(
                    original,
                    original_stream if original.isfile() else None,
                )
            q13_info = tarfile.TarInfo(args.q13_member)
            q13_info.size = len(q13_payload)
            q13_info.mode = 0o664
            q13_info.mtime = 0
            destination.addfile(q13_info, BytesIO(q13_payload))

    metadata["source_member_in_output"] = not args.drop_source_member

    args.metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.metadata}")
    print(f"Wrote {args.source_output}")
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
