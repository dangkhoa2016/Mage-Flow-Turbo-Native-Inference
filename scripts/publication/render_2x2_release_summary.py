from __future__ import annotations

import argparse
import json
from pathlib import Path


def _seconds(value: int | float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value / 1000:.3f} s"


def _ratio(value: int | float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}×"


def _mib(value: int | float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.0f} MiB"


def _rss_gib(kb: int | float | None) -> str:
    if kb is None:
        return "N/A"
    return f"{kb / (1024 * 1024):.2f} GiB"


def render_summary(data: dict) -> str:
    if data.get("comparability") != "passed":
        raise ValueError("cannot render a release summary from non-comparable evidence")

    lines = [
        "# v1.0.0 Strict 2x2 Native Benchmark",
        "",
        "All four evidence cells share the same frozen release source and canonical request; per-resolution status below remains authoritative if a genuine later-resolution limit was recorded.",
        "Mage-Flow-Turbo inference is executed by the pinned native `stable-diffusion.cpp` `sd-cli` runtime.",
        "The BF16 SafeTensors profile is **not** a Hugging Face Transformers inference backend; SafeTensors describes the model artifact format/source distribution.",
        "",
        f"- Source HEAD: `{data.get('source_head')}`",
        f"- Source TREE: `{data.get('source_tree')}`",
        "",
        "## Latency",
        "",
        "| Resolution | Q8 CPU | Q8 T4 | Q8 CPU→T4 | BF16 CPU | BF16 T4 | BF16 CPU→T4 | CPU BF16/Q8 | T4 BF16/Q8 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in data.get("rows") or []:
        q8 = row.get("q8") or {}
        bf16 = row.get("bf16") or {}
        cpu = row.get("cpu") or {}
        cuda0 = row.get("cuda0") or {}
        q8_cpu = q8.get("cpu") or {}
        q8_cuda = q8.get("cuda0") or {}
        bf16_cpu = bf16.get("cpu") or {}
        bf16_cuda = bf16.get("cuda0") or {}
        resolution = row.get("resolution")
        lines.append(
            "| "
            f"{resolution}×{resolution} | "
            f"{_seconds(q8_cpu.get('elapsed_ms'))} | "
            f"{_seconds(q8_cuda.get('elapsed_ms'))} | "
            f"{_ratio(q8.get('cpu_to_cuda_speedup'))} | "
            f"{_seconds(bf16_cpu.get('elapsed_ms'))} | "
            f"{_seconds(bf16_cuda.get('elapsed_ms'))} | "
            f"{_ratio(bf16.get('cpu_to_cuda_speedup'))} | "
            f"{_ratio(cpu.get('bf16_over_q8_latency_ratio'))} | "
            f"{_ratio(cuda0.get('bf16_over_q8_latency_ratio'))} |"
        )

    lines.extend(
        [
            "",
            "## Memory",
            "",
            "| Resolution | Q8 CPU peak RSS | BF16 CPU peak RSS | Q8 T4 peak VRAM | BF16 T4 peak VRAM |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in data.get("rows") or []:
        q8 = row.get("q8") or {}
        bf16 = row.get("bf16") or {}
        q8_cpu = q8.get("cpu") or {}
        q8_cuda = q8.get("cuda0") or {}
        bf16_cpu = bf16.get("cpu") or {}
        bf16_cuda = bf16.get("cuda0") or {}
        resolution = row.get("resolution")
        lines.append(
            "| "
            f"{resolution}×{resolution} | "
            f"{_rss_gib(q8_cpu.get('peak_rss_kb'))} | "
            f"{_rss_gib(bf16_cpu.get('peak_rss_kb'))} | "
            f"{_mib(q8_cuda.get('gpu_peak_mib'))} | "
            f"{_mib(bf16_cuda.get('gpu_peak_mib'))} |"
        )

    partials: list[str] = []
    for key, cell in (data.get("cells") or {}).items():
        if cell.get("status") != "passed":
            partials.append(
                f"- `{key}`: `{cell.get('status')}`; failed resolution: "
                f"`{(cell.get('failed_resolution') or {}).get('resolution')}`; "
                f"error: `{cell.get('error')}`"
            )
    if partials:
        lines.extend(["", "## Recorded limits", "", *partials])

    lines.extend(
        [
            "",
            "Ratios are emitted only when the required underlying records passed. `N/A` means the strict comparison is unavailable; it is never an inferred value.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="render-2x2-release-summary")
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    data = json.loads(Path(args.comparison).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_summary(data), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
