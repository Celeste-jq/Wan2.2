#!/usr/bin/env python3
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze communication kernels from *_raw_overlap.csv files produced by "
            "tools/analyze_module_kernels.py."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing *_raw_overlap.csv files.",
    )
    parser.add_argument(
        "--out-dir",
        default="communication_analysis",
        help="Directory for generated communication summary CSVs.",
    )
    return parser.parse_args()


def parse_float(text):
    if text is None:
        return 0.0
    stripped = str(text).strip()
    if stripped == "":
        return 0.0
    return float(stripped)


def classify_communication_type(name):
    lowered = name.lower()
    if "allgather" in lowered or "all_gather" in lowered:
        return "all_gather"
    if "alltoall" in lowered or "all_to_all" in lowered:
        return "all_to_all"
    if "reducescatter" in lowered or "reduce_scatter" in lowered:
        return "reduce_scatter"
    if "broadcast" in lowered:
        return "broadcast"
    if "memcpy" in lowered:
        return "memcpy"
    if "send" in lowered or "recv" in lowered:
        return "send_recv"
    if "hccl" in lowered or "hcom" in lowered:
        return "hccl_other"
    return "other"


def is_communication_row(row):
    category = str(row.get("category", "")).strip().lower()
    if category == "communication":
        return True

    name = str(row.get("kernel_name_norm", "")).strip()
    lowered = name.lower()
    return any(
        token in lowered
        for token in [
            "allgather",
            "all_gather",
            "alltoall",
            "all_to_all",
            "reducescatter",
            "reduce_scatter",
            "broadcast",
            "send",
            "recv",
            "hccl",
            "hcom",
            "memcpyasync",
        ]
    )


def load_raw_overlap(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_paths = sorted(input_dir.glob("*_raw_overlap.csv"))
    if not raw_paths:
        raise FileNotFoundError(f"No *_raw_overlap.csv found in {input_dir}")

    module_summaries = []
    kernel_rows = []
    type_rows = []

    for raw_path in raw_paths:
        module_name = raw_path.name[: -len("_raw_overlap.csv")]
        rows = load_raw_overlap(raw_path)
        if not rows:
            continue

        module_total_us = sum(parse_float(row.get("overlap_us")) for row in rows)
        comm_rows = [row for row in rows if is_communication_row(row)]
        communication_total_us = sum(parse_float(row.get("overlap_us")) for row in comm_rows)

        if communication_total_us <= 0:
            continue

        module_summaries.append(
            {
                "module": module_name,
                "module_total_us": module_total_us,
                "communication_total_us": communication_total_us,
                "communication_pct": communication_total_us / module_total_us * 100.0,
                "communication_call_count": len(comm_rows),
            }
        )

        kernel_grouped = defaultdict(
            lambda: {"call_count": 0, "total_overlap_us": 0.0, "communication_type": ""}
        )
        type_grouped = defaultdict(lambda: {"call_count": 0, "total_overlap_us": 0.0})

        for row in comm_rows:
            kernel_name = str(row.get("kernel_name_norm", "")).strip()
            overlap_us = parse_float(row.get("overlap_us"))
            comm_type = classify_communication_type(kernel_name)

            kernel_slot = kernel_grouped[kernel_name]
            kernel_slot["call_count"] += 1
            kernel_slot["total_overlap_us"] += overlap_us
            kernel_slot["communication_type"] = comm_type

            type_slot = type_grouped[comm_type]
            type_slot["call_count"] += 1
            type_slot["total_overlap_us"] += overlap_us

        for kernel_name, item in kernel_grouped.items():
            kernel_rows.append(
                {
                    "module": module_name,
                    "kernel_name": kernel_name,
                    "communication_type": item["communication_type"],
                    "call_count": item["call_count"],
                    "total_overlap_us": item["total_overlap_us"],
                    "avg_overlap_us": item["total_overlap_us"] / item["call_count"],
                    "module_communication_pct": (
                        item["total_overlap_us"] / communication_total_us * 100.0
                    ),
                    "module_total_pct": item["total_overlap_us"] / module_total_us * 100.0,
                }
            )

        for comm_type, item in type_grouped.items():
            type_rows.append(
                {
                    "module": module_name,
                    "communication_type": comm_type,
                    "call_count": item["call_count"],
                    "total_overlap_us": item["total_overlap_us"],
                    "avg_overlap_us": item["total_overlap_us"] / item["call_count"],
                    "module_communication_pct": (
                        item["total_overlap_us"] / communication_total_us * 100.0
                    ),
                    "module_total_pct": item["total_overlap_us"] / module_total_us * 100.0,
                }
            )

    module_summaries.sort(key=lambda x: x["communication_total_us"], reverse=True)
    module_order = {row["module"]: index for index, row in enumerate(module_summaries)}
    kernel_rows.sort(
        key=lambda x: (
            module_order.get(x["module"], 10**9),
            -x["total_overlap_us"],
            x["kernel_name"],
        )
    )
    type_rows.sort(
        key=lambda x: (
            module_order.get(x["module"], 10**9),
            -x["total_overlap_us"],
            x["communication_type"],
        )
    )

    write_csv(
        out_dir / "communication_module_summary.csv",
        [
            "module",
            "module_total_us",
            "communication_total_us",
            "communication_pct",
            "communication_call_count",
        ],
        module_summaries,
    )
    write_csv(
        out_dir / "communication_kernel_summary.csv",
        [
            "module",
            "kernel_name",
            "communication_type",
            "call_count",
            "total_overlap_us",
            "avg_overlap_us",
            "module_communication_pct",
            "module_total_pct",
        ],
        kernel_rows,
    )
    write_csv(
        out_dir / "communication_type_summary.csv",
        [
            "module",
            "communication_type",
            "call_count",
            "total_overlap_us",
            "avg_overlap_us",
            "module_communication_pct",
            "module_total_pct",
        ],
        type_rows,
    )

    print(f"[OK] module summary: {out_dir / 'communication_module_summary.csv'}")
    print(f"[OK] kernel summary: {out_dir / 'communication_kernel_summary.csv'}")
    print(f"[OK] type summary: {out_dir / 'communication_type_summary.csv'}")


if __name__ == "__main__":
    main()
