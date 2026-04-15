#!/usr/bin/env python3
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze module-level kernels from MindStudio/Ascend kernel_detail.csv "
            "using time ranges exported from the timeline."
        )
    )
    parser.add_argument(
        "--kernel-detail",
        required=True,
        help="Path to kernel_detail.csv exported from profiling data.",
    )
    parser.add_argument(
        "--ranges",
        help="Path to module_ranges.json. Required unless --show-columns is used.",
    )
    parser.add_argument(
        "--out-dir",
        default="module_kernel_analysis",
        help="Directory for generated CSV reports.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Number of top kernels to export per module.",
    )
    parser.add_argument(
        "--show-columns",
        action="store_true",
        help="Print kernel_detail.csv columns and exit.",
    )
    return parser.parse_args()


def normalize_colname(text):
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def infer_unit_from_col(col_name):
    lowered = col_name.lower()
    if "(ns)" in lowered or lowered.endswith("ns"):
        return "ns"
    if "(ms)" in lowered or lowered.endswith("ms"):
        return "ms"
    return "us"


def convert_to_us(value, unit):
    if unit == "ns":
        return value / 1000.0
    if unit == "ms":
        return value * 1000.0
    return value


def detect_columns(fieldnames):
    norm_map = {name: normalize_colname(name) for name in fieldnames}

    def find(candidates):
        candidate_norms = [normalize_colname(c) for c in candidates]
        for cand in candidate_norms:
            for name, normalized in norm_map.items():
                if normalized == cand:
                    return name
        for cand in candidate_norms:
            for name, normalized in norm_map.items():
                if cand in normalized:
                    return name
        return None

    name_col = find([
        "kernel_name", "Kernel Name", "name", "Name", "op_name", "task_name"
    ])
    start_col = find([
        "start_time", "Start Time", "start", "timestamp", "ts"
    ])
    duration_col = find([
        "duration", "Duration", "dur", "Task Duration", "kernel_duration", "elapsed_time"
    ])
    end_col = find([
        "end_time", "End Time", "end", "End"
    ])

    if name_col is None:
        raise ValueError(f"Cannot find kernel/op name column. Available columns: {fieldnames}")
    if start_col is None:
        raise ValueError(f"Cannot find start time column. Available columns: {fieldnames}")
    if duration_col is None and end_col is None:
        raise ValueError(
            f"Cannot find duration or end time column. Available columns: {fieldnames}"
        )

    return name_col, start_col, duration_col, end_col


def classify_kernel(name):
    lowered = str(name).lower()

    if any(token in lowered for token in [
        "attention", "attn", "flash", "fusedattn", "softmax", "rope"
    ]):
        return "attention"

    if any(token in lowered for token in [
        "matmul", "gemm", "batchmatmul", "bmm", "cube"
    ]):
        return "matmul"

    if any(token in lowered for token in [
        "conv3d", "conv2d", "conv"
    ]):
        return "conv"

    if any(token in lowered for token in [
        "allgather", "all_gather", "alltoall", "all_to_all", "reducescatter",
        "reduce_scatter", "broadcast", "hccl", "send", "recv", "memcpyasync"
    ]):
        return "communication"

    if any(token in lowered for token in [
        "layernorm", "rmsnorm", "groupnorm", "batchnorm", "norm"
    ]):
        return "norm"

    if any(token in lowered for token in [
        "interpolate", "upsample", "pad", "resize"
    ]):
        return "resize_pad"

    if any(token in lowered for token in [
        "cast", "transpose", "permute", "reshape", "view", "contiguous", "copy",
        "concat", "cat", "slice", "gather", "scatter", "split"
    ]):
        return "cast_layout"

    if any(token in lowered for token in [
        "add", "sub", "mul", "div", "exp", "sqrt", "rsqrt", "gelu", "relu",
        "silu", "sigmoid", "tanh"
    ]):
        return "elementwise"

    return "other"


def parse_float(text):
    if text is None:
        return None
    stripped = str(text).strip()
    if stripped == "":
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def load_kernel_rows(kernel_detail_path):
    with open(kernel_detail_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise ValueError("kernel_detail.csv has no header.")

        name_col, start_col, duration_col, end_col = detect_columns(fieldnames)
        start_unit = infer_unit_from_col(start_col)
        duration_unit = infer_unit_from_col(duration_col) if duration_col else None
        end_unit = infer_unit_from_col(end_col) if end_col else None

        rows = []
        for raw in reader:
            name = raw.get(name_col, "")
            start_raw = parse_float(raw.get(start_col))
            if start_raw is None:
                continue
            start_us = convert_to_us(start_raw, start_unit)

            if duration_col is not None:
                duration_raw = parse_float(raw.get(duration_col))
                if duration_raw is None:
                    continue
                duration_us = convert_to_us(duration_raw, duration_unit)
                end_us = start_us + duration_us
            else:
                end_raw = parse_float(raw.get(end_col))
                if end_raw is None:
                    continue
                end_us = convert_to_us(end_raw, end_unit)
                duration_us = end_us - start_us

            if duration_us <= 0:
                continue

            row = dict(raw)
            row["kernel_name_norm"] = name
            row["start_us"] = start_us
            row["end_us"] = end_us
            row["duration_us"] = duration_us
            row["category"] = classify_kernel(name)
            rows.append(row)

    return fieldnames, rows


def load_ranges(ranges_path):
    with open(ranges_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ranges_time_unit = data.get("ranges_time_unit", "us")
    modules = {}
    for name, value in data.items():
        if name == "ranges_time_unit":
            continue
        modules[name] = value
    return ranges_time_unit, modules


def range_value_to_us(value, unit):
    return convert_to_us(float(value), unit)


def slice_module_rows(rows, ranges, range_unit):
    sliced = []
    for range_idx, item in enumerate(ranges):
        start_us = range_value_to_us(item["start"], range_unit)
        end_us = range_value_to_us(item["end"], range_unit)
        if end_us <= start_us:
            continue

        for row in rows:
            overlap_start = max(row["start_us"], start_us)
            overlap_end = min(row["end_us"], end_us)
            overlap_us = overlap_end - overlap_start
            if overlap_us <= 0:
                continue

            copied = dict(row)
            copied["overlap_us"] = overlap_us
            copied["range_idx"] = range_idx
            copied["range_start_us"] = start_us
            copied["range_end_us"] = end_us
            sliced.append(copied)
    return sliced


def group_kernel_summary(module_rows):
    grouped = defaultdict(lambda: {
        "kernel_name": "",
        "call_count": 0,
        "total_overlap_us": 0.0,
    })
    for row in module_rows:
        name = row["kernel_name_norm"]
        slot = grouped[name]
        slot["kernel_name"] = name
        slot["call_count"] += 1
        slot["total_overlap_us"] += row["overlap_us"]

    results = []
    for item in grouped.values():
        avg = item["total_overlap_us"] / item["call_count"]
        results.append({
            "kernel_name": item["kernel_name"],
            "call_count": item["call_count"],
            "total_overlap_us": item["total_overlap_us"],
            "avg_overlap_us": avg,
        })
    results.sort(key=lambda x: x["total_overlap_us"], reverse=True)
    return results


def group_category_summary(module_rows):
    grouped = defaultdict(lambda: {
        "category": "",
        "call_count": 0,
        "total_overlap_us": 0.0,
    })
    for row in module_rows:
        category = row["category"]
        slot = grouped[category]
        slot["category"] = category
        slot["call_count"] += 1
        slot["total_overlap_us"] += row["overlap_us"]

    results = []
    for item in grouped.values():
        avg = item["total_overlap_us"] / item["call_count"]
        results.append({
            "category": item["category"],
            "call_count": item["call_count"],
            "total_overlap_us": item["total_overlap_us"],
            "avg_overlap_us": avg,
        })
    results.sort(key=lambda x: x["total_overlap_us"], reverse=True)
    return results


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_pct(value):
    if math.isnan(value):
        return ""
    return f"{value:.6f}"


def main():
    args = parse_args()

    with open(args.kernel_detail, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])

    if args.show_columns:
        print("Columns in kernel_detail.csv:")
        for name in header:
            print(f"- {name}")
        return

    if not args.ranges:
        raise ValueError("--ranges is required unless --show-columns is used.")

    _, kernel_rows = load_kernel_rows(args.kernel_detail)
    range_unit, modules = load_ranges(args.ranges)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_module_rows = []

    for module_name, ranges in modules.items():
        module_rows = slice_module_rows(kernel_rows, ranges, range_unit)
        if not module_rows:
            print(f"[WARN] no kernels found for module {module_name}")
            continue

        total_overlap_us = sum(row["overlap_us"] for row in module_rows)
        kernel_summary = group_kernel_summary(module_rows)
        category_summary = group_category_summary(module_rows)

        for row in kernel_summary:
            row["module_pct"] = row["total_overlap_us"] / total_overlap_us * 100.0
        for row in category_summary:
            row["module_pct"] = row["total_overlap_us"] / total_overlap_us * 100.0

        kernel_summary = kernel_summary[:args.topk]

        kernel_fields = [
            "kernel_name", "call_count", "total_overlap_us", "avg_overlap_us", "module_pct"
        ]
        category_fields = [
            "category", "call_count", "total_overlap_us", "avg_overlap_us", "module_pct"
        ]

        write_csv(
            out_dir / f"{module_name}_top_kernels.csv",
            kernel_summary,
            kernel_fields,
        )
        write_csv(
            out_dir / f"{module_name}_category_summary.csv",
            category_summary,
            category_fields,
        )

        raw_fields = sorted(module_rows[0].keys())
        write_csv(
            out_dir / f"{module_name}_raw_overlap.csv",
            module_rows,
            raw_fields,
        )

        all_module_rows.append({
            "module": module_name,
            "kernel_rows": len(module_rows),
            "module_total_us": total_overlap_us,
            "module_total_ms": total_overlap_us / 1000.0,
        })

        print(f"[OK] {module_name}")
        print(f"     total: {total_overlap_us / 1000.0:.3f} ms")
        print(f"     top kernels: {out_dir / f'{module_name}_top_kernels.csv'}")
        print(f"     category summary: {out_dir / f'{module_name}_category_summary.csv'}")

    if all_module_rows:
        all_module_rows.sort(key=lambda x: x["module_total_us"], reverse=True)
        write_csv(
            out_dir / "all_modules_summary.csv",
            all_module_rows,
            ["module", "kernel_rows", "module_total_us", "module_total_ms"],
        )
        print(f"[OK] summary: {out_dir / 'all_modules_summary.csv'}")


if __name__ == "__main__":
    main()
