# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime

import torch
import torch.distributed as dist

try:
    import torch_npu
except ImportError:  # pragma: no cover - local non-NPU dev environment
    torch_npu = None


def _sync_device():
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def maybe_mstx_range(name, enabled=True):
    if not enabled or torch_npu is None or not hasattr(torch_npu.npu, "mstx"):
        yield
        return

    range_id = torch_npu.npu.mstx.range_start(name)
    try:
        yield
    finally:
        torch_npu.npu.mstx.range_end(range_id)


class PhaseTimer:

    def __init__(self, sync_fn=None):
        self.sync_fn = sync_fn or _sync_device
        self.elapsed = defaultdict(float)

    @contextmanager
    def phase(self, name):
        self.sync_fn()
        start = time.time()
        try:
            yield
        finally:
            self.sync_fn()
            self.elapsed[name] += time.time() - start


class ProfilingContext:

    def __init__(
        self,
        enabled=False,
        rank=0,
        world_size=1,
        output_dir=None,
        emit_mstx=True,
        summary_enabled=True,
    ):
        self.enabled = enabled
        self.rank = rank
        self.world_size = world_size
        self.output_dir = output_dir
        self.emit_mstx = emit_mstx
        self.summary_enabled = summary_enabled
        self.timer = PhaseTimer()
        self.metadata = {}

        if self.enabled and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

    @property
    def trace_dir(self):
        if self.output_dir is None:
            return None
        return os.path.join(self.output_dir, "trace")

    @property
    def summary_json_path(self):
        if self.output_dir is None:
            return None
        return os.path.join(self.output_dir, "summary.json")

    @property
    def summary_txt_path(self):
        if self.output_dir is None:
            return None
        return os.path.join(self.output_dir, "summary.txt")

    @contextmanager
    def phase(self, name):
        if not self.enabled:
            yield
            return

        with maybe_mstx_range(name, self.emit_mstx):
            with self.timer.phase(name):
                yield

    def set_metadata(self, **kwargs):
        self.metadata.update(kwargs)

    def get_phase_totals(self):
        return dict(self.timer.elapsed)

    def build_summary(self):
        phase_times = self.get_phase_totals()
        total = sum(phase_times.values())
        percentages = {}
        if total > 0:
            for name, value in phase_times.items():
                percentages[name] = value / total

        summary = {
            "timestamp": datetime.now().isoformat(),
            "rank": self.rank,
            "world_size": self.world_size,
            "total_profiled_time_sec": total,
            "phase_times_sec": phase_times,
            "phase_percentages": percentages,
        }
        summary.update(self.metadata)
        return summary

    def write_summaries(self):
        if not self.enabled or not self.summary_enabled or self.output_dir is None:
            return None

        summary = self.build_summary()
        with open(self.summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        with open(self.summary_txt_path, "w", encoding="utf-8") as f:
            f.write(f"rank: {summary['rank']}\n")
            f.write(f"world_size: {summary['world_size']}\n")
            f.write(f"task: {summary.get('task', '')}\n")
            f.write(f"prompt: {summary.get('prompt', '')}\n")
            f.write(f"seed: {summary.get('seed', '')}\n")
            f.write(f"total_profiled_time_sec: {summary['total_profiled_time_sec']:.6f}\n")
            for name, value in sorted(summary["phase_times_sec"].items()):
                pct = summary["phase_percentages"].get(name, 0.0) * 100.0
                f.write(f"{name}: {value:.6f}s ({pct:.2f}%)\n")
        return summary

    def write_aggregate_summary(self):
        if not self.enabled or not self.summary_enabled or self.output_dir is None:
            return None

        local_summary = self.build_summary()
        all_summaries = [local_summary]
        if dist.is_available() and dist.is_initialized():
            all_summaries = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_summaries, local_summary)

        if self.rank != 0:
            return None

        phase_names = set()
        for item in all_summaries:
            phase_names.update(item["phase_times_sec"].keys())

        aggregate = {
            "timestamp": datetime.now().isoformat(),
            "task": self.metadata.get("task"),
            "world_size": self.world_size,
            "rank_count": len(all_summaries),
            "phase_aggregate_sec": {},
        }
        for name in sorted(phase_names):
            values = [item["phase_times_sec"].get(name, 0.0) for item in all_summaries]
            aggregate["phase_aggregate_sec"][name] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }

        aggregate_path = os.path.join(os.path.dirname(self.output_dir), "aggregate_summary.json")
        with open(aggregate_path, "w", encoding="utf-8") as f:
            json.dump(aggregate, f, indent=2, ensure_ascii=False)
        return aggregate
