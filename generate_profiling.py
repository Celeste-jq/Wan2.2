# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import sys

import generate as base_generate


def _profiling_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--profile_output_dir",
        type=str,
        default="profiling_runs",
        help="Directory root for per-rank NPU profiling outputs.",
    )
    parser.add_argument(
        "--profile_warmup_steps",
        type=int,
        default=2,
        help="Warm-up sampling steps before the measured profiling run.",
    )
    parser.add_argument(
        "--profile_level",
        type=str,
        default="level1",
        choices=["level0", "level1", "level2"],
        help="Requested torch_npu profiler level.",
    )
    parser.add_argument(
        "--profile_summary",
        type=base_generate.str2bool,
        default=True,
        help="Whether to export per-rank summary.json and summary.txt files.",
    )
    parser.add_argument(
        "--profile_with_stack",
        action="store_true",
        default=False,
        help="Enable stack capture in profiler output.",
    )
    parser.add_argument(
        "--profile_record_shapes",
        action="store_true",
        default=False,
        help="Enable shape recording in profiler output.",
    )
    parser.add_argument(
        "--profile_memory",
        action="store_true",
        default=False,
        help="Enable memory profiling in profiler output.",
    )
    return parser


def _parse_args():
    parser = _profiling_parser()
    profiling_args, remaining = parser.parse_known_args()

    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + remaining
    try:
        args = base_generate._parse_args()
    finally:
        sys.argv = original_argv

    if args.task != "i2v-A14B":
        raise NotImplementedError(
            "generate_profiling.py currently supports only --task i2v-A14B."
        )

    for key, value in vars(profiling_args).items():
        setattr(args, key, value)
    args.profile_enabled = True
    return args


if __name__ == "__main__":
    base_generate.generate(_parse_args())
