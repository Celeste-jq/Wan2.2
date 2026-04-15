# Wan2.2 NPU Profiling

This repository includes an NPU profiling entry point for MindIE Wan2.2.

## Supported scope

The first supported profiling scenario is:

- MindIE Wan2.2
- NPU runtime
- `i2v-A14B`
- multi-rank `torchrun`

The original `generate.py` remains unchanged for normal generation runs. Use `generate_profiling.py` when you want NPU profiling output and module timing summaries.

## Example command

```bash
torchrun --nproc_per_node=4 generate_profiling.py \
  --task i2v-A14B \
  --ckpt_dir /home/xx/models/Wan-AI/Wan2.2-I2V-A14B \
  --size 832*480 \
  --frame_num 81 \
  --sample_steps 40 \
  --dit_fsdp \
  --t5_fsdp \
  --cfg_size 1 \
  --ulysses_size 4 \
  --image /home/xx/wan22/images.jpg \
  --prompt "A cat playing with yarn" \
  --base_seed 0
```

## Profiling-specific arguments

- `--profile_output_dir`: root directory for profiling outputs, default `profiling_runs`
- `--profile_warmup_steps`: warm-up sampling steps before measured capture, default `2`
- `--profile_level`: requested profiler level, one of `level0`, `level1`, `level2`
- `--profile_summary`: whether to write `summary.json` and `summary.txt`, default `true`
- `--profile_with_stack`: include stack information in profiler output
- `--profile_record_shapes`: include tensor shape information in profiler output
- `--profile_memory`: include memory profiling data

## Output layout

Each run creates a timestamped output directory under `--profile_output_dir`.

Example layout:

```text
profiling_runs/
  20260415_120000_i2v-A14B/
    aggregate_summary.json
    rank0/
      trace/
      summary.json
      summary.txt
    rank1/
      trace/
      summary.json
      summary.txt
```

## Module labels

The profiling flow separates major execution phases with explicit labels:

- `TEXT_ENCODER`
- `DIT_HIGH`
- `DIT_LOW`
- `VAE_ENCODE`
- `VAE_DECODE`

For `i2v-A14B`:

- `DIT_HIGH` means the high-noise DiT expert forward
- `DIT_LOW` means the low-noise DiT expert forward
- `VAE_ENCODE` covers input image to latent conversion
- `VAE_DECODE` covers final latent to video reconstruction

These labels are intended for MindStudio or msprof timeline inspection and for the per-rank `summary.json` and `summary.txt` files.

## Notes

- Warm-up runs happen before the measured profiling run.
- Profiling data is written per rank to avoid collisions under `torchrun`.
- The current profiling entry point is intentionally limited to `i2v-A14B`, because that is the active validated target path for DiT/VAE module attribution.

## Offline kernel analysis

For offline module-level kernel analysis based on `kernel_detail.csv`, see:

```text
docs/kernel_analysis.md
```

The analysis script is:

```text
tools/analyze_module_kernels.py
```
