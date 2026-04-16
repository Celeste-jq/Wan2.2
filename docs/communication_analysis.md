# Communication Kernel 分析

这个工具用于在已有 `*_raw_overlap.csv` 基础上，单独统计各模块里的通信 kernel。

输入来自：

```text
tools/analyze_module_kernels.py
```

生成的：

```text
*_raw_overlap.csv
```

## 用法

```bash
python3 tools/analyze_communication_kernels.py \
  --input-dir module_kernel_analysis \
  --out-dir communication_analysis
```

## 输出

### 1. `communication_module_summary.csv`

按模块汇总通信总量，字段包括：

- `module`
- `module_total_us`
- `communication_total_us`
- `communication_pct`
- `communication_call_count`

适合先看：

- `DIT_LOW` 和 `DIT_HIGH` 哪个模块通信更重
- 通信总时长在模块内占比是多少

### 2. `communication_kernel_summary.csv`

按模块和 kernel 名称汇总通信热点，字段包括：

- `module`
- `kernel_name`
- `communication_type`
- `call_count`
- `total_overlap_us`
- `avg_overlap_us`
- `module_communication_pct`
- `module_total_pct`

适合回答：

- 是哪几个 HCCL/kernel 名称最重
- 某个 kernel 在通信内部占多少
- 某个 kernel 在整个模块里占多少

### 3. `communication_type_summary.csv`

按通信类型汇总，当前支持：

- `all_gather`
- `all_to_all`
- `reduce_scatter`
- `broadcast`
- `memcpy`
- `send_recv`
- `hccl_other`
- `other`

适合先快速判断：

- 当前通信主要是哪一种 collective
- 是 collective 为主，还是 memcpy / 其他同步为主
