# Kernel Detail 模块分析

这个文档说明如何用 `kernel_detail.csv` 和时间线里的模块范围，离线分析：

- `DIT_LOW` 内部最重的 kernel
- `DIT_HIGH` 内部最重的 kernel
- `VAE_DECODE` 内部最重的 kernel

这套方法的目标不是做全局 profiling 汇总，而是回答：

- 哪些 kernel 属于 DiT
- 哪些 kernel 属于 VAE
- DiT/VAE 各自内部最重的算子类别是什么

## 为什么优先看 `kernel_detail.csv`

`op_statistic.csv` 更适合看全局热点，但默认是整次 profiling 周期的聚合统计。  
如果你想按模块拆分，例如区分 `DIT_LOW` 和 `VAE_DECODE`，核心依据应该是：

1. 在 MindStudio Insight 时间线里找到模块的时间范围
2. 用这些时间范围去切 `kernel_detail.csv`
3. 再做模块内的 kernel 排序和类别汇总

## 脚本位置

离线分析脚本在：

```text
tools/analyze_module_kernels.py
```

## 第一步：先看 `kernel_detail.csv` 的列名

```bash
python3 tools/analyze_module_kernels.py \
  --kernel-detail kernel_detail.csv \
  --show-columns
```

这个命令会打印当前 CSV 的列名，方便确认时间列和 kernel 名称列是否被脚本正确识别。

## 第二步：在 Insight 时间线里抄出模块时间范围

做一个 `module_ranges.json` 文件，示例：

```json
{
  "ranges_time_unit": "us",
  "DIT_LOW": [
    {"start": 123456.0, "end": 130789.0},
    {"start": 231000.0, "end": 238500.0}
  ],
  "DIT_HIGH": [
    {"start": 140000.0, "end": 147000.0}
  ],
  "VAE_DECODE": [
    {"start": 500000.0, "end": 520000.0}
  ]
}
```

说明：

- `ranges_time_unit` 支持 `ns`、`us`、`ms`
- 每个模块可以填多个范围
- 脚本会把这些范围和 `kernel_detail.csv` 的时间段做 overlap 计算

## 第三步：跑离线分析

```bash
python3 tools/analyze_module_kernels.py \
  --kernel-detail kernel_detail.csv \
  --ranges module_ranges.json \
  --out-dir module_kernel_analysis \
  --topk 20
```

输出目录中会出现：

- `DIT_LOW_top_kernels.csv`
- `DIT_LOW_category_summary.csv`
- `DIT_LOW_raw_overlap.csv`
- `DIT_HIGH_top_kernels.csv`
- `DIT_HIGH_category_summary.csv`
- `VAE_DECODE_top_kernels.csv`
- `VAE_DECODE_category_summary.csv`
- `all_modules_summary.csv`

## 输出文件怎么理解

### 1. `*_top_kernels.csv`

这是模块内部最重的 kernel 排序。重点看：

- `kernel_name`
- `call_count`
- `total_overlap_us`
- `avg_overlap_us`
- `module_pct`

适合回答：

- `DIT_LOW` 里最重的是 attention，还是 matmul
- `VAE_DECODE` 里最重的是 conv3d，还是插值/小算子

### 2. `*_category_summary.csv`

这是把 kernel 粗分类之后的汇总。脚本会把名字大致归到：

- `attention`
- `matmul`
- `conv`
- `communication`
- `norm`
- `resize_pad`
- `cast_layout`
- `elementwise`
- `other`

适合回答：

- `DIT_LOW` 主要是 attention 瓶颈，还是通信瓶颈
- `VAE_DECODE` 主要是卷积瓶颈，还是 resize/pad/碎片化问题

### 3. `*_raw_overlap.csv`

这是原始 overlap 明细。适合你后面做更细的二次分析，比如：

- 只看某个 range
- 只看某类 kernel
- 自己重新定义分类规则

## 推荐分析顺序

优先顺序建议是：

1. `DIT_LOW`
2. `DIT_HIGH`
3. `VAE_DECODE`
4. `VAE_ENCODE`

原因很简单：如果模块级 profiling 里 `DiT_total` 已经远大于 `VAE_total`，那应该先分析 DiT 内部热点。

## 如何用结果判断瓶颈

### 情况 1：`DIT_LOW` 里 `attention + matmul` 占绝大多数

说明当前主瓶颈还是 DiT 主干本身。

### 情况 2：`DIT_LOW` 里 `communication` 比例很高

说明 `ulysses_size=4` 下的 HCCL / sequence parallel 开销已经不小了，接下来应该优先看并行策略而不是单算子。

### 情况 3：`DIT_LOW` 里 `cast_layout` 很高

说明不是单纯算力瓶颈，而是 dtype / transpose / contiguous / 小 kernel 碎片化较多。

### 情况 4：`VAE_DECODE` 里 `conv` 很高

说明 VAE 更偏卷积型热点。

### 情况 5：`VAE_DECODE` 里 `resize_pad + cast_layout` 很高

说明 VAE 更偏访存和碎片化问题，不是纯卷积核本身的问题。

## 和 `op_statistic.csv` 的关系

建议把 `op_statistic.csv` 当作全局对照，而不是模块归因主依据。

更稳的用法是：

1. 用模块级时间范围切 `kernel_detail.csv`
2. 得到 `DIT_LOW_top_kernels.csv`
3. 再拿 `op_statistic.csv` 对照看全局 top op 是否一致

## 最后建议

当你已经有模块级占比时，最值得先整理的两张表是：

1. 模块级占比表
2. `DIT_LOW` / `DIT_HIGH` / `VAE_DECODE` 的 top kernel 表

这样你就能比较快地判断：

- 下一步该先优化 DiT 还是 VAE
- 该先看 attention、通信，还是 cast/layout
