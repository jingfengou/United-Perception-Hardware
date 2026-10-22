# 跨厂商数值修复与完整复测

> 历史报告：结论仅适用于正文指定的源码、环境和预算，不自动升级为当前版本验收。原始材料见[归档索引](../README.md)，当前工具入口见[迁移说明](../../../benchmark/hardware/README.md)。复现历史结果须使用对应冻结源码，不能拿当前脚本替换后续跑旧阶段。

这是[原生路径差异定位](CROSS_VENDOR_DIAGNOSIS.md)的后续工作。**上层模型、损失、数据增强、优化器配方和原有模型配置未修改**；生产代码改动仅位于 `up/accelerator/`。默认 `native` NVIDIA 路径保留，新增显式的实验性 `portable_v1` 数值配置，不把这项配置的结果冒充原生 FP32 与原版的逐位兼容结论。

## 当前状态

- 第一版共同数值配置已完成三模型、两平台、各五种子的 **30 次训练 / 44,600 步**，但三模型均未确认预设的 +/-1 点平均精度等价。这轮失败结果、完整输入、checkpoint 审计和指标复算全部保留。
- 后续完整操作和固定顺序修复已冻结。两平台各 **202 项 CPU 契约测试**通过；15 个 GPU 数学用例的完整输出/梯度哈希一致。短程 YOLO 4 步、DETR 64 步、ViT 32 步的 loss 与完整参数状态一致，属于诊断证据，**不是完整训练验收**。
- 新的[完整复测计划](https://github.com/jingfengou/United-Perception-Hardware/blob/a3e2752f454c72d6b46e0b64d0debeef0d7105a3/docs/hardware/validation/cross_vendor_fix_20261021/fixed-order/plan.json)仍使用种子 131--135、原数据和预算、原 +/-1 点标准；没有挑选 seed/epoch 或合并两轮结果。**沐曦 15 次 / 22,300 步已全部完成，状态审计和指标独立复算全部通过；NVIDIA 完整复测尚未启动，等待可分配 GPU。不能宣称修复后完整精度已经等价。**

| 阶段 | NVIDIA 完整训练 | C500 完整训练 | 跨厂商精度结论 |
| --- | --- | --- | --- |
| 第一版共同数值配置 | 15 次 | 15 次 | 三模型均未确认等价，失败结果保留 |
| 固定顺序冻结版 | 0/15 次 | 15/15 次 | 尚未评估，不拿上一阶段 NVIDIA 结果代替 |

## 为什么只统一版本和 seed 不够

| 已观察的分叉 | 证据范围 | 适配层处理 |
| --- | --- | --- |
| 卷积前向关闭 DNN，反向又使用 DNN | PyTorch 卷积反向重新读取全局开关；GPU profiler 对照显示前向/反向路径不同 | 配置存续期间保持 DNN 关闭，并按完整操作实现 2D 卷积前后向 |
| YOLO 的 BatchNorm 归约顺序 | 相同卷积输出后，BN 首次出现 FP64/FP32 数值分叉 | 共同的 GPU BN 前后向及固定顺序归约，保留 running stats 语义 |
| DETR 梯度裁剪的 L2 norm | 裁剪前非零 FP32 梯度一致，`_foreach_norm` 后裁剪缩放分叉 | 共同的 L2 norm/foreach norm，不改裁剪阈值或优化器 |
| DETR 的 L1 距离 | 相同输入的 `_cdist_forward` 出现非零 FP32 分叉 | 完整的 L1 cdist 前向、反向和广播归约 |
| ViT 位置向量的广播反向 | 相同输入的 `[64,17,768]` 沿 batch 求和首先出现非零 FP32 分叉 | Dispatch 层覆盖 autograd 内部 FP32 sum，宽精度固定顺序后回到原 dtype |
| 两种编译器生成不同的乘加/归约 | 相同 Triton 表达式仍出现 FP64 差异，单独关闭隐式融合不足 | GPU 显式 FMA、固定 K 次序、明确蝶形加法，不依赖厂商隐含归约树 |

共同 GPU Philox 随机流同时覆盖 Dropout 前后向；不会关闭随机层或用 CPU 生成训练掩码。部分早期 Dropout 原始哈希差异仅为 `+0/-0`，不能把它当成非零误差，更不能声称已唯一解释最终 AP 差值。新追踪同时记录零符号归一后的 FP32 哈希。

这些证据定位的是现有软件栈和算子实现的数值路径，不是 GPU 物理硬件故障，也未把每个最终精度差值分摊给某一个算子。数学公式一致不代表有限精度求和顺序一致。

见[诊断摘要及原始证据说明](https://github.com/jingfengou/United-Perception-Hardware/blob/a3e2752f454c72d6b46e0b64d0debeef0d7105a3/docs/hardware/validation/cross_vendor_fix_20261021/diagnostics/README.md)。DETR/ViT 追踪区分零符号和非零差异；较早 YOLO 追踪只有原始字节哈希，不把缺少的零符号检查写成已经通过。卷积反向读取全局 DNN 状态的来源可查 [PyTorch 2.10 Convolution.cpp](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/Convolution.cpp) 中的 `ctx.userEnabledCuDNN()`。

## 已完成但未通过的第一版

精度为百分数，差值为 C500 减 NVIDIA 的百分点。全部使用最终 epoch；配对 Student-t TOST，alpha=0.05、等价边界 +/-1 点。

| 模型 | NVIDIA 均值 | C500 均值 | 均值差 | 差值 90% CI | 等价 |
| --- | --- | --- | --- | --- | --- |
| UP YOLOv5-S | 23.5346 | 22.0708 | -1.4638 | [-13.8111, +10.8834] | 未确认 |
| 外部官方 DETR R50 | 77.4454 | 78.1779 | +0.7325 | [-0.3505, +1.8155] | 未确认 |
| UP ViT-B/16 | 21.2600 | 20.6600 | -0.6000 | [-4.4695, +3.2695] | 未确认 |

[完整统计](https://github.com/jingfengou/United-Perception-Hardware/blob/a3e2752f454c72d6b46e0b64d0debeef0d7105a3/docs/hardware/validation/cross_vendor_fix_20261021/summary.json)、[完整性核对](https://github.com/jingfengou/United-Perception-Hardware/blob/a3e2752f454c72d6b46e0b64d0debeef0d7105a3/docs/hardware/validation/cross_vendor_fix_20261021/coverage.json)及 NVIDIA/MetaX 原始证据和源文件快照位于同一目录。30 份 checkpoint 均完成有限值、模型状态哈希和恢复指标核对，最终预测独立复算。15 对全部训练输入与 pre-forward RNG 哈希一致，但没有一对最终模型状态逐位一致。统计后处理使用本地 Python 3.13.11 / SciPy 1.17.0；训练环境仍为 Python 3.12 / SciPy 1.16.0，不将这两个环境混写。

## 冻结版完整复测

本轮仍是用户选择的 CIFAR-10/PennFudan 固定预算，不是 ImageNet/COCO 全量收敛或性能验收。

| 模型 | 数据与配方 | 每种子预算 |
| --- | --- | --- |
| UP YOLOv5-S | PennFudan 120/50，320px，batch64，随机初始化，SiLU，EMA 关闭 | 300 epoch / 600 step |
| 外部官方 DETR R50 | 同一固定官方适配器、权重、参数组及 PennFudan 划分 | 12 epoch / 720 step |
| UP ViT-B/16 | CIFAR-10 10000/2000，64px、17 tokens、12 blocks，原增强与优化器 | 20 epoch / 3140 step |

每个平台 15 次训练，共 30 次 / 44,600 步。两平台的 `up/` 规范化源码 manifest 均为 `7678e99eb967c6e04c3f7e1cd57ad09bb23810fd8e9e7ed2615bffaf45be845a`，覆盖 443 个 Python 文件。启动器强制核对该值，并检查实际关闭 DNN 和关键共享算子的正调用次数。文件换行规范化仅用于源代码对应关系，不对训练张量或指标做舍入。

应用环境沿用前轮已对齐的 PyTorch 2.10 / TorchVision 0.25 主版本及 23 项共享依赖。厂商二进制、驱动、SDK、Triton 构建、Python 补丁版本仍不同，版本细节沿用[对齐报告](CROSS_VENDOR_DIAGNOSIS.md#环境对齐)。不会擅自升级共享服务器驱动。

### 已完成的 C500 结果

以下是固定顺序版的**单平台**五种子最终结果，均值及样本标准差单位为百分数。没有使用最好 epoch；不能将本表与上一阶段 NVIDIA 均值相减，冒充相同数值实现下的比较。

| 模型 | C500 均值 +/- 标准差 | 种子数 | 每次完整步数 |
| --- | --- | --- | --- |
| UP YOLOv5-S / AP | 21.8671 +/- 9.0188 | 5 | 600 |
| 外部官方 DETR R50 / AP | 76.7118 +/- 0.5214 | 5 | 720 |
| UP ViT-B/16 / Top-1 | 22.9500 +/- 3.2604 | 5 | 3140 |

[单平台完整性与逐种子记录](https://github.com/jingfengou/United-Perception-Hardware/blob/a3e2752f454c72d6b46e0b64d0debeef0d7105a3/docs/hardware/validation/cross_vendor_fix_20261021/fixed-order/metax-completion.json)、[可浏览运行摘要](https://github.com/jingfengou/United-Perception-Hardware/tree/a3e2752f454c72d6b46e0b64d0debeef0d7105a3/docs/hardware/validation/cross_vendor_fix_20261021/fixed-order/metax/results/)及[原始证据包](https://github.com/jingfengou/United-Perception-Hardware/blob/a3e2752f454c72d6b46e0b64d0debeef0d7105a3/docs/hardware/validation/cross_vendor_fix_20261021/fixed-order/metax-evidence.tar.gz)保留全部 loss/LR、每一步输入/RNG 哈希、最终预测、15 份 checkpoint 审计和 15 份独立指标复算。归档 SHA-256 为 `f74add81faf96b7ccf4171d9f03bb977e2b0040b3c2510fa20d5bd391d92e02a`，[成员索引](https://github.com/jingfengou/United-Perception-Hardware/blob/a3e2752f454c72d6b46e0b64d0debeef0d7105a3/docs/hardware/validation/cross_vendor_fix_20261021/fixed-order/metax-evidence.tar.gz.index.json)包含 99 个文件。源文件与冻结计划及当前发布 UP 代码对应，不加载 NVIDIA 扩展。

[短程与完整训练前缀核对](https://github.com/jingfengou/United-Perception-Hardware/blob/a3e2752f454c72d6b46e0b64d0debeef0d7105a3/docs/hardware/validation/cross_vendor_fix_20261021/fixed-order/short-vs-full-prefixes.json)确认种子 131 的初始化、输入及前 4/64/32 步 loss 一致，但这不能替代缺失的 NVIDIA 完整训练。新阶段没有双向交换最终权重的 12 次交叉推理，不能把历史原生策略的成功结果重复算作新阶段验收。

NVIDIA 资源核对发现 A800/A100 仍有其他任务及排队进程，即使瞬时显存接近空闲，也不擅自占用或暂停它们。NVIDIA 完整重测、全部输入配对、最终权重跨平台评估和 +/-1 点 TOST 仍待补齐；目前不生成一份貌似完成的跨厂商统计。沐曦任务已退出，容器 CPU 配额恢复到原来的 8 核。YOLO 的种子方差和 ViT 的低绝对精度继续保留，不把小数据回归解释为充分收敛。

[GitHub CI 记录](https://github.com/jingfengou/United-Perception-Hardware/blob/a3e2752f454c72d6b46e0b64d0debeef0d7105a3/docs/hardware/validation/cross_vendor_fix_20261021/ci-af6c019.json)确认代码提交 `af6c019` 的 Python 3.10/3.12、CPU PyTorch 2.6 检查通过，不是 GPU PyTorch 2.6 或完整跨厂商精度的支持声明。数值用例和 202 项目标环境测试来自上文实际 2.10 环境；12 项离线证据工具测试为单独的本地环境。

## 复现

1. 复用[首次真实数据对照](CROSS_VENDOR_ACCURACY.md#复现步骤)冻结的数据与划分、DETR 权重和兼容前置，不重新抽样。不发布原始图片、权重、SDK 或私钥。
2. 精确恢复本轮执行代码时，在干净的本仓库副本中展开本阶段的 `nvidia-source.tar.gz` 或 `metax-source.tar.gz`。这些是 Python/config/tool/test 源码，不包含厂商 `.so`；NVIDIA 扩展必须在匹配环境中重新构建。核对 `.index.json` 的归档及成员哈希，不用日后更新的源码冒充本轮受测版本。
3. 在两平台都明确选择 `portable_v1`。它保留 FP32 参数/边界，但内部有共同 GPU FP64 计算和顺序内核；不是全程 FP32，也不是性能推荐。模式限制、开销和 RNG 恢复契约见[数值配置说明](PORTABLE_NUMERICS.md)。

```bash
export DEFAULT_TASKS=cls:det PYTHONHASHSEED=0
export PYTORCH_DISABLE_CUDA_CUDNN_TF32=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python tools/run_cross_vendor_accuracy.py \
  --plan docs/hardware/validation/cross_vendor_fix_20261021/fixed-order/plan.json \
  --backend nvidia --source "$PWD" \
  --data /workspace/up-cross-vendor-20261021/data/data.json \
  --devices 0 --output /workspace/new-fixed-order/results
```

沐曦使用 `--backend metax`，仅分配允许使用的物理卡；多个设备参数只是独立单卡作业排队，不是分布式训练。计划自动开启全部 batch 审计及实际数值配置读回。两边必须使用同一个计划，不能只给一边启用宽精度模式。

4. 完整训练后使用 `tools/package_diagnosis_cohort.py` 核对 checkpoint 和最终预测，并打包原始报告。单平台用 `tools/summarize_vendor_cohort.py --root PHASE_DIRECTORY --backend metax` 生成完整性摘要；该摘要明确不评估跨厂商等价。

```bash
# PHASE 是任务专属目录，results 位于其内；不要覆盖已有证据归档。
python tools/audit_cross_vendor_accuracy.py --environment --output "$PHASE/environment.json"
# 将对应环境实际执行测试得到的 XML 放到 $PHASE/contracts-final.xml。
python tools/package_diagnosis_cohort.py \
  --root "$PHASE" --results "$PHASE/results" \
  --data-root /workspace/up-cross-vendor-20261021/data \
  --executed-tools "$PWD/tools" \
  --plan docs/hardware/validation/cross_vendor_fix_20261021/fixed-order/plan.json \
  --backend metax
```

两平台均完成后，再用 `tools/summarize_portable_cohort.py --root PHASE_DIRECTORY --write-statistics` 生成成对统计。统计目录需要两份证据包及索引、两份 source 包及索引和同一 `plan.json`。该工具拒绝覆盖既有统计，验证全部预算、输入/RNG 配对、执行源码及上层算法与参考提交的对应关系；首次生成后不带 `--write-statistics` 可复核。精确的执行工具版本在各平台 source 包中，不在训练期间替换它们。

暂不将本轮写成三模型已通过完整精度验收。AMP、多卡、编译模式、其他模型、未覆盖的随机分布及全量基准仍需分别验证；选择同一个数值配置也不是任意输入逐位一致的普遍保证。
