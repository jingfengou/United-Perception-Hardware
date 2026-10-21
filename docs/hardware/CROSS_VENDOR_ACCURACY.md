# 沐曦与 NVIDIA 的真实数据多种子对照

## 结论

2026-10-21 完成四模型、两平台、每个平台各五个固定种子的 **40 次训练、76,000 个优化器步骤**。40 次均跑完预先声明的预算，保存的模型/优化器/调度器状态审计通过；新建模型和优化器加载后的指标保持一致，保存的最终预测独立复算也全部一致。

**四模型均跑通，不等于四模型精度等价。按预先固定的 +/-1 个百分点标准，本轮只有 ResNet18 确认平均精度等价；YOLO、DETR、ViT 均未能确认。** 下表精度为百分数，差值与区间为百分点，方向为“沐曦减 NVIDIA”。

| 模型 / 主要指标 | NVIDIA 均值 +/- 标准差 | C500 均值 +/- 标准差 | 均值差 | 差值的 90% CI | +/-1 点等价 |
| --- | --- | --- | --- | --- | --- |
| UP YOLOv5-S / AP | 26.0174 +/- 10.2191 | 27.6099 +/- 8.8550 | +1.5925 | [-12.1917, +15.3766] | 未确认 |
| UP ResNet18 / Top-1 | 66.2200 +/- 0.5404 | 66.2900 +/- 1.0133 | +0.0700 | [-0.8113, +0.9513] | 通过，p=0.04384 |
| 外部官方 DETR 适配器 / AP | 77.6032 +/- 0.6539 | 77.0841 +/- 0.6269 | -0.5191 | [-1.0330, -0.0052] | 未确认，p=0.05839 |
| UP ViT-B/16 / Top-1 | 22.3400 +/- 4.1699 | 20.2500 +/- 1.4870 | -2.0900 | [-6.4571, +2.2771] | 未确认 |

- DETR 的下界为 -1.032966，略超出 -1，但仍严格按原标准记为未确认，不把界限改成 1.05 或通过四舍五入将其写成通过。
- YOLO 和 ViT 的区间较宽；YOLO 均值较高不证明更好，ViT 均值较低也不足以定位硬件退化。五个种子的全部结果保留，不选择有利种子、重复或 epoch。
- ViT 的两边绝对精度都低。这套 64px、强增强、从零开始的 20 epoch 配方是有界回归，不代表正常 ImageNet 精度、充分收敛或可用产品精度。
- 本轮 NVIDIA 的 20 次重跑均逐位复现上一轮对应 NVIDIA 路由版的 loss、最终模型状态和指标，见[重跑核对](validation/cross_vendor_20261021/nvidia-reproduction.json)。新增验证工具没有改变这些已测 NVIDIA 结果；不能将这一点扩展为跨厂商逐位相同。

跨平台每对的数据、算法配置、初始参数、首批真实输入、完整学习率轨迹和步数都匹配；训练轨迹与最终权重并不逐位相同。当前没有将差异定位到某个内核或唯一原因。两边使用不同厂商支持的软件栈，因此不能仅凭这张表将差异归因于 GPU、硬件路由或某一个 PyTorch 版本。

原始数据：[全 20 对种子 CSV](validation/cross_vendor_20261021/pairs.csv)、[统计 JSON](validation/cross_vendor_20261021/summary.json)、[完整性核对](validation/cross_vendor_20261021/coverage.json)、[独立 SciPy 统计复核](validation/cross_vendor_20261021/statistics-recheck.json)。单次运行摘要和配置可直接浏览 [NVIDIA](validation/cross_vendor_20261021/nvidia/results/) / [沐曦](validation/cross_vendor_20261021/metax/results/)。完整原始报告、逐步 loss/LR、日志、最终预测和标签分别保存在 [NVIDIA 证据包](validation/cross_vendor_20261021/nvidia-evidence.tar.gz) / [沐曦证据包](validation/cross_vendor_20261021/metax-evidence.tar.gz)，每个包附逐文件哈希索引；网页摘要注明对应原始报告的成员路径及哈希，不替代原始记录。

## 本轮范围

按用户确认，**沿用此前 CIFAR-10/PennFudan 的完整固定训练预算**，不是 ImageNet/COCO 全量训练。[计划](validation/cross_vendor_20261021/plan.json)及其哈希在本轮完整训练启动前固定；短程预检另行保存，不作为精度等价结果。

两边都运行本仓库同一份 UP 模型、BaseRunner、数据增强、损失、优化器和评估流程，仅选择 `nvidia` 或 `metax` 后端。不拿 Transformers/Ultralytics 的其他模型替代 UP 模型，不将旧 NVIDIA 原版/路由版的训练结果算作本次跨厂商结果。

| 模型 | 固定真实数据 | 每次预算 | 单次优化器步数 | 每个平台的种子 |
| --- | --- | --- | --- | --- |
| UP ResNet18 | CIFAR-10，10,000 train / 2,000 test | 64px，20 epoch，batch64 | 3,140 | 131、132、133、134、135 |
| UP YOLOv5-S SiLU | PennFudan，120 train / 50 holdout | 320px，300 epoch，batch64，EMA off | 600 | 同上 |
| 外部官方 DETR R50 适配器 | 同一 PennFudan 划分 | 320/512px，12 epoch，batch2 | 720 | 同上 |
| UP ViT-B/16 | 同一 CIFAR-10 子集 | 64px，20 epoch，batch64 | 3,140 | 同上 |

合计 40 次完整训练、76,000 个优化器步骤。ViT 保留全部 12 个 Base block 和 768 维特征，但输入为 64px/17 tokens；DETR 是同一份已固定的外部官方适配器，单 person 类、官方 COCO 权重初始化，不是上游 UP 自有模型。YOLO 在 120 张图上 300 epoch 只有 600 步。详细超参数沿用[上一轮配方](MULTISEED_ACCURACY.md#训练配方)，DETR 使用修正后的骨干/其他参数分组学习率。

## 什么必须保持一致

- 同一份图片、标签、划分顺序、DETR 源码/权重和 YOLO anchors，先在一台机器准备一次，再分发同一份数据包，不分别抽样或转换权重。
- 相同模型源码和算法配置、种子、初始模型状态、首个真实训练 batch、逐步学习率及完整训练步数。算法配置哈希只排除厂商选择字段，不排除其他训练设置。
- 严格 FP32，两边均关闭 matmul/cuDNN TF32 和 cuDNN benchmark，设置确定性请求；不增加 CPU 数值回退，不使用梯度回放或舍入。
- 每次重新构造模型及优化器加载 checkpoint，复测最终指标；另审计保存的模型/优化器/调度器张量有限值及模型哈希。

不同厂商的 GPU 随机数实现、原生内核和软件版本不要求逐位相同；记录而不是删除这些差异。NVIDIA 保留原生算子，沐曦使用已有共享同设备实现，不将 NVIDIA 二进制复制到沐曦。Loss 和最终权重的一致性仅作诊断，不代替精度判断。

## 环境与统计口径

NVIDIA 为 A800、Python 3.12.3、Torch `2.8.0a0+5228986c39.nv25.06`、TorchVision `0.22.0a0+95f10a4e`；沐曦为 C500、厂商镜像 `maca-pytorch:3.8.1.2-torch2.10-py312-ubuntu24.04-amd64`、Python 3.12.11、Torch `2.10.0+metax3.8.1.0`、TorchVision `0.25.0+metax3.8.1.0`。不覆盖厂商 PyTorch。**这是两套实际可用平台栈的对照，不能把差异全部归因于硬件或路由。**

每个模型固定使用最后一个 epoch，报告全部 5 对种子。差值统一为“沐曦减 NVIDIA”，Top-1/AP 均转换为百分点；提前固定 +/-1 点的配对 TOST、alpha=0.05。均值差 90% CI 完全落在 (-1,+1) 内才确认该固定 holdout/预算上的平均精度等价，否则明确报告未能确认，不把均值接近或普通差异检验不显著当成通过。四个模型分别判断，不通过挑种子、换阈值或合并模型得到总通过。

## 复现步骤

先按[主 README](../../README.md)及[沐曦环境说明](METAX_VALIDATION.md)配置两边环境。仅 NVIDIA 构建原生扩展；沐曦使用已有厂商环境和 `UP_BUILD_BACKEND=none`。验证工具恢复的三个历史 Python/NumPy 别名与此前一致，不改计算内核。

1. 在一台机器按[上一轮数据准备命令](MULTISEED_ACCURACY.md#复现)生成冻结 CIFAR/PennFudan 清单、官方 DETR 清单和 YOLO anchor 清单。已有本轮数据时直接复用，禁止重新采样。
2. 两边使用相同的绝对数据路径。Docker 可将任务目录绑定到该路径；本次现有沐曦容器使用同名符号链接。复制时包括 masks，供原始标签校验，不只复制训练读取的图片。

```bash
python tools/prepare_cross_vendor_data.py \
  --manifest /workspace/previous-accuracy-data/data-with-detr-and-anchors.json \
  --output /workspace/up-cross-vendor-20261021/data

python tools/prepare_cross_vendor_data.py \
  --manifest /workspace/up-cross-vendor-20261021/data/data.json --verify
```

将准备好的整个 `data/` 分发到另一平台相同路径，先核对压缩包及 `data.json` SHA-256，再分别执行 `--verify`。本次数据包 SHA-256 为 `7a2171e949b90b86d58b61f59f438ca4abe3fd50645520b28ecda562a313e80f`，清单为 `d62a893d9ea3733b55e972e0609a5243022a26b3f5fa67076acf7e88a08a6830`。

3. 在每个平台执行完整预算。`--devices` 是允许使用的物理 GPU 编号，每个训练任务只使用一张卡；下面的 Python 必须来自该平台验证过的虚拟环境。

本次 NVIDIA 镜像使用任务内的标准 pycocotools 2.0.10，实际启动时额外设置 `PYTHONPATH=/workspace/up-hardware-routing-20261021-accuracy/compat`，不使用镜像自带的定制 COCO 评估扩展。新环境可在任务虚拟环境安装该标准包，或按上一轮说明安装到独立 `--target` 目录并把它加入 PYTHONPATH；沐曦虚拟环境已安装标准 2.0.10。训练与独立复算须使用同一评估实现，不覆盖厂商 PyTorch。

```bash
export USER=oujingfeng LOGNAME=oujingfeng
export PYTHONHASHSEED=0 DEFAULT_TASKS=cls:det
export PYTORCH_DISABLE_CUDA_CUDNN_TF32=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

python tools/run_cross_vendor_accuracy.py \
  --plan docs/hardware/validation/cross_vendor_20261021/plan.json \
  --backend nvidia --source "$PWD" \
  --data /workspace/up-cross-vendor-20261021/data/data.json \
  --devices 0,1,2,3 --output /workspace/up-cross-vendor-20261021/results
```

沐曦将 `--backend nvidia` 换为 `--backend metax`，不是改变原通信参数 `--backend dist` 的含义。两边输出目录可以同名，但收集时分别放入 `nvidia/`、`metax/`。可用 `--models` 分组排队；不能改变计划中的种子或预算。重复执行只复用哈希、协议和预算均验证通过的完整报告；遇到非空失败目录会停止并保留，不自动覆盖或把其当作完成。

4. 完成后分别审计所有 checkpoint，并从保存的最终预测独立复算指标。分类按原 UP 的 CPU float32 计数公式计算 Top-1/Top-5，同时核对每个文件名和标签；检测直接使用标准 pycocotools 的 COCOeval。复算结果必须与 Runner 报告完全一致，不放宽指标容差，不需要重新进行 GPU 推理。

```bash
python tools/audit_cross_vendor_accuracy.py \
  --runs /workspace/up-cross-vendor-20261021/results/*-seed*/ \
  --output /workspace/up-cross-vendor-20261021/checkpoint-audits.json

python tools/recompute_cross_vendor_metrics.py \
  --runs /workspace/up-cross-vendor-20261021/results/*-seed*/ \
  --output /workspace/up-cross-vendor-20261021/metric-reevaluations.json

python tools/collect_cross_vendor_evidence.py \
  --root /workspace/up-cross-vendor-20261021 \
  --plan docs/hardware/validation/cross_vendor_20261021/plan.json \
  --backend nvidia --output /workspace/nvidia-evidence.tar.gz

python tools/run_cross_vendor_accuracy.py \
  --plan docs/hardware/validation/cross_vendor_20261021/plan.json \
  --summarize --nvidia /workspace/collected/nvidia \
  --metax /workspace/collected/metax --output /workspace/collected/summary.json
```

汇总缺少任一规定种子、完整步数或公平配对条件不一致时会拒绝生成完成结论。短程 `--probe-steps 2` 输出明确标记为 `probe_completed`，不准计入完整训练。统计推断只覆盖固定数据划分上的训练随机性，不包含完整数据集抽样误差、原分辨率收敛、多卡或性能验收。

汇总程序成功退出只说明记录完整、统计计算完成，不表示所有模型等价；须查看 `all_models_equivalent` 和各模型的 `equivalence_test.equivalent_within_margin`。本轮前者为 `false`。配对 Student t 推断依赖种子差值的独立性/近似正态假设，n=5 的区间和检验能力有限。

恢复检查为新建模型/优化器加载后重新评估；保存了调度器状态，但本轮没有验收任意中途断点续训与不中断训练的逐位连续性。

证据包按 `nvidia/`、`metax/` 分开保存，包含报告、有效配置、最终预测、复算结果、审计、日志、标签与环境，不包含数据图片、checkpoint 大文件、厂商二进制或私钥。解包后复算可增加 `--data-root <包内的 backend/data>` 指定标签位置。完整 checkpoint 保留在各服务器的任务目录。

本轮调度使用 NVIDIA 四个独立单卡 worker；沐曦五个 YOLO worker 与三个其他模型 worker 分组执行。自建沐曦容器的 CPU 配额由 8 临时增加到 24，结束后恢复；不改 batch、增强、采样器、精度、种子或学习率。两机日志时钟约有 260 秒偏差，记录见[调度清单](validation/cross_vendor_20261021/scheduling.json)。本轮不据墙钟日志比较性能。

本轮只增加验证工具、协议、测试和记录，没有修改 `up/` 或模型配置模板。436 个应用 Python 文件的规范化哈希在 40 份报告中均与 Git 源码一致；工具对应 `cd48819` / `7f881b6`，应用内容与 `516aa21` 相同。两平台隐藏设备后的 CPU 契约/协议/证据测试各 164 项通过。另有 8 次真实数据两步预检，共 16 步，排除在主实验 40 次/76,000 步之外。

测试后两台机器均无本任务 GPU 计算进程，沐曦容器保留供使用、CPU 配额已恢复为 8。NVIDIA 的 Windows SSH 客户端曾在远端全部完成后挂起，确认远端无训练进程、全部报告及子任务退出检查正常后，仅关闭了该客户端；不将此客户端退出误报为训练失败，详情见完整性记录。
