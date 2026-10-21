# 沐曦与 NVIDIA 的真实数据多种子对照

## 本轮范围

按用户确认，**沿用此前 CIFAR-10/PennFudan 的完整固定训练预算**，不是 ImageNet/COCO 全量训练。2026-10-21 已冻结[计划](validation/cross_vendor_20261021/plan.json)并开始执行。完整结果尚未产生，不能据短程预检宣称精度等价。

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

4. 完成后分别审计所有 checkpoint，再把报告和审计文件收集到同一台机器做配对汇总。

```bash
python tools/audit_cross_vendor_accuracy.py \
  --runs /workspace/up-cross-vendor-20261021/results/*-seed* \
  --output /workspace/up-cross-vendor-20261021/checkpoint-audits.json

python tools/run_cross_vendor_accuracy.py \
  --plan docs/hardware/validation/cross_vendor_20261021/plan.json \
  --summarize --nvidia /workspace/collected/nvidia \
  --metax /workspace/collected/metax --output /workspace/collected/summary.json
```

汇总缺少任一规定种子、完整步数或公平配对条件不一致时会拒绝生成完成结论。短程 `--probe-steps 2` 输出明确标记为 `probe_completed`，不准计入完整训练。统计推断只覆盖固定数据划分上的训练随机性，不包含完整数据集抽样误差、原分辨率收敛、多卡或性能验收。
