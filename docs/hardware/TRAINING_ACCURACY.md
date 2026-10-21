# NVIDIA 真实数据训练对比

这是早期 3/5 epoch、两个种子的历史报告。最新 Faster R-CNN 多种子完整预算，以及 ResNet18、YOLOv5-S、ViT-B/16 和外部 DETR 的对照，见[五模型报告](MULTISEED_ACCURACY.md)。本页原始失败与差异定位不因后续检查而删除或改写。

本轮要回答的是：**硬件隔离改造是否改变原来 UP 模型的训练过程和验证精度。** 不把两步合成输入测试当作精度验收。

## 对比范围

日期：2026-10-21；设备：A800-1 的 NVIDIA A800-SXM4-80GB，使用当时空闲的物理 GPU 2，逐个执行任务，不停止或占用其他任务的 GPU。

上游为 `52aaeb9ed27a490820696d7754d19f98cd6ecd11`；改造版运行时代码对应 `c7be549`。本轮新增训练工具、接口文档、清单和检查，不修改模型算法、硬件运行时或原生内核。

当前这个上游快照的 `up/` 和 `configs/` 中没有 DETR 实现或配置。因此先验证上游确实提供的 ResNet18 和 Faster R-CNN，而不是引入外部 DETR 后把它算作原框架回归。若需要 DETR，需另外确认来源/分支并增加模型接入及独立基线。

用户确认服务器没有 ImageNet/COCO 数据和预训练权重，本轮采用可下载的小规模真实数据。**这是短程训练回归，不是 ImageNet/COCO 全量训练、论文精度复现或完整收敛验收。**

| 模型 | UP 原始配置 | 数据与改动 |
| --- | --- | --- |
| ResNet18 | `configs/cls/resnet/res18.yaml` | CIFAR-10 原训练分区固定抽取 10,000 张、原测试分区固定抽取 2,000 张；10 类；64px 输入；从随机初始化训练 5 epoch，SGD lr=0.05，batch=64 |
| Faster R-CNN R50-FPN | `configs/det/faster_rcnn/faster_rcnn_r50_fpn_1x.yaml` | PennFudan 固定图像级 120/50 划分；背景+行人 2 类；320/512 短/长边；ImageNet ResNet50 预训练骨干；SGD lr=0.0025，batch=2，3 epoch |

两边完全相同的配置覆盖记录在各次运行的 `effective_config.json`。数据划分种子为 20261021；训练种子分别为 131、132。验证图像不参与训练；PennFudan 是图像级而不是场景级划分，不用于宣称场景泛化精度。

数据来源：[CIFAR-10 官方说明](https://www.cs.toronto.edu/~kriz/cifar.html)、[多伦多大学 CIFAR-10 分发](https://huggingface.co/datasets/uoft-cs/cifar10/tree/0b2714987fa478483af9968de7c934580d0bb9a2/plain_text)、[PyTorch PennFudan 数据说明](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)。网络较慢时使用固定提交的分发镜像，保存源地址和 SHA-256；CIFAR Parquet 的校验值与上游发布值一致。数据、图片和权重不提交到本仓库。

### 实际数据链路的环境要求

本次完整原版 Runner 检查补出了以下兼容问题，均在原版出现，不能归咎于硬件路由改造：

- 容器原装 `pycocotools==2.0+nv0.8.1` 的增强构造流程与 UP 的 PetrelCOCO 子类不兼容，报缺少 use_ext。检测对照通过 `pip --target` 在任务专用目录安装标准 `pycocotools==2.0.10`，两边共用，不覆盖容器系统库。
- 原配置指向的官方 `resnet50-19c8e357.pth` 使用旧归档格式，与 PyTorch 2.6 之后默认的 weights_only 加载规则冲突。准备脚本先校验官方权重，再只转换序列化格式，逐张量检查值完全一致。训练使用可安全按 weights_only 加载的新文件，不全局关闭反序列化安全限制。
- 原版锚框生成仍使用已移除的 `np.float`。检测测试启动器给两边同样补回其原本的 `float` 别名，仍使用 float64，不修改锚框公式或模型源码。分类使用的原始驱动快照另存于[验证目录](validation/accuracy_20261021/driver-resnet18.py)，每个配对内部的驱动哈希一致。

分类没有使用 COCO 数据加载/评估，保留原容器分发；检测使用上述任务内标准 COCO 库。每个原版/改造版配对内部的环境完全相同。

## 如何保证公平

1. 两边使用同一 Python、PyTorch、TorchVision 和编译环境，在独立进程导入各自源码，执行各自的 **BaseRunner、数据加载、模型、损失、优化器和原生扩展**；不是在改造版组件上只替换一个原始 ModelHelper。
2. 使用上游已有的 `DEFAULT_TASKS=cls:det` 限制无关任务导入，不伪造缺失的点云、量化或厂商 SDK。
3. 原版使用已移除的 `collections.Iterable` 和 `np.float`。测试启动器为每个配对提供同一旧别名兼容，不修改上游源码、原有 dtype 或数值公式。
4. 固定随机种子、可见设备、数据划分和完整有效配置；启用确定性算法，关闭 TF32、cuDNN benchmark，使用 FP32、0 个 DataLoader worker，避免 UUID worker 种子造成对照差异。
5. 比较初始权重哈希、每步 loss、最终模型权重和验证指标；前置要求包括数据清单、配置、步数、环境、测试驱动哈希一致。
6. 每次训练后新建 ModelHelper 和优化器，加载 checkpoint，再运行完整验证，要求恢复前后指标相同。

数值阈值预先固定为 `rtol=1e-4, atol=1e-5`；指标绝对容差 `1e-6`。Top-1/Top-5 单位为百分数，COCO-style AP 单位为 0 到 1；AP_S 等未包含对应目标的项可能为 -1，不当作有效精度分数。

## 结果文件

10 次真实数据训练全部完成，共 4,220 个优化器更新步骤；原版和改造版都完成了完整 BaseRunner 训练、推理和新建模型/优化器后的 checkpoint 加载验证。**“完成执行”与“通过精度一致性验收”是两回事。**

| 模型 | 种子 | 原版指标 | 改造版指标 | 改造版减原版，百分点 | 严格对比 |
| --- | --- | --- | --- | --- | --- |
| ResNet18 / CIFAR-10 Top-1 | 131 | 47.5000% | 47.5000% | 0.0000 | 通过 |
| ResNet18 / CIFAR-10 Top-1 | 132 | 47.2000% | 47.2000% | 0.0000 | 通过 |
| Faster R-CNN / PennFudan AP | 131 | 36.9782% | 41.1134% | +4.1352 | **未通过** |
| Faster R-CNN / PennFudan AP | 132 | 33.5100% | 34.3786% | +0.8686 | **未通过** |

分类两组的 785 步 loss 和所有最终权重均逐位相同。检测两组的 loss、权重和指标有差异，严格检查返回非零，未降低阈值或把更高分数算作通过。

### 检测差异的重复性诊断

相同源码、相同种子 131、相同初始权重、配置和数据，再各训练一次：

| 同版本重复对照 | 第一次 AP | 重复 AP | 差异，百分点 |
| --- | --- | --- | --- |
| 原版对原版 | 36.9782% | 39.8206% | +2.8424 |
| 改造版对改造版 | 41.1134% | 34.9479% | -6.1655 |

这些对照本身也未通过严格一致检查。原版重复运行的初始权重、初始验证指标相同，但第二步 loss 已出现偏差。

进一步对原生 RoIAlign 使用固定 features、RoIs 和固定上游梯度，在同一进程重复反向 10 次：原版最大梯度差 `6.67572021484375e-6`，改造版 `7.152557373046875e-6`，均不逐位相同。两边 RoIAlign 内核文件 SHA-256 相同，源码中的 atomicAdd 也保持原样；PyTorch 全局确定性开关没有让这个自定义内核变成确定性实现。

**结论：分类在本轮短程配对中未见变化；检测不能据这几次结果认定精度等价，也不能把观测到的分差归因于改造或视为收益。** 已证实存在原生反向非确定性和同版本训练漂移，但不能单凭此诊断排除其他差异。若要完成检测精度验收，仍需事先确定统计标准、更多重复/种子和完整训练预算；本轮检测验收状态保留为未通过/待进一步统计验证。

后续做了[真实批次的逐层观测和梯度回放定位](validation/accuracy_20261021/divergence/README.md)：首步各模块前向、数据、随机状态和 RoIAlign backward 输入逐位一致，最早差异出现在原生 RoIAlign 的返回梯度；仅在诊断中回放该梯度后，两版连续 3 步的全部参数梯度、loss 和更新后权重逐位一致。原版对原版也复现同一差异起点。这比“原版也会漂移”提供了更强的早期因果证据，但不将 3 步定位当作完整训练或最终 AP 等价通过，也不改动正式训练的内核。

### 可审核证据

- [分类种子 131](validation/accuracy_20261021/resnet18-seed131-comparison.json)、[分类种子 132](validation/accuracy_20261021/resnet18-seed132-comparison.json)。
- [检测种子 131](validation/accuracy_20261021/faster_rcnn-seed131-comparison.json)、[检测种子 132](validation/accuracy_20261021/faster_rcnn-seed132-comparison.json)。
- [原版重复对照](validation/accuracy_20261021/faster_rcnn-seed131-self-repeat-comparison.json)、[改造版重复对照](validation/accuracy_20261021/faster_rcnn-seed131-routed-self-repeat-comparison.json)。
- [原版 RoIAlign 重复反向](validation/accuracy_20261021/roi-align-repeatability-baseline.json)、[改造版重复反向](validation/accuracy_20261021/roi-align-repeatability-changed.json)。
- [原版源码清单](validation/accuracy_20261021/source-baseline.json)、[改造版源码清单](validation/accuracy_20261021/source-changed.json)：407/436 个 Python 源文件与对应 Git 快照在仅归一化 CRLF/LF 后相同；实际原始字节哈希另外保留，不混为同一项核对。源码一致不代表这些文件均有执行覆盖。
- 每个运行目录的 report.json / effective_config.json 保存完整 loss、指标、状态/源码/二进制哈希和有效配置；[验证目录](validation/accuracy_20261021/)中还保存数据划分、包环境和两份实际测试驱动快照。[分类驱动](validation/accuracy_20261021/driver-resnet18.py)、[检测驱动](validation/accuracy_20261021/driver-faster-rcnn.py)的文件哈希与各次报告一致；生产工具后续增加了拒绝零 epoch、空训练和未完成报告的检查，不改变训练计算。

GPU checkpoint 与完整日志保存在 A800-1 的 `/workspace/up-hardware-routing-20261021-accuracy/`，不提交数据、图片或大权重。环境准备阶段的失败尝试未作为成功训练计数。

本次证据目录通过 `.gitattributes` 固定为 LF，避免 Windows checkout 自动改写报告/驱动字节、使报告中保存的 SHA-256 失效；受测源码原有行尾差异仍单独记录。

## 复现

在具有该 NVIDIA 环境和已构建原生扩展的机器运行。另备原版、改造版两个源码目录，二者不得混用同一个 `up` 导入路径。网络准备只使用 CPU，不分配 GPU。

```bash
python tools/prepare_accuracy_data.py --output /workspace/up-accuracy-data \
  --train-size 10000 --test-size 2000
```

若原始 CIFAR 下载缓慢，可加 `--cifar-source hf-mirror`，需要额外安装 `pyarrow`。PennFudan 镜像可通过 `--pennfudan-url` 显式指定，本轮固定提交的地址记录在数据清单中；没有静默更换数据源。

若使用本轮带 NVIDIA 增强 COCO 库的容器，检测前增加以下任务内覆盖；不要替换厂商 PyTorch/TorchVision：

```bash
python -m pip install --target /workspace/up-accuracy-compat --no-deps pycocotools==2.0.10
export PYTHONPATH=/workspace/up-accuracy-compat${PYTHONPATH:+:$PYTHONPATH}
```

```bash
CUDA_VISIBLE_DEVICES=2 python tools/validate_training_accuracy.py \
  --source /workspace/up-baseline --data /workspace/up-accuracy-data/data-cifar10.json \
  --model resnet18 --seed 131 --epochs 5 --output /workspace/results/resnet18-baseline
CUDA_VISIBLE_DEVICES=2 python tools/validate_training_accuracy.py \
  --source /workspace/up-routed --data /workspace/up-accuracy-data/data-cifar10.json \
  --model resnet18 --seed 131 --epochs 5 --output /workspace/results/resnet18-routed
python tools/validate_training_accuracy.py \
  --compare /workspace/results/resnet18-baseline /workspace/results/resnet18-routed \
  --output /workspace/results/resnet18-comparison.json
```

Faster R-CNN 将 model 改为 `faster_rcnn`、data 改为 `data.json`、epochs 改为 3；另用 seed=132 重复两组。各次输出目录必须为空，禁止不小心加载上次权重续训。比较命令发现不一致时返回非零，不自动放宽阈值。

完整 ImageNet/COCO 验收仍需独立数据准备、原始任务配置、同等训练预算、多随机种子及验证标准；本工具的 CIFAR/PennFudan 配置覆盖不能当作那两个全量任务的训练配方。国产硬件仍需对应真机适配和验证。
