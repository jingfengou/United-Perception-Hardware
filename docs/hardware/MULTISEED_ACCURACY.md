# NVIDIA 五模型多随机种子训练对照

## 要回答的问题

选择 `runtime.accelerator.backend: nvidia` 后，硬件路由改造是否影响训练指标？本轮将先前的 3/5 epoch 检查扩展为**完整执行预先声明的训练预算、多种子配对及同版本重复对照**。不使用梯度回放、梯度舍入、替换 RoIAlign 或静默 CPU 回退。

五模型的预设预算已全部执行完成，训练 loss、保存的模型/优化器状态有限，推理和权重恢复检查通过。四个模型的配对 loss、最终模型状态及指标逐位相同；Faster R-CNN 通过平均 AP 的预设统计等价检验，但不逐步一致。AP/Top-1 均以下表的百分数形式表示，差值和区间单位是**百分点**。

| 模型 | 配对种子数 | 原版均值 +/- 标准差 | NVIDIA 路由版均值 +/- 标准差 | 均值差 | 均值差的 90% CI | 预设 +/-1 点等价检验 | 逐步数值检查 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Faster R-CNN / AP | 10 | 60.4172 +/- 1.0198 | 60.4419 +/- 1.1244 | +0.0247 | [-0.7103, +0.7596] | 通过，TOST p=0.01891 | 0/10 通过 |
| ResNet18 / Top-1 | 5 | 66.2200 +/- 0.5404 | 66.2200 +/- 0.5404 | 0 | [0, 0] | 通过 | 5/5 通过且 loss/最终模型状态逐位相同 |
| YOLOv5-S / AP | 5 | 26.0174 +/- 10.2191 | 26.0174 +/- 10.2191 | 0 | [0, 0] | 通过 | 5/5 通过且 loss/最终模型状态逐位相同 |
| 官方 DETR 适配器 / AP（修正优化器配方） | 5 | 77.6032 +/- 0.6539 | 77.6032 +/- 0.6539 | 0 | [0, 0] | 通过 | 5/5 通过且 loss/最终模型状态逐位相同 |
| ViT-B/16 / Top-1 | 5 | 22.3400 +/- 4.1699 | 22.3400 +/- 4.1699 | 0 | [0, 0] | 通过 | 5/5 通过且 loss/最终模型状态逐位相同 |

**Faster R-CNN 结论的含义**：在本次固定 PennFudan 划分、12 epoch、FP32 配方下，证据支持两个版本的**平均 AP 在 +/-1 个百分点内统计等价**，不是权重逐位一致，也不是每一个种子的分差都小于 1 点。10 对中的分差范围为 -1.3197 到 +2.6651 点，不能选其中某一次来宣称提升或退化。描述性 95% CI 为 [-0.8823, +0.9317]，不替代预先指定的 90% CI 判定标准。

### 同版本重复对照

这 6 次额外训练同样完整跑完 12 epoch，但不混入主要 10 对样本：

| 版本 | Seed | 第一次 AP | 重复 AP | 重复减第一次，百分点 |
| --- | --- | --- | --- | --- |
| 原版 | 131 | 62.5426 | 62.8892 | +0.3466 |
| 原版 | 132 | 61.3802 | 60.9897 | -0.3905 |
| 原版 | 133 | 59.7319 | 57.1745 | -2.5574 |
| 路由版 | 131 | 61.2229 | 60.6010 | -0.6219 |
| 路由版 | 132 | 60.9330 | 60.4615 | -0.4715 |
| 路由版 | 133 | 58.8907 | 59.5971 | +0.7064 |

原版同一种子也有最大约 2.56 点的重复漂移。该现象用于呈现原生非确定性，不作为免除精度检查、扣减方差或放宽容差的理由。

### 原始输入尺寸补充功能检查

小数据训练之外，路由版另外使用原生网络配置、原始分类头规模和下列分辨率各执行 2 步合成训练：

| 模型 | 输入与分类头 | 前后向有限值 | 推理与新建模型/优化器恢复 | 原 ModelHelper 参数键与前向兼容 |
| --- | --- | --- | --- | --- |
| YOLOv5-S SiLU | batch2 / 640x640，80 个前景类（配置 num_classes=81 含背景） | 通过；分类 loss 约 0.204/0.206，未跳过多类分支 | 通过 | 通过 |
| ViT-B/16 | batch2 / 224x224，1000 类、197 tokens | 通过 | 通过 | 通过 |

这一项只说明原始尺寸/分类分支的功能可执行和旧 Helper 兼容，不是原始尺寸的真实数据精度、完整原版独立进程对照或性能验收。独立原版/改造版完整 Runner 对照仍以主表的实际训练分辨率为准。

## 范围不能混淆

- 数据仍是 CIFAR-10 固定 10,000/2,000 子集与 PennFudan 固定 120/50 图像级划分。服务器没有 ImageNet/COCO 全量数据，本轮不是这两个全量基准的论文精度或收敛复现。
- “完整”指每次均跑完以下预算，没有提前停止或只跑几步；不表示已经证明数据集上的最优收敛，更不表示所有 UP 模型均已验证。
- ResNet18、Faster R-CNN、YOLOv5-S、ViT-B/16 是上游 UP 自有模型。**该 UP 快照没有 DETR**，本轮使用外部官方 DETR 的同一个适配器接入两边 UP Runner，单列为新增模型验证。
- ViT 保留 12 层、768 维、12 头和 16px patch 的完整 Base 网络，但 CIFAR 训练输入为 64px、位置编码对应 17 个 token，不是原 224px/197-token ImageNet 配方；YOLO 使用 320px mosaic 而非原 640px COCO 输入。缩小输入不能用于推断原分辨率的性能或精度。
- 指标的统计等价不等于训练 loss、权重逐位一致。此前检测逐位不一致和 RoIAlign 反向非确定性的记录保留，不通过放宽原检查阈值改写历史结论。
- ViT 这套 64px、10,000 张、20 epoch、从零初始化且保留强增强的配方绝对 Top-1 较低，只用于改造前后回归；不能因两边一致而宣称已经达到可用精度或充分收敛。
- 各模型的预训练来源、输入和预算不同，表格只比较同一模型的原版/路由版，不能用作跨模型优劣排行榜。

## 预先固定的计划

在新增训练结果产生前固定[计划文件](validation/multiseed_20261021/plan.json)，不按结果改种子、容差或训练预算。

| 模型 | 数据 | Epoch | 每次优化器步数 | 配对种子 | 额外同版本重复 |
| --- | --- | --- | --- | --- | --- |
| UP Faster R-CNN R50-FPN | PennFudan 120/50 | 12 | 720 | 131-140，共 10 对 | 131/132/133，原版和改造版各再跑一次 |
| UP ResNet18 | CIFAR-10 10,000/2,000 | 20 | 3,140 | 131-135，共 5 对 | 无 |
| 官方 DETR R50 经同一外部适配器 | PennFudan 120/50 | 12 | 720 | 131-135，共 5 对 | 无 |

原计划 46 次单卡训练、57,320 个优化器步骤已完成。用户随后增加 YOLO/ViT，并发现初版 DETR 的统一学习率配方在两边都严重遗忘，另固定[追加计划](validation/multiseed_20261021/extension-plan.json)，不覆盖原计划或删除不利结果：

| 追加模型 | 数据 | Epoch | 每次优化器步数 | 配对种子 |
| --- | --- | --- | --- | --- |
| UP YOLOv5-S SiLU | PennFudan 120/50 | 300 | 600 | 131-135，共 5 对 |
| UP ViT-B/16，64px CIFAR token grid | CIFAR-10 10,000/2,000 | 20 | 3,140 | 131-135，共 5 对 |
| 官方 DETR R50，分组学习率 | PennFudan 120/50 | 12 | 720 | 131-135，共 5 对 |

追加 30 次、44,600 步也已完成。五模型主要对照含同版本重复共 66 次、94,720 步；保留的初版 DETR 另有 10 次、7,200 步诊断，不把两个 DETR 配方混合成更多种子。GPU 预检查及失败启动不计入这些数量。

不同实验可以在独立 GPU 上同时执行，但同一种子的一对原版/改造版必须在同一物理 GPU 顺序运行；这不是多卡 DDP 训练验收。

所有运行均使用最后一个 epoch 的 checkpoint，不挑最佳 epoch、最好种子或有利的重复。seed 奇偶交替决定两版本的先后顺序。同版本重复只用于描述非确定性，不加入主要配对样本、抵消分差或降低标准差。

### 训练配方

- Faster R-CNN：原 UP R50-FPN/256 通道 neck/1024 FC，2 类；320/512 短/长边，batch=2，FP32，官方 ImageNet R50 骨干初始化，SGD lr=0.0025、momentum=0.9、weight_decay=0.0001；60 步 warmup，epoch 8/11 衰减为 0.1 倍。
- ResNet18：原 UP ImageNet-style stem 不变，10 类，64px，batch=64，随机初始化，SGD lr=0.05；epoch 12/18 衰减为 0.1 倍，无 warmup。不是标准 CIFAR 小卷积 stem 的论文配方。
- DETR：标准 R50、6 层 encoder/6 层 decoder、hidden_dim=256、8 个注意力头、100 个 queries，开启 5 个辅助解码层损失。官方 COCO checkpoint 初始化，分类头抽取 person/no-object 两行；AdamW 所有可训练参数 lr=1e-4、weight_decay=1e-4，梯度范数裁剪 0.1，epoch 8/11 衰减。此处统一 lr 的小数据微调配方不等同于官方 300/500 epoch COCO 配方。
- 追加 DETR 配方：模型、损失、数据、初始化、12 epoch 和梯度裁剪不变；采用官方式骨干 1e-5、其余参数 1e-4 的 AdamW 分组，关闭 UP 的 batch LR 放大。不是宣称复现官方 COCO 训练。
- YOLOv5-S：原 UP DarkNetv5/DarknetPAFPN/YoloV5Head/postprocess，单前景类，随机初始化；320px mosaic、random perspective/HSV/flip，batch=64（末批 56），保留原 SGD 参数分组、lr=0.00015625、momentum=0.937、Nesterov、weight_decay=0.0005，3 epoch warmup、300 epoch cosine，EMA 关闭；每 10 epoch 评估，主要指标仍取最后一个 epoch。120 张训练图使 300 epoch 仅为 600 步，不是 300 epoch COCO 工作量。
- ViT-B/16：原 UP 12 层 Base 架构、64px、10 类、随机初始化、batch=64、20 epoch；保留 AutoAugment、Cutmixup、label smoothing 0.1、dropout/attention dropout/drop path 0.1，AdamW lr=0.00075、weight_decay=0.05，首 epoch warmup、cosine 衰减，不按 batch 放大。

上面的 lr 为配置值，**保留原 UP 的 batch-size 缩放规则**：检测配置默认 scheduler 按全局 batch=2 缩放，Faster R-CNN 在 60 步内由 0.0025 升到 0.005，末段约为 5e-5；DETR 无 warmup 时对应有效初始 lr=2e-4。分类原配置使用 no_scale_lr，不放大 0.05。每次报告都保存实际逐步学习率，两边必须完全一致，不只核对 YAML 中的配置值。

分类固定使用原 train/test 分区；检测的 50 张 holdout 图像不参与更新，但其图像级划分不能证明跨场景泛化。数据和预处理细节沿用[上一轮报告](TRAINING_ACCURACY.md)。两轮预算不同，不能把旧的 3 epoch AP 混进新的 12 epoch 统计。

## 对照完整性

两边分别导入原版和改造版整个 UP，执行各自 BaseRunner、ModelHelper、数据加载、损失、优化器、评估与原生扩展。上游冻结在 `52aaeb9ed27a490820696d7754d19f98cd6ecd11`；改造版 `up/` 运行时内容与 `092f428` 一致。本轮新增验证工具、测试和文档，不改 `up/` 或 CUDA 内核。

同一配对必须一致：有效配置、原始数据清单、初始模型状态、初始随机状态、首个真实 batch、首次前向随机状态、学习率完整轨迹、训练步数、驱动哈希和设备/软件环境。checkpoint 需新建模型和优化器加载，再次评估与保存前一致。任何前置项不匹配或预算不完整都拒绝纳入统计。

保留 FP32、固定 Python/NumPy/Torch 种子、`PYTHONHASHSEED=0`、0 个 DataLoader worker、关闭 TF32/cuDNN benchmark、开启 PyTorch 确定性请求；原生 RoIAlign 的 atomicAdd 非确定性仍按原样保留，详情见[首次差异定位](validation/accuracy_20261021/divergence/README.md)。

环境为 A800-1 的 NVIDIA A800-SXM4-80GB、Python 3.12.3、PyTorch `2.8.0a0+5228986c39.nv25.06`、TorchVision `0.22.0a0+95f10a4e`、CUDA 12.9、NumPy 1.26.4、SciPy 1.15.3。检测任务及本轮统一导入使用任务内的标准 pycocotools 2.0.10，不覆盖容器系统库。`collections.Iterable`、`np.float` 兼容别名同样作用于两边；追加模型启动器还恢复 AutoAugment 所需的 `np.int = int` 旧别名，保持原 dtype，不替换增强算法。

### 调度与数据增强探针

YOLO 的耗时主要在 CPU 数据增强。独立的[OpenCV 线程探针](validation/multiseed_20261021/opencv-thread-probe.json)比较了默认 48 线程和 2 线程：5 个种子各 8 个真实增强 batch，图像、标签、元信息及后续 Python/NumPy/Torch RNG 哈希均相同，但每组耗时均约 15.4-15.6 秒，没有明显提速。因此**正式训练没有采用该修改**，保持原环境与启动器哈希不变。

为使用 ViT 完成后释放的 GPU0，仅替换了 YOLO 的排队进程：保留正在执行的 seed131/132 训练子进程继续到结束，将未开始的 seed135 改排到 GPU0，seed133/134 仍用 GPU1/2。没有终止或重跑在途训练，没有改变任何预算、超参数、数据、随机种子或数值容差；每一对始终在同一物理卡顺序执行。原执行清单、排队调整记录和[实际续接脚本](validation/multiseed_20261021/drivers/resume_yolo_schedule.py)一并保留，不能将排队进程的退出码误读为模型训练失败。

### 两类判断

1. **逐步数值检查**沿用 `rtol=1e-4, atol=1e-5, metric_atol=1e-6`，比较所有 loss、最终权重和指标；不因统计检验通过就改写该检查。
2. **主要指标统计等价**对每个种子计算“改造版减原版”，Top-1 与 AP 都转换为百分点。预设等价范围为 `(-1, +1)` 个百分点，做配对 Student t 的 TOST，alpha=0.05。均值差的 90% CI 必须完全处于该范围内，两项单侧检验均 p<0.05，才判为本数据/预算上的等价；另报描述性 95% CI。

“逐位相同”仅指记录到的逐步 loss、最终 `model.state_dict()` 和评估值，不代表已保存并比较每一步全部激活、梯度和优化器张量。权重恢复检查为新建模型/优化器加载与重新评估，不包括恢复后继续完整训练一个 epoch 的连续性验收。

均值接近 0 或普通差异检验 p>0.05 **不能单独证明等价**。若 CI 跨出边界，结论是“未能确认预设范围内等价”，不自动当成改造导致退化。该推断依赖训练种子配对差值的独立性/近似正态假设，且只反映固定 holdout 上的训练随机性，不包含数据抽样误差，不外推到完整 COCO/ImageNet 或其他硬件。

方法参考：[TOST 配对检验](https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.ttost_paired.html)、[SciPy 单侧均值检验](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html)。实现与独立 SciPy 单侧检验交叉核对，并测试零方差、边界、均值零但高方差、空/非有限记录等情形。

## DETR 的来源与限制

外部源码来自 [facebookresearch/detr](https://github.com/facebookresearch/detr/tree/29901c51d7fe8712168b8d0d64351170bc0f83e0)，固定提交 `29901c51d7fe8712168b8d0d64351170bc0f83e0`，Apache-2.0。官方权重为 `detr-r50-e632da11.pth`，完整 SHA-256：`e632da11ec76ae67bac2f8579fbed3724e08dead7d200ca13e019b197784eadc`。

[适配器](../../tools/accuracy_detr_adapter.py)使用官方模型、匈牙利匹配、SetCriterion、Transformer 和后处理，不手写简化 DETR。额外处理 UP 的 batch/标注/预测结构、padding mask、缩放后的 inclusive 框与 DETR half-open 框转换，以及下述标签损失的等价展平；两边使用完全相同的适配器。构造时验证全部外部 Python 源文件与权重哈希。

匈牙利匹配沿用官方实现中的 CPU SciPy 求解，这是原算法本来就有的步骤，不是后端缺少能力时新增的 CPU 回退；网络前后向和优化器仍在选定 GPU 上执行。

DETR 报告的 original_config 字段表示复用了 Faster R-CNN 的 COCO 数据加载模板，**不是实际网络类型**；实际 effective_config.json 的 net 只有 DetrForUP，优化器/损失均已替换，model_provenance 和 external_model 字段单独标注官方来源。

权重先校验官方文件，再仅转换为 tensor-only 序列化格式，逐张量验证值不变。不在训练中下载文件，不全局关闭 weights_only 安全加载。官方 Backbone 的隐式 ImageNet 权重下载由显式 `weights=None` 构造加完整官方 DETR 权重加载替代，网络结构和参数键不变。

这是固定单前景类 person、单设备的验证适配器，只适用于本次 PennFudan 配方，不是通用多类 COCO 或完整 UP DETR 产品化接入。没有验证它的分布式/ONNX/国产硬件路径；官方 util 中未被单卡路径调用的 CUDA 分布式辅助函数也没有因此获得跨厂商支持。

### 本轮 DETR 的环境兼容修正

首次两次 GPU 启动均在第一步交叉熵前向报 `nll_loss2d_forward_out_cuda_template` 缺少确定性实现，尚无优化器更新。这两次作为 setup failure 保留，不计入完成训练数量。CPU 预检查本来可以执行，但不能代替 GPU 链路验证。

适配器 v2 仅将类别损失的 logits 从 `[B,Q,C]` 展平为 `[B*Q,C]`，labels 同样展平，使用当前 Torch 支持的二维 NLL 路径；类别权重、加权平均、匹配和其余官方 SetCriterion 行为不变。两边均使用同一 v2，不关闭严格确定性设置，也不做 CPU 回退。

独立 CUDA 数值检查中，与原三维表达式相比，loss 差 `5.960464477539063e-8`、logits 梯度最大差 `9.313225746154785e-10`，均在既定数值容差内；二维路径重复 5 次梯度逐位相同，并完成整个标准 DETR 的 GPU 反向/AdamW 预检查。参考三维表达式仅在这个独立数值探针中临时允许非确定性，正式训练仍保持严格确定性。重排损失与梯度还由 float32/float64 单元测试交叉验证。

### 初版 DETR 配方的失败不隐藏

初版统一学习率、UP batch 放大后的 5 对训练均完整执行，原版/路由版 loss、最终模型状态和指标逐位相同，但平均最终 AP 仅 4.2321%，远低于初始官方权重约 74.7625%。因此它证明了该路径的路由一致性，却**不构成正常微调或精度合格**。这组数据完整保留在原计划结果中，不用来充当主表的 DETR 训练通过。

追加配方纠正官方式分组学习率和 UP 缩放的差异；这是根据失败现象作出的显式配方修正，不是已经通过消融试验证明唯一原因，更不是硬件路由修复。追加前固定全部 5 个种子、预算、最终 epoch 选择和原数值容差，不挑选最好结果。

## 复现

先准备与本轮兼容的 NVIDIA PyTorch/编译环境，在**两个独立源码目录**构建原生扩展；原版 checkout 上述冻结提交，改造版 checkout 本报告对应提交。不要把任一 `up` 目录混进另一进程的 PYTHONPATH。基础安装和编译方式见[主 README](../../README.md)。

在改造版根目录执行，以下路径可以整体替换。网络准备阶段不使用 GPU：

```bash
python tools/prepare_accuracy_data.py --output /workspace/up-accuracy-data \
  --train-size 10000 --test-size 2000 --cifar-source hf-mirror \
  --pennfudan-url https://raw.githubusercontent.com/aws-samples/aws-stepfunctions-byoc-mlops-using-data-science-sdk/df44f158c2a2dafc6a50d60c80176097635d29b1/PennFudanPed.zip
python -m pip install scipy==1.15.3
python -m pip install --target /workspace/up-accuracy-compat --no-deps pycocotools==2.0.10
git clone https://github.com/facebookresearch/detr.git /workspace/official-detr
git -C /workspace/official-detr checkout 29901c51d7fe8712168b8d0d64351170bc0f83e0
python tools/prepare_detr_accuracy.py --source /workspace/official-detr \
  --data /workspace/up-accuracy-data/data.json \
  --output /workspace/up-accuracy-data/data-detr.json \
  --weights-dir /workspace/up-accuracy-data/weights
```

若不使用 CIFAR Parquet 镜像，可使用原始分发；镜像方式还需已有 pyarrow 19.0.1。不要为验证覆盖厂商 Torch/TorchVision。旧 PyTorch/新 TorchVision 等任意版本混搭不保证可运行。

```bash
python tools/run_multiseed_accuracy.py \
  --plan docs/hardware/validation/multiseed_20261021/plan.json \
  --baseline /workspace/up-baseline --routed /workspace/up-routed \
  --data /workspace/up-accuracy-data/data.json --compat /workspace/up-accuracy-compat \
  --models faster_rcnn --devices 2,3 --output /workspace/up-multiseed-results
python tools/run_multiseed_accuracy.py \
  --plan docs/hardware/validation/multiseed_20261021/plan.json \
  --baseline /workspace/up-baseline --routed /workspace/up-routed \
  --data /workspace/up-accuracy-data/data.json --compat /workspace/up-accuracy-compat \
  --models resnet18 --devices 0,1 --output /workspace/up-multiseed-results
```

前两组可以在不同空闲卡上同时运行。下面保留的是初版 DETR 失败配方的完整诊断复现，**它不是推荐的微调配方**；最简单的复现顺序是等前两组全部结束后执行：

```bash
python tools/run_multiseed_accuracy.py \
  --plan docs/hardware/validation/multiseed_20261021/plan.json \
  --baseline /workspace/up-baseline --routed /workspace/up-routed \
  --data /workspace/up-accuracy-data/data-detr.json --compat /workspace/up-accuracy-compat \
  --models detr --devices 1,2,3 --output /workspace/up-multiseed-results
python tools/run_multiseed_accuracy.py --summarize \
  --plan docs/hardware/validation/multiseed_20261021/plan.json \
  --output /workspace/up-multiseed-results
```

### YOLO、ViT 与修正后的 DETR

追加计划使用 `multiseed_v2` 和单独输出目录，不能拿它覆盖旧报告后继续训练。YOLO 两边使用同一份锚框文件，避免源码根路径不同导致配置不公平。准备步骤只拷贝上游锚框 JSON 并记录哈希，不重新聚类或更改锚框：

```bash
python tools/prepare_multiseed_extension.py --source /workspace/up-baseline \
  --data /workspace/up-accuracy-data/data-detr.json \
  --output /workspace/up-accuracy-data/data-five-models.json
python tools/run_multiseed_accuracy.py \
  --plan docs/hardware/validation/multiseed_20261021/extension-plan.json \
  --baseline /workspace/up-baseline --routed /workspace/up-routed \
  --data /workspace/up-accuracy-data/data-five-models.json --compat /workspace/up-accuracy-compat \
  --models vit_b16 --devices 0,3 --output /workspace/up-five-models-results
python tools/run_multiseed_accuracy.py \
  --plan docs/hardware/validation/multiseed_20261021/extension-plan.json \
  --baseline /workspace/up-baseline --routed /workspace/up-routed \
  --data /workspace/up-accuracy-data/data-five-models.json --compat /workspace/up-accuracy-compat \
  --models yolov5s --devices 1,2 --output /workspace/up-five-models-results
```

上面是两个独立进程的命令示例，只有两组 GPU 都由本任务分配且空闲时才并行启动。等显式选择的卡空闲后运行修正配方 DETR：

```bash
python tools/run_multiseed_accuracy.py \
  --plan docs/hardware/validation/multiseed_20261021/extension-plan.json \
  --baseline /workspace/up-baseline --routed /workspace/up-routed \
  --data /workspace/up-accuracy-data/data-five-models.json --compat /workspace/up-accuracy-compat \
  --models detr --devices 3 --output /workspace/up-five-models-results
python tools/run_multiseed_accuracy.py --summarize \
  --plan docs/hardware/validation/multiseed_20261021/extension-plan.json \
  --output /workspace/up-five-models-results
```

每个汇总文件只包含对应计划中的模型；原计划和追加计划不能把同名 DETR 合并。主表的 Faster R-CNN/ResNet 来自原计划汇总，YOLO/ViT/修正 DETR 来自追加计划汇总。

原始分辨率的补充功能检查可以单独执行；它不替代上述真实数据训练：

```bash
DEFAULT_TASKS=cls:det CUDA_VISIBLE_DEVICES=0 python tools/validate_models.py \
  --baseline /workspace/up-baseline --device 0 --model yolov5s --output /tmp/yolo-native-shape.json
DEFAULT_TASKS=cls:det CUDA_VISIBLE_DEVICES=0 python tools/validate_models.py \
  --baseline /workspace/up-baseline --device 0 --model vit_b16 --output /tmp/vit-native-shape.json
```

仅重新核算已发布的统计时不需要 GPU、图片或大 checkpoint；这只复核已记录证据，**不是重新训练或独立验证未公开的权重数值**。复制到新的临时目录再汇总，避免覆盖仓库的冻结记录。例如在已有 SciPy 1.15.3 的环境中：

```bash
scratch=$(mktemp -d)
cp -a docs/hardware/validation/multiseed_20261021/results "$scratch/original"
cp -a docs/hardware/validation/multiseed_20261021/five-models-v2/results "$scratch/extension"
python tools/run_multiseed_accuracy.py --summarize \
  --plan docs/hardware/validation/multiseed_20261021/plan.json --output "$scratch/original"
python tools/run_multiseed_accuracy.py --summarize \
  --plan docs/hardware/validation/multiseed_20261021/extension-plan.json --output "$scratch/extension"
```

两个计划的全部原始记录现均已齐备，可执行以上完整汇总。专用训练启动器固定了小数据类别数/输入/配方，不能直接代替正式 ImageNet/COCO 的生产训练配置。

`--devices` 是明确分配的物理设备号，每卡一个工作进程。同一 seed 的配对和重复分配在同一卡。实际运行脚本先检查显存占用；生产工具后续进一步收紧为同时检查活跃计算进程，即使显存占用小也拒绝启动，不停止他人进程。

调度器可以重新执行原命令核对已完成的 report 后继续，前提是驱动、数据、计划和设备安排未变化。已有不完整输出时会报错，不能自动删除、覆盖或伪装成成功续训；需保留失败日志、核实原因，再使用新的空输出目录。严格数值比较失败返回 1 是检测差异的记录，不是允许忽略训练错误；前置公平性失败仍会停止整组。

## 证据

结果已保存在 [validation/multiseed_20261021](validation/multiseed_20261021/)：计划、环境和数据/源码清单、每次完整配置与报告、配对比较、统计汇总以及实际执行的驱动快照。可先看[归档索引](validation/multiseed_20261021/README.md)、[五模型 JSON 汇总](validation/multiseed_20261021/five-model-summary.json)或[逐种子 CSV](validation/multiseed_20261021/five-model-pairs.csv)。图像、大权重和完整 GPU checkpoint 不提交 Git。

117 项 CPU 契约/协议测试通过，硬件边界检查为 0 违规，接口快照核对为 30 个运行时成员、73 个路由操作；[代码提交的 GitHub CI](https://github.com/jingfengou/United-Perception-Hardware/actions/runs/39557740541)也在 Python 3.10/3.12、CPU PyTorch 2.6 上通过。保存的全部 76 次完整训练状态（含 10 次初版 DETR 诊断）均做了模型和优化器有限值扫描；这些检查不改写 Faster R-CNN 的逐步不一致结论。

[实际驱动快照](validation/multiseed_20261021/drivers/)的文件哈希与运行报告/执行清单一致。生产调度器后续只加强 GPU 占用、模型身份和跨种子源码/数据一致性检查；训练驱动、适配器、统计公式、预算和阈值不变。汇总使用强化检查后的工具，拒绝将混合数据或不同源码的种子合并。

服务器原始日志/权重位于 A800-1 的 `/workspace/up-hardware-routing-20261021-multiseed/`。本轮不增加三家国产硬件的真机支持声明。
