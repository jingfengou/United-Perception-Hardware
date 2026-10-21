# YOLO / DETR / ViT 的跨厂商差异定位

本文保留原生路径的历史结果。后续[适配层数值修复与完整复测](CROSS_VENDOR_FIX.md)单独记录显式 `portable_v1` 配置的失败阶段、固定前后向/归约顺序和新复测，不覆盖本文表格，也不改变默认 `native` 的历史结论。

## 验证口径

这是对[第一轮四模型结果](CROSS_VENDOR_ACCURACY.md)的后续定位，不覆盖原始 40 次训练。新的[预先声明计划](validation/cross_vendor_diagnosis_20261021/plan.json)保留 YOLO、DETR、ViT 原来的全部种子 131--135、数据、模型、增强、训练预算和 +/-1 个百分点的等价标准。仅对齐环境、增加观测，不用测试集调参，不改变模型算法，不关闭 Dropout 来制造通过。

正式训练仍使用各厂商原生的 PyTorch 随机算子。`--paired-masks` 只用于单独的短程因果对照，**不参与正式精度表，也不是产品中的 CPU 回退**。

**结论：已排除本轮实际训练输入不一致，定位到随机掩码映射与现有厂商栈的确定性缺口，并修复了中间层的卷积库开关遗漏；但还没有达到三模型平均精度等价的验收标准。** 不能把一次局部修正写成所有精度差异均已解决。

## 环境对齐

| 项目 | 新 NVIDIA 环境 | 沐曦 C500 环境 |
| --- | --- | --- |
| PyTorch | `2.10.0+cu126` | `2.10.0+metax3.8.1.0` |
| TorchVision | `0.25.0+cu126` | `0.25.0+metax3.8.1.0` |
| Python | 3.12.3 | 3.12.11 |
| NumPy / SciPy | 1.26.4 / 1.16.0 | 相同 |
| Pillow / OpenCV | 11.2.1 / 4.11.0.86 | 相同 |
| pycocotools / PyYAML | 2.0.10 / 6.0.3 | 相同 |

其他应用依赖见 [NVIDIA 对齐配置](../../requirements/cross-vendor-2.10-nvidia.txt)和[沐曦配置](../../requirements/metax-c500-validation.txt)。仅在新 NVIDIA 虚拟环境安装官方 wheel，重新编译本仓库 NVIDIA 扩展；保留原 NVIDIA 2.8 环境，不替换沐曦镜像的厂商 PyTorch。NVIDIA 使用已有 CUDA 12.9 编译工具链构建扩展，wheel 自带 CUDA 12.6 运行库，实际构建与模型运行均需验证，不能直接复用 2.8 的 `.so`。

[机器核对结果](validation/cross_vendor_diagnosis_20261021/aligned-packages.json)确认 23 个共享应用依赖的版本一致；两个环境的 `pip check` 通过。此次是整个环境组合的对齐，不是仅改变一个 PyTorch 版本的单变量试验。不能用新版 NVIDIA 与旧版 NVIDIA 的差值单独估计某一个包的影响。

验证启动器在两边相同地恢复旧代码使用的 `collections.Iterable`、`np.float`、`np.int` 导入别名，保留原 dtype；原始报告的 `compatibility_shims` 字段逐项记录。这只是共同的导入兼容前置，没有替换计算内核或梯度，也不代表未经过该启动器的旧 CLI 已自动兼容所有现代依赖。

这仍不是相同的二进制环境：GPU、驱动、厂商 SDK、PyTorch 本地构建和 Python 补丁版本不同。不能将剩余差异一律归因于硬件本体。PyTorch 官方也不承诺不同版本、平台间的逐位复现，参见 [2.10 可复现性说明](https://docs.pytorch.org/docs/2.10/notes/randomness.html)。

## 已定位的分叉

### ViT：Dropout 的随机映射

首步真实输入、初始化参数以及种子相同，但 12 个 MLP 的 `64 x 17 x 3072` Dropout 输出零掩码不同；这些位置的输入没有零元素，因此不是把输入自身的零误判为被丢弃。NVIDIA 升到 2.10 后仍可观察到该差异。已测较小形状的 Dropout 掩码一致，不能从小张量测试推断所有形状都一致。

GPU RNG 状态哈希相同也不意味着每种形状的随机输出相同。用实际张量比较，而不是只比较 seed 或 state。短程 `--paired-masks` 对照使用独立 CPU Generator 生成同一组掩码，在两边 GPU 上执行乘法、前向、反向和优化器更新；仍调用原生 Dropout 推进原 GPU 随机流。它只检验掩码差异是否造成首步分叉，不证明完整训练的精度等价，也没有被写入生产后端。

### YOLO：先出现常规 FP32 计算差异

第一轮早期追踪的前四批数据与 CPU 随机状态一致。Focus 的 `space2depth` 结果逐位一致，紧接的普通卷积样本最大绝对差为 `2.0862e-7`，随后 BN 样本约 `1.7881e-6`；前两步总 loss 相同，第三步出现约 `1.22e-3` 的差异。梯度及后续更新也随之分叉。

将 NVIDIA 上的 NMS / IoU / TorchVision NMS 临时全部换成现有共享实现，前四步参数状态与 loss 仍逐位复现原生 NVIDIA 对照。这排除了这些路由差异作为**该早期分叉**的原因，不是对所有输入、全部训练步骤和全部算子的普遍证明。

### DETR：版本对齐不能自动消除微小数值差异

相同输入的首步总 loss，在 NVIDIA 2.8 和 2.10 上均为约 `3.01340675`，沐曦为 `3.01340580`，差约 `9.54e-7`；后续梯度更新逐渐分叉。需要结合随机掩码观测、共同权重评估与正式多种子复测判断，不能仅凭旧结果约 -0.52 点的均值差宣称存在硬件适配错误。DETR 仍是同一份固定官方外部适配器，不是上游 UP 自有模型。

### 厂商栈：不依赖 UP 的卷积反向也存在不重复

用独立的 [纯 PyTorch 最小复现](../../tools/probe_torch_repeatability.py)，固定输入、卷积参数和上游梯度，重新构建计算图重复反向 10 次，不使用 UP、模型训练、优化器或 GPU 随机输入。在相同 C500 物理卡 5 上：

| 对照 | 相对第一次的卷积权重梯度最大绝对差 |
| --- | --- |
| NVIDIA，原 DNN 路径 | 0 |
| C500，原 DNN 路径 | 0.000152587890625 |
| C500，明确关闭 DNN 库 | 0 |

这里已设置 `use_deterministic_algorithms(True)`、`cudnn.deterministic=True`、benchmark false 和严格 FP32。说明**请求开启确定性不等于当前厂商栈的全部计算实际可重复**。所测 BN、LayerNorm、Linear、Softmax、GELU 样例的重复性另有记录，不推广到所有形状。原始报告及输入/上游梯度哈希在[诊断证据包](validation/cross_vendor_diagnosis_20261021/diagnostics/evidence.tar.gz)，摘要见[诊断 JSON](validation/cross_vendor_diagnosis_20261021/diagnostics/summary.json)。

这定位到现有软件栈中的 DNN 卷积路径，**没有证明是 GPU 物理硬件错误，也没有进一步唯一定位到某一驱动或 mcDNN 内核**。保留精确镜像、SDK/驱动和包版本，便于交给厂商复现。当前 C500 组合为驱动 3.6.11、宿主 MACA 3.5.3.18、已有镜像 `maca-pytorch:3.8.1.2-torch2.10-py312-ubuntu24.04-amd64`，镜像摘要仍为 `sha256:1a10a6cc56741f5e84da5dd6f0aa9dc4cfd420d66188e43983c9a3efd9d85ab0`。不擅自升级共享服务器驱动；该组合的厂商兼容性矩阵尚未正式确认。

### 已修复的中间层问题

原 `MetaXAccelerator` 继承了卷积控制的空实现，Runner 传入 `nocudnn=True` 时没有关闭该库。现在 NVIDIA 与 MetaX 显式共用 `_CudaConvolutionControls`；设置和临时上下文都下传到对应 PyTorch 兼容接口，异常退出也恢复状态，**没有继承 NVIDIA 的 SDK 依赖或专用算子**。

此修复只解决“关闭请求被忽略”，默认仍启用原卷积策略，不改变现有模型、学习率和参数名。上面的独立卷积样例可规避症状，**但不能据此保证整个模型可重复**：C500 相同卡 5、相同种子 131、关闭 DNN 的两次完整 DETR 训练，初始状态及全部 720 个输入相同，仍从首步 loss 出现微小分叉；最终 AP 为 77.16350 与 77.09305，差 -0.07045 点，最终参数不逐位一致。该失败对照、状态审计和指标复算全部保留，不将其中一次选作更有利的主结果。

## 正式复测

完成两个预先分别声明的阶段，不合并统计：原卷积策略三模型共 **30 次训练 / 44,600 步**；关闭卷积库的 DETR 共 **10 次 / 7,200 步**。另外上述同卡完整重复一次为 720 步，仅作重复性诊断，不放入五种子均值。合计 41 次完整训练、52,520 步，不包含短程探针和纯算子重复。

每一对的全部训练输入序列、初始权重、算法配置、应用源码、LR 轨迹及步数一致；40 次主/附加对照均完成恢复、保存状态有限值审计和预测独立复算，额外重复也完成同样审计。见[完整性记录](validation/cross_vendor_diagnosis_20261021/coverage.json)。新环境使用严格 FP32，**不代表逐位数值相同**。

精度为百分数，差值及区间为百分点；统一为沐曦减 NVIDIA，仍用 alpha=0.05、+/-1 点的配对 TOST：

| 策略 / 模型 | NVIDIA 均值 +/- 标准差 | C500 均值 +/- 标准差 | 均值差 | 差值 90% CI | 等价 |
| --- | --- | --- | --- | --- | --- |
| 原卷积策略 / UP YOLOv5-S | 33.8766 +/- 10.8502 | 28.2214 +/- 9.7837 | -5.6552 | [-23.1790, +11.8687] | 未确认 |
| 原卷积策略 / 外部 DETR R50 | 76.7692 +/- 1.1290 | 77.2962 +/- 1.0120 | +0.5270 | [-0.9889, +2.0428] | 未确认 |
| 原卷积策略 / UP ViT-B/16 | 22.3400 +/- 4.1699 | 19.8900 +/- 1.5554 | -2.4500 | [-5.9551, +1.0551] | 未确认 |
| 两边关闭卷积库 / 同一 DETR | 77.5547 +/- 0.9720 | 76.7415 +/- 0.5944 | -0.8132 | [-1.7061, +0.0797] | 未确认 |

数据见[原策略统计](validation/cross_vendor_diagnosis_20261021/native/summary.json)、[全部 15 对 CSV](validation/cross_vendor_diagnosis_20261021/native/pairs.csv)、[关闭库统计](validation/cross_vendor_diagnosis_20261021/no-dnn/summary.json)、[全部 5 对 CSV](validation/cross_vendor_diagnosis_20261021/no-dnn/pairs.csv)。各阶段目录均有 NVIDIA/MetaX 原始证据包、逐文件 SHA-256 索引、网页可读单次摘要/配置/审计。完整 loss、LR、逐步输入/RNG 哈希和最终预测在原始包内，摘要不替代原始报告。

发布前两平台隐藏 GPU 的契约与验证工具测试均为 **173 通过、0 失败、0 跳过**，原始 XML 在[诊断证据目录](validation/cross_vendor_diagnosis_20261021/diagnostics/)。硬件边界静态检查为 0 违规。最终[发布索引](validation/cross_vendor_diagnosis_20261021/publication-index.json)覆盖证据包、执行快照和可浏览摘要的文件哈希；测试通过与证据完整不等于精度等价通过。

### 相同权重的双向交叉推理

固定种子 131，把 NVIDIA 和 C500 各自训练的 YOLO / DETR / ViT 参数导出为 tensor-only 文件，核对文件与模型状态哈希，在两种卡上分别评估全部同一 holdout。**6 份冻结权重、12 次评估，全部指标字典完全一致**，也都与对应权重的训练最终报告一致。不是只比较显示时的四舍五入结果。

| 权重来源 | 模型 | NVIDIA / C500 的相同指标 |
| --- | --- | --- |
| NVIDIA | YOLOv5-S | AP 18.279858 |
| C500 | YOLOv5-S | AP 38.577327 |
| NVIDIA | DETR | AP 75.570242 |
| C500 | DETR | AP 78.361088 |
| NVIDIA | ViT-B/16 | Top-1 19.550001 |
| C500 | ViT-B/16 | Top-1 17.150000 |

这支持本轮差异主要发生在**训练形成不同权重的过程中**，不是把同一权重换到 C500 就在本次 holdout 上产生相同幅度的指标下降。它只覆盖这些权重和固定输入，不证明任意数据、所有内核或全量基准等价。固定输入的 NMS 返回下标一致；37 框样例的 IoU 最大绝对差为 `1.1921e-7`，不写成所有输出逐位一致。

### 尚未解决的范围

- 未把每个最终 AP/Top-1 差值按比例归因到某一个算子，也未唯一定位关闭 DNN 后仍存在的完整训练分叉。独立 PyTorch 复现需要厂商确认/修复当前软件栈，不能靠 UP 路由层伪造相同梯度或静默转 CPU。
- YOLO 的种子方差大，ViT 两边绝对精度低。固定小数据预算不是充分收敛，不能将现有统计当成 ImageNet/COCO 全量结果；未换训练配方、去掉随机层、选择最好 seed/epoch 或放宽阈值来获得通过。
- 本轮仍为单卡训练任务的并行排队，不是多卡通信验收；新 2.10 组合的全部专用算子、ONNX、AMP、完整 Runner 中途恢复连续性未因此获得验证。旧环境和历史成功/失败结果不覆盖、不撤销。

## 如何复现

1. 复用第一轮冻结的 CIFAR-10 / PennFudan 数据及清单，不重新抽样。两平台使用相同的绝对数据路径、源码和权重。
2. NVIDIA 在独立环境安装 `torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126`，再安装 NVIDIA 对齐配置；沐曦保留厂商环境。构建 NVIDIA 扩展后核对实际导入路径及 `pip check`。
3. 精确复现本次原策略产物时，`up/` 使用 Git 提交 `1d891db70679e0f26ff9f985460e807a5fdce78b`，叠加本轮 [NVIDIA 执行工具快照](validation/cross_vendor_diagnosis_20261021/native/nvidia-executed-tools.tar.gz) / [MetaX 执行工具快照](validation/cross_vendor_diagnosis_20261021/native/metax-executed-tools.tar.gz)；不取代对应厂商 SDK。关闭库对照使用修复后的本仓库代码，两阶段应用源码均已独立核对，见完整性记录中的 manifest。当前代码默认策略相同，但不能用新的源文件哈希冒充旧受测快照。

两边均关闭 TF32，执行正式训练，`--devices` 使用允许分配的物理卡号。

```bash
export DEFAULT_TASKS=cls:det PYTHONHASHSEED=0
export PYTORCH_DISABLE_CUDA_CUDNN_TF32=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
python tools/run_cross_vendor_accuracy.py \
  --plan docs/hardware/validation/cross_vendor_diagnosis_20261021/plan.json \
  --backend nvidia --source "$PWD" \
  --data /workspace/up-cross-vendor-20261021/data/data.json \
  --devices 0,1,2,3 --output /workspace/new-diagnosis/results
```

沐曦改为 `--backend metax`。新计划自动打开 `--audit-batches`，保存每一步输入与 pre-forward RNG 哈希，配对时要求**全部训练输入**的序列哈希一致。完整训练、checkpoint 审计、独立指标复算和 TOST 汇总使用[第一轮的相同步骤](CROSS_VENDOR_ACCURACY.md#复现步骤)，但换为本轮计划和独立的新输出目录。

关闭库的正式对照改用 `plan-no-dnn.json` 及另一个输出目录，计划会同时对两边传入 `--disable-convolution-library`，并要求训练报告实际读回 `convolution_library_enabled=false`。不要只在一边关闭、再称为相同配置。

日常 UP CLI 使用既有 `--nocudnn` 参数；程序直接构造 Runner 时设置 `config['args']['nocudnn']=True`。训练 CLI 会重建 `args`，**仅往 YAML 中写 `args.nocudnn` 不能替代命令行参数**：

```bash
python -m up train --config your_metax_model.yaml --backend dist --nocudnn
```

该参数不是已通过的精度修复建议，可能改变速度及数值路径，默认不启用。

4. 另用新目录运行诊断。两边分别执行后，将两个输出目录放到一台机器，用 `--compare LEFT RIGHT --output comparison.json` 比较；训练张量报告的数值差为明确标记的样本，哈希覆盖完整张量。

```bash
python tools/probe_cross_vendor_numerics.py --kind rng --rng-wide \
  --backend nvidia --output /workspace/new-diagnosis/rng
python tools/probe_cross_vendor_numerics.py --kind training \
  --backend nvidia --source "$PWD" \
  --data /workspace/up-cross-vendor-20261021/data/data.json \
  --model vit_b16 --seed 131 --steps 2 \
  --output /workspace/new-diagnosis/vit-native
```

同一训练诊断额外传 `--paired-masks` 做共同掩码因果对照；YOLO 额外传 `--portable-ops` 做算子实现消融。必须使用各自的新输出目录，报告均标为 `diagnostic_only`，禁止将其合并为正式完整训练结果。跨观测工具版本比较时默认拒绝不同张量键，只有显式 `--allow-schema-difference` 才比较交集，同时列出全部缺失键。

5. 相同权重的跨平台评估：用 `--export-model --checkpoint OUR_COMPARISON_STATE.pt --output model-only.pt` 从本次自己生成的可信 checkpoint 导出 tensor-only 参数；分发同一文件、核对哈希，再用 `--kind inference --checkpoint model-only.pt`，同时提供上面的 source/data/model/backend/output。导出会读取含调度器 Python 状态的完整 checkpoint，不要用于不可信文件；推理只接受 `weights_only=True` 的模型参数。

批量导出/校验使用 [run_checkpoint_swaps.py](../../tools/run_checkpoint_swaps.py)；纯 PyTorch 重复性用例可脱离 UP 运行：

```bash
python tools/probe_torch_repeatability.py --cases conv2d --repeats 10 --output native.json
python tools/probe_torch_repeatability.py --cases conv2d --repeats 10 --disable-dnn --output disabled.json
```

用新文件名保留每次结果。证据的压缩包哈希、成员哈希、完整预算、全部输入配对和源代码对应关系，可用 [summarize_diagnosis_evidence.py](../../tools/summarize_diagnosis_evidence.py) 重新核对；该脚本还检查 23 项依赖对齐，并生成可浏览摘要。数值探针张量保存在两台服务器和本地任务的 `dist/cross-vendor-diagnosis/`，仓库发布报告、比较数值、张量哈希与复现脚本，不发布训练权重、图片、SDK 或私钥。
