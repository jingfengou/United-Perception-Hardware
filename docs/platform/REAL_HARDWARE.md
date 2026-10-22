# 平台拆分前后真机对照

日期：2026-10-22。**本轮只做平台拆分的短程功能与数值回归，没有恢复此前暂停的多随机种子精度复测。**

拆分前基线：`0c9a43c7493563f837ad9dd27761cb00bf5fa1ce`。拆分后以 `b13614a938001642be193d12f449dfdb808a84d9` 为基础，加本次两处路由检查优化。实际执行源码以证据中的归一化 manifest 为准，不以文档提交时间代替源码身份。

## 1. 结论

这里比较的是**同一厂商、同一环境内拆分前后**，不是 NVIDIA 与沐曦之间的精度等价。

| 平台 | 模型 | 训练 / 训练内 eval / 恢复 / 独立推理功能 | 拆分前后严格数值比较 | 同权重的新旧推理 |
| --- | --- | --- | --- | --- |
| NVIDIA A800 | UP ResNet18 | 全部通过 | 全部通过，最终权重逐位相同 | 逐位相同 |
| NVIDIA A800 | UP YOLOv5-S SiLU | 全部通过 | 同上 | 逐位相同 |
| NVIDIA A800 | 外部官方 DETR R50 适配器 | 全部通过 | 同上 | 逐位相同 |
| NVIDIA A800 | UP ViT-B/16 | 全部通过 | 同上 | 逐位相同 |
| 沐曦 C500 | UP ResNet18 | 全部通过 | 未通过：第三步梯度、优化器状态超差 | 逐位相同 |
| 沐曦 C500 | UP YOLOv5-S SiLU | 全部通过 | 未通过：第三步梯度、优化器状态超差 | 逐位相同 |
| 沐曦 C500 | 外部官方 DETR R50 适配器 | 全部通过 | 未通过：训练后 eval / 推理预测超差 | 逐位相同 |
| 沐曦 C500 | UP ViT-B/16 | 全部通过 | 预设容差内通过，不是逐位相同 | 逐位相同 |

**NVIDIA 的四模型检查通过。不能把沐曦的“能跑通”写成四模型训练数值全部一致。** 沐曦旧版自身重复运行也出现了同量级差异，说明这些现象不需要平台拆分就能发生；本次有限对照不能证明所有差异都只来自某一个底层内核。

另在 A800 分别重新编译旧 `up.extensions._C` 和新 `up_platform.backends.nvidia._C`，32 项已有原生算子前向/所需梯度比较全部通过，无基线缺项、无新增回归。43 个 C++/CUDA/header 源文件内容一致，仅路径迁移。此次不声称原生非确定性算子必须逐位一致。

## 2. 实际跑了什么

- 每份源代码、每个模型独立启动进程。分类用 CIFAR-10，检测用 PennFudan，固定取已有冻结划分的前 8 张训练图、前 4 张验证图，没有重新抽样或下载另一份数据。
- 单种子 `131`，batch size 2，每模型 3 次真实 Runner 更新；第 2、3 步后分别调用真实 `BaseRunner.evaluate()`。不是完整 epoch 或收敛验证。
- ResNet18 输入 64；ViT 保留 12 层、768 宽、patch 16，测试 64 输入；YOLO 使用已有 320 小图配方；DETR 使用 320 短边、512 最大边的既有适配器配方。DETR 不是 UP 原生模型。
- 记录相同初始权重、每步输入及 RNG 哈希，比较 loss、完整参数梯度、权重与 BN buffers、优化器/调度器状态、eval 预测、重新构造后的模型与优化器恢复。
- 新进程使用 `inference` 用途，通过真实 `BaseInference` 加载权重、预处理一张真实图片并执行 `predict()`；另对训练验证保存的同一输入批次比较输出。
- eval 中确认 purpose/provider 始终为 training；返回后逐模块状态恢复，权重、梯度、optimizer、scaler 不更新，参数身份不变。第 3 步确实继续训练。
- 沐曦额外做了四次**同种子旧版自身重复训练**，以及四模型“旧版 checkpoint -> 新版独立推理”对照。它们不是多随机种子精度任务。

全程为默认 `native` 数值路径，严格 FP32 控制，关闭 TF32，请求确定性算法；**没有启用 `portable_v1`、FP64 替代计算、AMP、分布式或模型计算的 CPU 回退**。CPU 的权重/梯度保存与比较属于审计。

固定判据为 `rtol=1e-4, atol=1e-5`；整数、结构、形状、dtype 必须匹配，非有限值失败。未因沐曦结果超差放宽容差。两边厂商 Torch 构建不同，只要求各平台内部新旧代码使用同一个解释器和依赖环境。

## 3. 沐曦的差异证据

所有对照的初始权重、逐步输入和 RNG 哈希相同。以下是第三步梯度的最大绝对差，以及最终 eval 预测张量的最大绝对差；不是数据集精度变化。

| 模型 | 梯度：旧版 vs 新版 | 梯度：旧版 vs 旧版重复 | eval：旧版 vs 新版 | eval：旧版 vs 旧版重复 |
| --- | ---: | ---: | ---: | ---: |
| ResNet18 | 0.00408985 | 0.00408948 | 0.00090790 | 0.00102234 |
| YOLOv5-S | 0.00003088 | 0.00005833 | 0.00001526 | 0.00001526 |
| DETR | 0.00000158 | 0.00000158 | 0.04592896 | 0.03353882 |
| ViT-B/16 | 0.00000128 | 0.00000113 | 0.00000197 | 0.00000191 |

ResNet/YOLO 的梯度超差在旧版重复中复现；DETR 旧版重复的训练后预测也超差。尽管请求 `deterministic_algorithms=True`，该厂商环境仍未实现这些测试的逐次训练可重复性。**这支持“原生训练重现性是重要来源”，不是对厂商所有算子或长期精度的完整归因。**

用同一份旧版训练权重排除训练轨迹差异后，四模型的 batch 预测和真实图片预测在新旧推理路径上全部逐位相同，最大绝对差均为 0。该结论限定于本轮输入、环境和用途。

## 4. 保留的配置问题与失败尝试

原始失败没有删除，见证据的 `run-01`、`run-02`；最终汇总在 `run-03`。已完成训练不重跑，后续通过引用原训练目录和 checkpoint 哈希复用证据。

1. 第一版测试推理配置漏填 `saver.save_dir/results_dir`，尚未进入推理就失败；补齐的是测试配置。
2. 原版和新版 YOLO 的完整训练 checkpoint 含 NumPy 调度器状态，Torch 2.10 默认 `weights_only` 加载会拒绝。推理测试另导出仅含 `model` 的 tensor-only 文件，逐项核对权重不变。完整训练 checkpoint 仍保留，不关闭框架默认安全加载，不给未知权重放行。
3. YOLO `aligned: true` 与新推理进程默认 UP 全局坐标标志不同，造成恰好 1 像素差。测试入口在两份源代码上显式应用同一 UP 坐标配置；没有改模型或平台算子。这也意味着本轮不是对所有原始 CLI 配置初始化组合的验收。
4. 旧版 Runner 的 evaluate 不恢复 training 标志。为继续同一个未改动的旧版 train 循环，测试包装器在记录旧行为后仅恢复调用前模块标志；新版必须靠自己的状态保护恢复。没有替换前后向或修改任何权重。
5. 初版小子集 manifest 继承了父集合的 indices/image hash 汇总字段；真正的选中元数据、图片字节、输入批次另有哈希审计。原记录未改写，公开工具已修正这些汇总字段，证据索引明确标注该限制。

生产代码本次只按 [Karpathy Guidelines](https://github.com/multica-ai/andrej-karpathy-skills/blob/main/skills/karpathy-guidelines/SKILL.md) 做两处最小整理：无 dtype 限制时不遍历输入张量；有变体时只解析一次函数签名。未改上层算法、默认精度、训练配方或厂商算子数学。新增回归覆盖跳过检查、嵌套 dtype 拒绝、变体检查及证据工具，CPU 共 260 项通过。

## 5. 复现

先准备本仓库既有的冻结 CIFAR-10/PennFudan 数据 manifest 和固定官方 DETR 代码/权重，过程见[历史数据准备](../hardware/CROSS_VENDOR_ACCURACY.md)。下面命令**只启动短程平台回归**，不会运行旧多种子控制器。

同一环境中准备两份源码；`BEFORE` 解包基线，`AFTER` 为本次发布源码。普通安装顺序仍是先平台包，再 UP。NVIDIA 扩展需分别构建，沐曦不能构建或复用 NVIDIA 二进制。

```bash
git archive 0c9a43c7493563f837ad9dd27761cb00bf5fa1ce up configs setup.py requirements.txt | tar -x -C "$BEFORE"

# 仅 NVIDIA。显式使用选中厂商的 Python，不替换现有 Torch wheel。
(cd "$BEFORE" && UP_BUILD_BACKEND=nvidia TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2 "$PYTHON" setup.py build_ext --inplace)
(cd "$AFTER/packages/up-platform" && UP_BUILD_BACKEND=nvidia TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2 "$PYTHON" setup.py build_ext --inplace)

CUDA_VISIBLE_DEVICES="$ALLOCATED_GPU" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
PYTHONHASHSEED=0 DEFAULT_TASKS=cls:det CUBLAS_WORKSPACE_CONFIG=:4096:8 \
PYTORCH_DISABLE_CUDA_CUDNN_TF32=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
"$PYTHON" "$AFTER/tools/validate_platform_equivalence.py" suite \
  --backend nvidia --baseline "$BEFORE" --changed "$AFTER" \
  --data /path/to/frozen/data.json --output /path/to/new-empty-result
```

沐曦在已有厂商容器及应用环境内执行相同命令，将 `--backend` 改成 `metax`，不执行 NVIDIA 构建段。只能使用实际获配 GPU，不自动挑空卡。输出目录必须全新；发现失败时保留原目录，不通过删目录/挑种子制造通过。

补做同权重独立推理，可以直接复用旧版训练目录，不再训练：

```bash
# 沿用上面的可见设备、线程和精度环境变量。
"$PYTHON" "$AFTER/tools/validate_platform_equivalence.py" inference \
  --source "$AFTER" --backend metax --model detr \
  --data /path/to/result/data/data.json \
  --training /path/to/result/baseline/detr/training \
  --output /path/to/new-empty-same-weight-inference
```

具体执行过的驱动版本、短程重复控制、推理补跑与 CPU 差异审计脚本均保存在证据包 `tools/`。阶段之间只修测试入口，平台执行源码未变；最终发布工具另修正 manifest 汇总字段。用各报告的 `driver_sha256` 和归一化源清单区分版本，不把更新后的驱动哈希冒充所有历史进程都使用过。

## 6. 环境与证据

| 平台 | 环境 |
| --- | --- |
| NVIDIA A800-SXM4-80GB | Python 3.12.3，Torch 2.10.0+cu126，TorchVision 0.25.0+cu126 |
| 沐曦 C500 | Python 3.12.11，Torch 2.10.0+metax3.8.1.0，TorchVision 0.25.0+metax3.8.1.0，既有厂商 Docker |

其余依赖、实际精度开关、每步审计、原始失败、恢复与推理记录见[本轮证据索引](validation/hardware_20261022/README.md)。大体积 checkpoint/梯度留在各自任务目录，公开索引记录大小和 SHA256，不把训练权重或数据集放进 Git。

**没有验证：** ImageNet/COCO 全量收敛、跨厂商平均精度等价、AMP、多卡、全体模型与算子、推理引擎、当前结构的完整 ONNX 流程、昇腾/寒武纪真机。历史精度报告和未通过阶段保持原结论，不因本轮短程通过而升级。
