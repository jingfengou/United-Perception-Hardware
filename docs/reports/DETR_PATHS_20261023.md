# C500 DETR：官方核心、UP native 与 UP v2

完成 18 次独立进程测量及 CPU 状态审计。**本测集的 UP native 模型执行路径额外耗时较小：推理均值 +0.31%，训练 +2.15%；v2 相对 UP native 分别增加 50.76% / 51.04%。** 这不是整套 Runner 或端到端数据流水线开销。

所测 native 前向稳定，但原始梯度不稳定；v2 可重复。**v2 与 native 的推理存在少量原容差外元素，严格数值等价未通过，不宣称精度无损。** 未恢复旧多随机种子任务。

## 对照对象

同一张物理 C500 GPU2（PCI `0000:0e:00.0`）、同一厂商容器、seed=131、相同初始模型和优化器、相同真实图片张量、mask、目标框与逐批 RNG。三条路径分别使用独立进程；推理和训练各执行三轮，顺序为 ABC、BCA、CAB，共 18 次。

| 路径 | 执行内容 |
| --- | --- |
| 官方核心 native | 固定版本官方 DETR 核心，普通厂商 PyTorch；进程禁止导入 `up` 和 `up_platform` |
| UP native | 相同核心，通过现有 `ModelHelper` 与 `DetrForUP` 适配器；平台用途显式选择 training 或 inference |
| UP v2 | 与 UP native 相同，仅显式选择 `metax_deterministic_v2` |

**训练主对照保留三者相同的既有二维分类损失布局，不是原封不动的官方默认损失。** 另行执行的原官方损失控制，在严格确定性策略下被 `nll_loss2d_forward_out_cuda_template` 拒绝；失败日志保留，不改成 warn-only 绕过，也不记作通过。损失布局与历史验证相同，本轮没有修改模型算法、平台数学实现或容差。

## 测量范围

- 模型为固定官方 DETR R50 核心和既有单类 PennFudan 配方，不是 HuggingFace COCO 80 类推理基准。
- FP32、TF32 off、`cudnn.benchmark=False`、`cudnn.deterministic=True`、严格 `torch.use_deterministic_algorithms(True)`。比较沿用 `rtol=1e-4, atol=1e-5`。
- 训练输入来自既有 120 train / 50 test 划分的真实 UP 数据管线，冻结 60 个训练批次、119 图次，含一个单图尾批和 10 种 NCHW 形状。推理冻结全部 50 张留出图片的 25 批次，含 15 种形状。
- 每个进程先覆盖全部对应输入进行预热，再恢复同一初始权重与优化器。训练计时为 120 次更新，重放两遍训练输入，共 238 图次；推理计时为 25 批次重放六遍，共 300 图次。
- 三者共用 AdamW、分组参数顺序、学习率、0.1 梯度裁剪和测试循环。120 步处在原配方第一次学习率衰减之前，不冒充历史 720 步完整预算。
- 输入提前常驻 GPU，RNG 恢复在计时之外；使用逐批同步的主机时钟和 CUDA event 区间。训练包含损失、原有 CPU Hungarian matcher、反向和更新；推理包含对应框解码与旧格式转换。event 区间也可能包含主机发射间隙，不是纯内核时间。
- UP 计时包含 `ModelHelper` 和原适配器的 mask/target 构造、格式转换等额外工作；**不包含完整 `BaseRunner`、数据加载、传输、日志、eval 和 checkpoint 流水线**。表中吞吐由相同计时区间换算，不是异步流水线端到端吞吐。
- 训练与推理分别初始化平台用途；不根据 `.eval()` 或 `no_grad()` 猜测。共同使用 `no_grad` 推理，不单独给官方基线启用其他优化路径。

## 性能与显存

取三次平均，不挑最快的一次；括号为三次运行均值的最小值至最大值，不是置信区间。

| 路径 | 推理 ms/批，batch2 | 训练 ms/更新 | 推理图次/s | 训练图次/s |
| --- | ---: | ---: | ---: | ---: |
| 官方核心 native | 26.5865（26.4806 至 26.6429） | 103.7435（103.4050 至 104.0779） | 75.23 | 19.12 |
| UP native | 26.6696（26.3956 至 26.8126） | 105.9762（105.2839 至 106.7313） | 75.00 | 18.72 |
| UP v2 | 40.2077（39.7398 至 40.9718） | 160.0650（157.8905 至 163.4707） | 49.75 | 12.39 |

UP native 相对官方核心，推理均值 +0.31%、训练 +2.15%；推理的重复区间重叠，不把三次观察当作稳定的精确开销定律。差值还包含 UP 适配器的输入/输出整理，不能全归因于平台路由。

v2 相对 UP native，推理耗时 +50.76%、训练 +51.04%，对应吞吐分别下降 33.66% / 33.78%。该代价包含 dispatch、固定 GEMM 和训练时沿用的卷积权重梯度实现，不是单独 GEMM 内核的降速。所有计时区间的 Triton 缓存清单均未变化；首次编译发生在单列的预热区间，不混入主表。

| allocator 峰值，MiB，三次相同 | 推理 allocated | 推理 reserved | 训练 allocated | 训练 reserved |
| --- | ---: | ---: | ---: | ---: |
| 官方核心 native | 526.88 | 556 | 1490.91 | 1526 |
| UP native | 527.23 | 556 | 1491.23 | 1526 |
| UP v2 | 527.23 | 556 | 1490.43 | 1554 |

三者预热后的 allocated 基线分别为推理 433.38 MiB、训练 694.11 MiB。UP native 比官方核心的 allocated 峰值多约 0.34 / 0.31 MiB。v2 在本输入集的训练 allocated 峰值反而少约 0.80 MiB，但 reserved 多 28 MiB；这是不同分配/复用路径的观察结果，不能推广成 v2 没有显存代价。

显存为 PyTorch allocator 的 `max_memory_allocated` / `max_memory_reserved`，**包含三者相同的 GPU 输入仓库**；报告同时保存预热后基线。它不是驱动层整卡占用，不包含所有 SDK 库外分配，也不应直接与此前逐批数据加载的完整训练峰值相比。

## 数值与重复性

固定权重数值审计不在计时区间内：推理选第 0、12、24 批，各重复 10 次；训练选第 0 批和第 57 批，后者恰为单图尾批，分别重复 10 / 20 次。训练的三个审计槽位是 `[0, 57, 57]`，不是三个独立输入形状。每次恢复同一 RNG，不执行优化器更新，检查原始预测、loss 和全部可训练参数的原始梯度。

另外保留 120 次真实更新的完整 loss 列表、最终模型及优化器，检查同路径独立进程重现性和跨路径差异。全部数值、原始张量文件 SHA 与失败项保留；数值比较不等于最终精度或收敛验收。

| 控制 | 结果 |
| --- | --- |
| 各路径固定权重推理重复 | 各路径自身及其独立进程之间，所测原始预测和解码结果逐位一致 |
| 官方核心 native 对 UP native 推理 | 三批数值探针、三次独立进程均逐位一致 |
| native 固定权重训练重复 | 前向预测和 loss 稳定；官方核心、UP native 的梯度分别在全部 81/81 次组内比较中不逐位相同 |
| native 梯度波动 | 官方核心最大绝对差 `2.4795532e-5`，UP native 为 `2.3365021e-5`；分别有 4/81、5/81 次比较出现原容差外元素，每次最多 1 个 |
| v2 固定权重训练重复 | 原始预测、loss 和原始梯度均逐位一致 |
| 120 更新独立进程重复 | 两种 native 各得到三份不同的最终模型/优化器；v2 三份逐步 loss、最终模型和优化器逐位一致 |
| native 对 v2 严格数值等价 | 未通过；固定权重推理、部分固定权重梯度及更新轨迹均保留差异 |

81 次组内比较来自三个进程、每进程三个审计槽位、每槽九次与首次观察比较；尾批槽位重复，不是 81 个独立种子或输入。v2 最终模型摘要为 `31e3e33a3c92d400ca1d4c55c55ddef4206270a9aa8c81f0250059bd1bd26f43`。

v2 相对 native 的三批推理探针中，原始预测最大绝对差 `2.3603439e-4`，共 2 个元素超容差；解码预测最大绝对差 `0.0070190`，另有 1 个元素超容差。最大差与超差计数不一定来自同一元素：实际超差的框 `x1` 为 `-0.0876331329` 对 `-0.0876140594`，差 `1.9073486e-5`，略高于其原有相对/绝对阈值，仍严格记作未通过。

训练第一份官方核心 native 对 UP native，固定权重前向/loss 相同、梯度最大差 `1.5258789e-5`（该份探针容差内）；120 更新后最大 loss 差为 `1.51179`，最终权重最大差 `0.00623695`，严格比较未通过。官方核心自身的重复也存在梯度和轨迹差异，因此不能把两条 native 随机轨迹之间的差异直接判为 UP 回归；本轮也不声称所有残差只来自 GEMM。

这回答了训练/推理的区别：**本 DETR 测集观察到的是 native 反向梯度非确定性，不是重复推理随机漂移；v2 推理的差异是不同数值策略之间的稳定差异。** 这不保证任意推理形状均无非确定性。v2 目前覆盖前向 mm，所以即使所测前向原本稳定，推理也会付出代价。此轮只测量，不临时更改策略或模型来消除失败。

## 复现

使用本轮匹配的厂商环境和已有冻结数据，不用 NVIDIA wheel 覆盖厂商 PyTorch。数据准备和哈希核验见[既有完整预算报告](CURRENT_VALIDATION_20261022.md#1-冻结源码和数据)。新结果目录必须不存在，不覆盖旧任务。

```bash
SOURCE=/path/to/United-Perception-Hardware
RUN=/path/to/new-detr-path-benchmark
DATA=/path/to/frozen/data.json
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONHASHSEED=0 DEFAULT_TASKS=cls:det
export USER="${USER:-up-validation}" LOGNAME="${LOGNAME:-up-validation}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_DISABLE_CUDA_CUDNN_TF32=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

python "$SOURCE/benchmark/hardware/benchmark_detr_paths.py" prepare \
  --source "$SOURCE" --data "$DATA" --output "$RUN/fixtures"
python "$SOURCE/benchmark/hardware/benchmark_detr_paths.py" suite \
  --source "$SOURCE" --fixtures "$RUN/fixtures" \
  --output "$RUN/results" --device 2

# 单独保留未经损失布局适配的严格策略控制，预期可能非零退出。
python "$SOURCE/benchmark/hardware/benchmark_detr_paths.py" run \
  --source "$SOURCE" --fixtures "$RUN/fixtures" \
  --phase training --criterion official --output "$RUN/official-loss-control"
```

每次运行输出计时分布、峰值/基线显存、完整 loss、固定状态重复比较、完整首份审计张量与最终状态。数值审计不包含在速度统计内。简单汇总可在 CPU 环境调用 `collect_detr_paths.validate_reports()` / `performance()`；完整证据审计入口 `collect_detr_paths.py --root ... --output ...` 还要求实际编排回执和源码归档，随本轮 Release 一同提供。

实际编排器 `launch.py` 在独立目录先验证 GPU 空闲、环境版本、源码和数据，再启动一次 suite；CPU 审计在其正常完成后执行。复刻完整归档时，将 Release 中的源码包、编排器放入新的独立目录，修改编排器的 `ROOT` 为该目录，并核实 `DATA`；保留原哈希约束，不覆盖原结果。只加载可信的本任务张量和 checkpoint。

## 环境与证据

Python 3.12.11，Torch `2.10.0+metax3.8.1.0`，TorchVision `0.25.0+metax3.8.1.0`，Triton `3.6.0+metax3.8.1.0`，NumPy 1.26.4，SciPy 1.16.0，pycocotools 2.0.10，Driver 3.6.11，MACA 3.8.1.3。

运行时与 `b296507` 相同；730 文件的归一化 manifest 为 `18e4171327dd0de6da77db99ebe0085c09150c3a008b09b014282bf55fe997c9`，数据 manifest SHA256 为 `d62a893d9ea3733b55e972e0609a5243022a26b3f5fa67076acf7e88a08a6830`。本轮只增加验证工具、测试和报告，没有修改模型、配置或平台实现。

[机器可读结果](detr-paths-20261023.json)保存三次运行的计时分布、显存、重复性、全部跨路径比较和明确的失败标志。[原始证据 Release](https://github.com/jingfengou/United-Perception-Hardware/releases/tag/metax-detr-paths-20261023)保存源码 tar、实际驱动、编排回执、日志、完整 loss 列表、CPU 审计和索引，共 47 个成员，下载后已逐成员核验。72 条完整状态/审计张量路径的逻辑总大小为 10,433,327,046 字节，逐文件 SHA 已校验，文件保留在服务器，不包含在 Release 中。

| 文件 | 字节 | SHA256 |
| --- | ---: | --- |
| `detr-paths-evidence.tar.gz` | 2112099 | `fba7ef571431183f3c412a93071eca8cb270a0e1205927eedeef7d45100f295f` |
| `detr-paths-evidence.index.json` | 19245 | `2bee8a0f64aef5d1fe319f937f90e2fb1172ab5bc856708c4509731e88e28ae0` |

推理/训练分布只保存计数、均值、中位数、p95 和总量，没有保存逐调用计时样本；不声称可以从归档重新计算任意计时分位数。梯度/权重的大张量审计在计时外执行，不使用带审计的总运行时间计算降速。未新增 CPU 模型代算；原有 CPU matcher 三者一致。

CPU 维护测试 380 tests + 3 subtests 通过，两条既有 ONNX 弃用 warning；31 个运行时成员 / 73 项操作契约无漂移，硬件边界违规 0，665 个受保护源码/配置及 6 个历史协议夹具保持不变。GitHub 发布和 CI 状态以实际提交页面为准。

GPU 与 CPU 审计进程均正常退出；容器只剩原有 `sleep infinity`，`mx-smi` 确认没有 GPU 进程。本轮数据、权重、缓存和历史任务未删除。

本轮不恢复旧多随机种子任务，不证明全量 COCO 收敛、最终精度等价、AMP、多卡、所有输入形状或其他 SDK 版本。所测前向稳定，也不能推广为所有沐曦推理算子均确定。
