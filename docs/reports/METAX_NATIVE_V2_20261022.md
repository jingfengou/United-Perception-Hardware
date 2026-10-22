# C500 DETR：native 与 v2 的精度、速度和显存

已完成同一张 C500 上的八次完整训练，共 5,760 次更新，全部 seed=131。**热缓存下，v2 每步耗时增加 54.1%，吞吐下降 35.1%；训练峰值已分配显存增加约 4.65 MiB。** 本轮比较完整 v2 策略，不是只测 GEMM 内核，也没有恢复旧多随机种子任务。

## 对照范围

- 同一张物理 C500 GPU2、同一容器、相同 PyTorch/SDK 和同一运行时源码。
- 全部使用 seed=131；每次从相同初始模型和优化器开始，不续训、不选最好 epoch。
- 沿用 PennFudan 固定 120 train / 50 test、320/512px、batch2 分组采样及单图尾批；12 epochs / 720 updates。
- AdamW、backbone 分组学习率、8/11 轮衰减和 0.1 梯度裁剪均复用原配方。
- DETR 使用既有官方核心适配器及其冻结的二维分类损失布局；本轮不改适配器，不冒充原封不动的官方 COCO 默认训练。
- 两边均为 FP32、TF32 off、`cudnn.benchmark=False`、`cudnn.deterministic=True`、`torch.use_deterministic_algorithms(True)` 且 `warn_only=False`。不放宽原 `rtol=1e-4, atol=1e-5, metric_atol=1e-6`。
- 每组固定 native → v2 → v2 → native 顺序，各两次；相同种子的重复不是多随机种子统计。

## 计时与显存口径

使用真实 `BaseRunner.train()`，不另写模型训练循环。增加 CUDA event、内存统计和轻量 loss/LR 记录；没有逐步梯度/权重哈希、张量落盘或每步强制同步。第一批输入哈希只发生在预热阶段。

两边核对数据内容、原配方、初始化模型/优化器/RNG、第一批输入、全部学习率和实际 batch 大小。为了不污染计时，本轮不重复前轮每步输入/RNG/梯度大张量审计；前轮正确性控制见 [GEMM 修复报告](METAX_GEMM_FIX_20261022.md)。

- **冷缓存完整训练：** 每次独立 Triton 缓存，记录首次实际运行成本。初始化、模型加载和初始 eval 另列；训练时间剔除每轮 eval 和 checkpoint 保存。
- **热缓存训练：** 复用一次完整 v2 运行生成的缓存，另起新进程、新模型，重新执行相同 720 步；核对缓存文件前后清单。
- **预热区间：** 事先固定第 1 轮 60 步，速度主表使用后 660 步。冷缓存中后续新形状仍可能首次编译，不能称为无编译的稳态。
- **核心区间：** CUDA event 从 forward 前到 optimizer update 后，排除数据读取/迁移，不把其耗时冒充纯 GPU 内核时间，区间仍可能包含 CPU matcher 和发射间隙。
- **训练峰值显存：** PyTorch `max_memory_allocated` 与 `max_memory_reserved`，每次 eval/save 前汇总、返回后重置峰值，排除评估自身的临时分配。reserved 包含正常保留下来的分配器缓存；不等于驱动层整卡显存，也不含所有 SDK 库外分配。

固定顺序不等于串行计算。v2 仍使用 GPU FP32；完整策略还包含 v1 的卷积权重梯度替代，因此本轮开销不能全部归因于 GEMM。

## 结果

### 冷缓存完整预算

| 模式 | 最终 bbox.AP，两次，0–1 | 720 步纯训练秒数，两次 | 第 1 轮平均秒数 | 后 660 步平均 ms/update |
| --- | --- | --- | ---: | ---: |
| native | 0.7721016191 / 0.7689055748 | 85.50 / 86.38 | 7.27 | 119.19 |
| v2 | 0.7706035845 / 0.7706035845 | 178.29 / 178.00 | 44.62 | 202.31 |

平均完整纯训练时间 85.94 → 178.15 秒，v2 为 native 的 2.073 倍。后 660 步也仍包含晚出现形状的首次编译，不能将其 1.697 倍耗时直接称为热缓存运行成本。

### 热缓存与重复性

完整 v2 缓存的 874 个文件在该组执行前后内容一致，没有新增编译缓存。以下速度取两次平均，不挑更快的一次。

| 模式 | 最终 bbox.AP，两次，0–1 | 720 步纯训练秒数，两次 | 后 660 步 ms/update，平均 | 实际 images/s，平均 |
| --- | --- | --- | ---: | ---: |
| native | 0.7698512063 / 0.7841633986 | 84.78 / 85.51 | 117.97 | 16.81 |
| v2 | 0.7706035845 / 0.7706035845 | 133.84 / 131.04 | 181.78 | 10.91 |

热缓存完整纯训练平均 85.15 → 132.44 秒，增加 55.5%。包含正常每轮 eval 和 checkpoint 保存的训练循环平均 107.51 → 159.58 秒；初始化/加载/初始 eval 不在这两个循环计时内，逐次记录见机器可读结果。

| 训练峰值，所有重复相同 | native | v2 | 增加 |
| --- | ---: | ---: | ---: |
| PyTorch 已分配显存 | 1088.77 MiB | 1093.42 MiB | 4.65 MiB，0.43% |
| PyTorch 保留显存 | 1170 MiB | 1190 MiB | 20 MiB，1.71% |

四份 v2 的逐步 loss/LR/batch 大小记录、最终模型和优化器均逐位一致；其 DETR 核心权重摘要仍为前轮已验证的 `371c5bbb491a6de7cc4258f8f5903b44423a18ef6b6dd5cab04c9a82a5c72074`。这说明冷/热缓存和基准计时没有改变本次固定策略的结果。

native 四次 AP 范围为 0.7689055748–0.7841633986，最终权重和 loss 存在重复性波动。四次平均 AP 为 0.7737554497，v2 为 0.7706035845，观察均值差为 **-0.3152 个百分点**；冷缓存组均值差为 +0.0100 点，热缓存组为 -0.6404 点。没有挑选更低的 native 对照宣布 v2 精度提升，也不能据此断言真实期望精度下降：只有一个种子、小型留出集和有限重复，**本轮不记作 native/v2 精度等价通过，不放宽原容差**。

结论是：本配置下，v2 的主要代价是速度，分配器显存增量较小，且所测训练可重复；不证明所有模型的精度无损或同等开销。默认 native 和显式 v2 的选择方式不变。

## 复现

先使用沐曦厂商容器及匹配的依赖，不用 NVIDIA wheel 覆盖厂商 PyTorch。`SOURCE` 是 `$RUN/source` 下的冻结源码，`DATA` 是既有数据 manifest；以下结果目录均须不存在。

```bash
export USER="${USER:-up-validation}" LOGNAME="${LOGNAME:-up-validation}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0
export DEFAULT_TASKS=cls:det CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_DISABLE_CUDA_CUDNN_TF32=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
DRIVER="$SOURCE/benchmark/hardware/benchmark_metax_detr.py"

# 单卡串行 ABBA，四次完整固定预算，首次编译缓存相互独立。
python "$DRIVER" suite --source "$SOURCE" --data "$DATA" \
  --device 2 --output "$RUN/results"

# 复用完整 v2 缓存；模型、初始化、种子和训练预算不变。
python "$DRIVER" suite --source "$SOURCE" --data "$DATA" \
  --device 2 --cache-from "$RUN/results/v2-0-triton-cache" --output "$RUN/warm-results"

# 两组完成后仅在 CPU 上核对 checkpoint，并从最终预测复算 COCO 指标。
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$SOURCE/benchmark/hardware" \
  python "$SOURCE/benchmark/hardware/collect_metax_detr_benchmark.py" \
  --root "$RUN" --output "$RUN/../metax-native-v2-evidence.tar.gz"
```

实际原始驱动与执行控制器随证据归档保存。后加的 `--cache-from` 是复现入口，不更改计时器、训练函数或运行时；发布时逐 AST 核对这些执行部分与实测快照一致。

## 环境与证据

运行时来自 `0b7f0c7ba5da96741e7661ed8dac2212c3694693`，归一化 manifest 为 `18e4171327dd0de6da77db99ebe0085c09150c3a008b09b014282bf55fe997c9`（730 个文件），原数据 manifest SHA256 为 `d62a893d9ea3733b55e972e0609a5243022a26b3f5fa67076acf7e88a08a6830`。

Python 3.12.11；Torch `2.10.0+metax3.8.1.0`、TorchVision `0.25.0+metax3.8.1.0`、Triton `3.6.0+metax3.8.1.0`；NumPy 1.26.4、SciPy 1.16.0、pycocotools 2.0.10；Driver 3.6.11、MACA 3.8.1.3。未替换环境或改动平台数学实现。

[机器可读结果](metax-native-v2-20261022.json)保存全部模式/重复、计时、环境、CPU 审计与源码核对。[原始证据 Release](https://github.com/jingfengou/United-Perception-Hardware/releases/tag/metax-native-v2-20261022)保存 82 个已逐成员校验的归档成员，包括实际执行驱动、完整源码 tar、控制器、逐步 loss/LR/计时、最终预测及原始日志。24 条 checkpoint 路径的字节校验索引随附，权重保留在服务器，不放入 Git。

八份最终模型/优化器均经 CPU 检查为有限值，保存的预测重新计算全部 12 项 COCO 指标，与训练报告最大差为 0。CPU 指标复算不是 CPU 模型代算。所有本任务进程已正常退出，GPU 已释放，旧结果和容器未删除。

| 下载文件 | 字节 | SHA256 |
| --- | ---: | --- |
| `metax-native-v2-evidence.tar.gz` | 3642246 | `7b345334a87368724e82cfeb11853f04bcf3478b056e24e95bc2a86cefd12d98` |
| `metax-native-v2-evidence.tar.gz.index.json` | 16723 | `e2d178ff55b2cd085343658b7f584b48f6daa19238c0571dd13f4621a173317e` |

维护验证：358 tests + 3 subtests 通过，两条既有 ONNX 弃用 warning；31 个运行时成员 / 73 项操作契约无漂移，硬件边界违规 0；665 个受保护源码/配置及 6 个历史协议夹具不变。当前提交的运行时与实测 730 文件清单一致，计时器和训练函数与实际驱动逐 AST 一致。后增缓存复现参数和汇总检查没有更改测量执行部分。

这是小型真实数据、单种子、FP32、单卡的成本与重复性对照，不是全量 COCO 收敛、跨厂商精度等价、AMP/多卡或所有形状的性能证明。native 和 v2 的最终 AP 差异不自动等于精度提升/退化；不以两次重复代替多种子统计，历史失败继续保留。
