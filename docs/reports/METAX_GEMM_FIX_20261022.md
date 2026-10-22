# C500 DETR：平台层 GEMM 修复与完整预算对照

已完成平台层修复与四份完整预算控制，共 2,880 次更新。**在同一 v2 策略下，整理前后、同种子重复及独立官方核心的逐步 loss、梯度、权重和最终 AP 一致。** 这不是旧 v1 / v2 或跨厂商精度等价通过；原失败继续保留。

## 修复边界

[前轮归因](DETR_ORIGIN_20261022.md)已用实际固定操作数确认：所测 C500 厂商 PyTorch 的 `aten.mm.default` 会在相同输入下产生末位波动，UP 不是其发生的必要条件。单次结果在原容差内，但优化迭代可放大差异。

本轮仅在 `up_platform/backends/metax/` 新增显式 `metax_deterministic_v2`：

- 沿用原 v1 的卷积权重梯度实现，原文件和 v1 语义不变。
- 二维 GPU FP32 `mm` 使用厂商 Triton 内核，每个输出 tile 只有一个写入者，按固定 K 分块顺序累加；`input_precision="ieee"`，输入、累加和输出均为 FP32。没有 split-K、atomic 累加或 autotune。
- 不按模型名、梯度名或故障案例的特定形状分支。通过 PyTorch dispatch 处理该完整 `mm` 调用，原 autograd 仍负责前向/反向关系；没有重写 DETR、优化器、损失或学习率。
- 不启用 portable_v1，不更换随机数，不使用 CPU/FP64 模型代算。CPU FP64 只用于小型算子审计参考。
- 新内核只在显式选择策略并首次执行 GPU mm 时加载。默认 native、NVIDIA 后端及旧策略不变。

实现见[厂商路由](../../packages/up-platform/src/up_platform/backends/metax/determinism_v2.py)和[固定 FP32 内核](../../packages/up-platform/src/up_platform/backends/metax/fixed_gemm_kernel.py)。IEEE 选项的接口定义见 [Triton dot 文档](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)；具体 C500 可重复性由本轮真机结果确认，不从这个参数单独推断。

核验范围为密集、同设备二维 FP32 `aten.mm.default`，包含转置/切片/零步幅布局及空维。GPU 非 FP32、稀疏 mm 和 out 变体不在该覆盖契约内；CPU 输入的原有 CPU 计算不改变，也不是把 GPU 操作回退到 CPU。BMM、addmm 和其他厂商操作没有因此自动变成确定性实现，不能声称已解决所有形状、AMP、多卡或导出问题。

内核采用固定 `32 x 32` 输出 tile、K 分块 32、4 warps，优先保证所测范围可重复；不是性能优化版本。实际 GEMM 算例已在两个独立进程各重复 100 次，均逐位一致，距 CPU FP64 参考最大绝对差 `1.9073486328125e-6`，原容差内。24 组 shape/layout 的前向和两侧梯度测试均通过，最大绝对差 `7.748603820800781e-7`。这些是算子结果，不替代完整训练验收。

## 对照设计

沿用 seed=131、PennFudan 120 train / 50 test、320/512px、原 batch2 分组采样及单图尾批、12 epochs / 720 updates、AdamW 分组学习率与 0.1 梯度裁剪。仍为 `rtol=1e-4, atol=1e-5, metric_atol=1e-6`，不选择最佳 epoch。

1. 整理前 `9bc45d2` 叠加同一份新厂商适配，完成 720 步。
2. 当前简化后的平台叠加新适配，完成 720 步，再以同一 seed 独立重复一次。
3. 不调用 UP 编排的官方 DETR 核心，从同一初始化开始重放相同输入/RNG/LR，启用同一局部适配，完成 720 步。

两版本都应用修复，是为了检验平台整理没有改变模型结果；**不能把随机的旧 v1 轨迹与新固定归约强行要求逐位相同**。旧失败仍成立，独立官方控制仍保留冻结的二维分类损失布局，不冒充原封不动的官方 COCO 默认训练。未经布局适配的官方 NLL 确定性拒绝没有在本轮通过修改模型绕过。

每次 UP 训练保存逐步原始/裁剪后梯度与权重摘要，前 70 步保存完整梯度。审计初始评估、12 次 epoch eval、最终评估和新模型/优化器恢复后的评估；检查状态、参数身份、模块模式和 training 用途。独立推理另起进程选择 inference，用最终权重检查全部 50 张留出图片。

## 完整结果

以下四次均使用同一张物理 C500 2 号卡、seed=131 和同一初始权重。不是四个随机种子，不选择最佳 epoch。

| 控制 | 更新次数 | 最终 bbox.AP（0–1） | 与当前版逐步对照 |
| --- | ---: | ---: | --- |
| 整理前 UP + v2 | 720 | 0.7706035844596513 | loss、原始/裁剪梯度、权重逐位一致 |
| 当前 UP + v2 | 720 | 0.7706035844596513 | 参照 |
| 当前 UP + v2，同种子重复 | 720 | 0.7706035844596513 | 同上 |
| 独立官方核心 + 相同局部适配 | 720 | 0.7706035844596513 | 同上；训练进程不导入 UP |

四份最终模型和优化器状态均有限、逐位一致，最终模型摘要均为 `371c5bbb491a6de7cc4258f8f5903b44423a18ef6b6dd5cab04c9a82a5c72074`。最大 loss / 最终权重绝对差为 0；没有用容差掩盖残差。

官方核心训练进程禁止导入 UP；其最终权重的 AP 使用同一独立 UP 推理入口复算，以保持后处理与评估协议一致，不能将这段 AP 流水线也称为“未使用 UP”。

- 三次 UP 各完成 15 次评估，共 45 次；预测摘要及全部 12 项 AP/AR 一致。训练内 eval 始终保持 training 用途，模块状态、权重、梯度、优化器及 scaler 未被评估更新；新模型和优化器恢复后的结果一致。
- 四份最终权重分别另起 inference 进程，实际调用 `BaseInference.predict()`，再对全部 50 张留出图片做相同批次比较；各 25 个预测张量与训练 eval 逐位一致，最大差为 0，没有构造优化器/scaler。全部预测摘要均为 `4ffda72c78df0139a3cc613e67870d05c6c8d8befef74e6b7a6d868724e5f995`。
- 原来首次分叉的第 58 步，两次同状态追踪均完成 6,185 条记录，未出现不同输出，最终梯度摘要相同。
- 两进程各 100 次真实 GEMM 重放及 24 组真实步幅/空维的前向、两侧梯度均通过；零步幅确实保留，例如输入步幅 `[0,1]` / `[256,0]`，不是 `.to()` 后被连续化的替代测试。

每次 UP 记录实际执行 148,695 次 mm 和 30,960 次卷积权重梯度替换；独立推理也实际执行了 mm。完整训练时间（含逐步梯度/权重摘要及 epoch eval）约 863–896 秒，**不是 GPU 吞吐基准**。

后续单独测量的同卡 native/v2 训练速度、峰值显存与最终 AP 见[成本对照](METAX_NATIVE_V2_20261022.md)。该基准区分首次编译与热缓存，不拿本节重审计的耗时计算降速比例。

### 与旧 v1 的区别

已核对与前轮归因阶段的 v1：初始权重、全部 720 个实际输入及学习率完全相同。但固定归约会形成一条新的可重复 FP32 轨迹，不保证复刻旧的不稳定轨迹。

该次旧 v1 留存日志的最终 AP 为 `0.767467278055288`，本次 v2 为 `0.7706035844596513`，差 `0.0031363064043632427`，**不在原 `metric_atol=1e-6` 内，不记作 v1/v2 精度等价通过**。更早[四模型回归](CURRENT_VALIDATION_20261022.md)的旧 v1 数字同样保留，不挑选旧结果宣布精度提升。本轮证明的是修复后的可重复性和同策略框架一致性，不是更高精度或 NVIDIA 等价。

## 实际环境

| 项目 | 版本 |
| --- | --- |
| Python | 3.12.11 |
| Torch / TorchVision | `2.10.0+metax3.8.1.0` / `0.25.0+metax3.8.1.0` |
| 厂商 Triton | `3.6.0+metax3.8.1.0` |
| NumPy / SciPy / pycocotools | 1.26.4 / 1.16.0 / 2.0.10 |
| Driver / MACA | 3.6.11 / 3.8.1.3 |

容器镜像为 `cr.metax-tech.com/public-library/maca-pytorch@sha256:1a10a6cc56741f5e84da5dd6f0aa9dc4cfd420d66188e43983c9a3efd9d85ab0`，本轮未替换厂商 PyTorch、SDK 或依赖。`torch.version.cuda` 是厂商兼容字段，不能据此安装 NVIDIA wheel。

## 使用

在模型 YAML 已有的 runtime 段增加数值策略，不改变算法配方：

```yaml
runtime:
  device: cuda
  fp16: false
  accelerator:
    backend: metax
    numerics: metax_deterministic_v2
```

先准备厂商匹配的 PyTorch/TorchVision/Triton，不要安装 NVIDIA wheel 覆盖厂商环境。独立推理使用同样的 accelerator 配置，入口仍明确选择 inference；训练中的 eval 不切换后端。

## 复现

执行源、结果和失败控制分别保留在新实验目录，不覆盖前轮 `detr-origin-20261022`。本轮固定的运行时归一化 manifest 为 `18e4171327dd0de6da77db99ebe0085c09150c3a008b09b014282bf55fe997c9`；整理前叠加同一修复为 `c3693d9380970dcd4c9b124a7e7ba7f420ec5be903c3de43f35f1ea7cbe7a31b`。原数据 manifest SHA256 为 `d62a893d9ea3733b55e972e0609a5243022a26b3f5fa67076acf7e88a08a6830`。

保持前轮环境和数据。`SOURCE` 为本轮冻结源码，`DATA` 为原始 manifest，`RUN` 为全新的输出目录；所有子目录必须不存在。

```bash
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0 DEFAULT_TASKS=cls:det
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_DISABLE_CUDA_CUDNN_TF32=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export USER="${USER:-up-validation}" LOGNAME="${LOGNAME:-up-validation}"
DRIVER=benchmark/hardware/diagnose_detr_origin.py

# CASE 使用前轮 Release 的真实矩阵案例，先核对下面的 SHA256。
CASE=/path/to/first-different-operation.pt
for REPEAT in 0 1; do
  python benchmark/hardware/validate_metax_gemm.py --source "$SOURCE" \
    --case "$CASE" --repeats 100 --output "$RUN/primitive-layout-$REPEAT"
done

# BASELINE 是 9bc45d2 的独立源码目录；仅叠加相同厂商适配，勿改其 UP/config。
for FILE in device.py determinism_v2.py fixed_gemm.py fixed_gemm_kernel.py; do
  cp "$SOURCE/packages/up-platform/src/up_platform/backends/metax/$FILE" \
     "$BASELINE/packages/up-platform/src/up_platform/backends/metax/$FILE"
done
python "$DRIVER" capture --source "$BASELINE" --data "$DATA" \
  --steps 720 --save-gradients 70 --profile metax_deterministic_v2 \
  --audit-evaluation --output "$RUN/baseline-full"

# 实际 tensor/RNG/LR tape；原 720 步完整预算与每 epoch 验证不变。
python "$DRIVER" capture --source "$SOURCE" --data "$DATA" \
  --steps 720 --save-gradients 70 --profile metax_deterministic_v2 \
  --audit-evaluation --output "$RUN/changed-full-0"
python "$DRIVER" capture --source "$SOURCE" --data "$DATA" \
  --steps 720 --save-gradients 70 --profile metax_deterministic_v2 \
  --audit-evaluation --output "$RUN/changed-full-1"

# 无 UP 编排，显式保留同一局部平台 mode 和冻结 criterion 布局。
python "$DRIVER" standalone --source "$SOURCE" --tape "$RUN/changed-full-0" \
  --steps 720 --profile metax_deterministic_v2 --criterion matched --output "$RUN/official-full"

# 三份 UP 最终权重分别独立 inference；再验证独立官方核心最终权重。
for NAME in baseline-full changed-full-0 changed-full-1; do
  python benchmark/hardware/validate_metax_detr_fix.py --source "$SOURCE" \
    --training "$RUN/$NAME" --output "$RUN/inference/$NAME"
done
python benchmark/hardware/validate_metax_detr_fix.py --source "$SOURCE" \
  --training "$RUN/changed-full-0" --official "$RUN/official-full" \
  --output "$RUN/inference/official-full"
```

真实案例见[前轮 Release](https://github.com/jingfengou/United-Perception-Hardware/releases/download/detr-origin-20261022/first-different-operation.pt)，SHA256 为 `0bcbe92b0db33291da6d7d1b232e53e53e2ee9613f77c8ca1e0dc3788b882ec1`。只加载自己生成或已校验的可信 tensor tape / checkpoint。USER/LOGNAME 用于解决无 passwd 条目的容器 UID 触发 getpass 错误，不改变数学计算。

完整训练的严格比较可复用 `collect_metax_gemm.compare_reports()`，它要求 720 步、原容差、同一 v2 策略和初始权重/输入，不会把缺失阶段当通过。`collect_metax_gemm.py` 使用冻结的 source/data/协议元数据校验模型、优化器、全部评估及推理，并打包原始记录；本轮实际控制器、逐步报告和工具快照随 Release 保留。

## 证据与限制

[机器可读结果](metax-gemm-fix-20261022.json)记录全部控制、实际环境、源码核对及校验值；[固定 Release](https://github.com/jingfengou/United-Perception-Hardware/releases/tag/metax-gemm-fix-20261022)保存原始文本、执行工具和大张量逐文件索引。大权重/数据保留在验证目录，不加入 Git；原始 v1 失败、探索性布局测试和控制过程均保留，不覆盖后改记成功。

归档的 **82 个成员**已逐成员校验。索引另列 **2,466 个张量/权重/源码快照二进制文件，共 56,694,338,524 字节**，保留于服务器实验根目录 `/workspace/up-run/metax-gemm-fix-20261022`，没有将几十 GB 权重加入仓库。

| 下载文件 | 字节 | SHA256 |
| --- | ---: | --- |
| `metax-gemm-fix-evidence.tar.gz` | 32538673 | `bbbc3f33f633a2bb3509f0a52ad80dbbde4e2a0aad0c251ed1e628be9a3b404c` |
| `metax-gemm-fix-evidence.tar.gz.index.json` | 388022 | `6d3dc3785f8f50676786ab1c0baa2fddab21d18257ff763911f8f4b64d26029b` |

前期布局探测使用 `.to()`，可能连续化切片/expand，故只记探索材料；最终门槛使用显式恢复并断言原步幅的新探测。训练/eval 审计函数的 AST 与完整训练执行快照一致，后增独立推理 CLI 不改变它们；提交的运行时与完整训练快照按 CRLF→LF 归一化逐文件核对，不忽略其他内容。

本地 **346 tests + 3 subtests 通过**，两条既有 ONNX 弃用 warning；31 运行时成员 / 73 操作契约无漂移，硬件边界违规 0，665 个受保护源码/配置和 6 个历史夹具未变。平台和 UP 公共 wheel 在无 Torch 的本地 Python 中构建通过，平台安装后可在无 UP/Torch/Triton 环境导入；这不是 GPU 精度的替代验收。

本轮仍不是旧多随机种子复测、NVIDIA/沐曦跨厂商统计等价、全量 COCO 收敛、AMP/DDP/导出或全部 shape/dtype 的证明。默认 native 的历史未通过项不升级，厂商原生 BLAS 本身未被修改；本仓库在明确选择的范围内绕开其波动。
