# 当前实现的真机与完整预算回归

> 完整训练已执行，结果包括未通过项。验证运行时冻结于 `f78e27a`；本轮只修改验证工具与文档，不改变平台数学实现、模型算法、预算或容差。

**结论：本轮受控 FP32 配置下，NVIDIA 四模型未观测到版本改动引入的精度或训练轨迹差异。C500 仅显式局部策略下的三模型完整预算通过，DETR 仍未通过，不能宣称所有厂商、模型或默认 native 均不影响精度。**

## 比较对象

- NVIDIA：原版 UP `52aaeb9ed27a490820696d7754d19f98cd6ecd11` 对当前 `f78e27a5f2139480a90e5f7a8cdcfa94c48ab48c`，每对固定同一张 RTX 4090、相同环境；YOLO 使用物理 GPU1，其余模型与短程套件使用 GPU0。
- 结构短程对照：简化前 `9bc45d21345190079aa90339e48bc1d4aeafca33` 对当前 `f78e27a`，分别在 NVIDIA、沐曦执行。
- C500 完整预算：同样比较 `9bc45d2` 与 `f78e27a`，两版均显式使用既有 `metax_deterministic_v1`。原版 UP 不能直接作为 C500 专用扩展基线。

本轮不恢复旧多随机种子或 `portable_v1` 阶段，不修改模型算法、不放宽容差。两平台的结果分别用于同平台版本回归，不据此声称跨厂商精度等价。

## 固定协议

| 模型 | 数据 / 预算 | 每版更新数 | 种子 |
| --- | --- | ---: | ---: |
| UP ResNet18 | CIFAR-10 子集，20 epochs | 3140 | 131 |
| UP YOLOv5-S SiLU | PennFudan，300 epochs | 600 | 131 |
| 官方 DETR R50 的外部 UP 适配器 | PennFudan，12 epochs | 720 | 131 |
| UP ViT-B/16 | CIFAR-10 子集，20 epochs | 3140 | 131 |

CIFAR-10 为固定 10,000 train / 2,000 test，PennFudan 为固定 120 train / 50 test；沿用历史冻结划分、初始权重与完整训练预算。DETR 不是上游 UP 原生模型。结果只取最终 epoch，不选最好权重、epoch 或种子。

ResNet/ViT 输入为 64px、batch64；ResNet 保留 UP 原始 ImageNet-style stem，ViT 保留完整 B/16 网络但使用 17-token 网格。YOLO 使用 320px mosaic、batch64、EMA off；DETR 使用 320/512px、batch2、原冻结分组学习率配方。因此“完整预算”不表示原分辨率、完整 ImageNet/COCO 训练。

严格判据沿用 `rtol=1e-4, atol=1e-5`，最终指标 `metric_atol=1e-6`。AP 原始值在 0 至 1 之间，表述为 AP 点时乘 100；分类 Top-1 已为百分数。**不使用历史多种子正负 1 个百分点统计界限替代本轮严格判据。**

## 已完成结果

- RTX 4090：32 项算子结果/所需梯度对照通过，无上游缺口；四模型 3 步训练、训练内 eval、checkpoint 恢复与独立推理对照全部通过。四模型完整预算的最终指标、逐步 loss 和最终权重也与原版 UP 逐位一致。
- C500 native：四模型功能阶段均完成；严格训练数值比较仅 ViT 通过，ResNet/YOLO/DETR 保留失败。另用同一份 native 短程权重在两版执行独立推理，四模型均逐位一致。
- C500 局部卷积梯度选项：ResNet/YOLO/DETR 的 9 次短程训练及独立推理完成，每模型两次版本/重复对照全部通过；不等于完整训练确定性。

| RTX 4090 完整预算 | 原版 UP | 当前 | 严格对照 |
| --- | ---: | ---: | --- |
| ResNet18 Top-1 | 65.9000015258789 | 65.9000015258789 | 通过，loss / 最终权重逐位一致 |
| DETR AP | 0.7805736186945724 | 0.7805736186945724 | 通过，loss / 最终权重逐位一致 |
| YOLO AP | 0.27423611431992984 | 0.27423611431992984 | 通过，loss / 最终权重逐位一致 |
| ViT Top-1 | 25.75 | 25.75 | 通过，loss / 最终权重逐位一致 |

| C500 完整预算 | 简化前 | 当前 | 严格对照 |
| --- | ---: | ---: | --- |
| ResNet18 Top-1 | 65.0999984741211 | 65.0999984741211 | 通过，loss / 最终权重逐位一致 |
| ViT Top-1 | 27.30000114440918 | 27.30000114440918 | 通过，loss / 最终权重逐位一致 |
| DETR AP | 0.7833888067264031 | 0.77452540224481 | 未通过 |
| YOLO AP | 0.3253444958463308 | 0.3253444958463308 | 通过，loss / 最终权重逐位一致 |

### DETR 失败与重复对照

两版均完成 720 步，初始状态及全部 720 个输入/前向前 RNG 审计一致，局部卷积梯度替换均实际执行 30,960 次。以首次更新为第 1 步计，第 59 步首次出现 loss 差异，第 65 步超出严格容差；这是首次记录到的 loss 差异，不是已定位的首个不同底层算子。最终 AP 差 -0.8863404481593062 点，权重最大绝对差 0.022658403497189283。

为区分版本影响与重现性，另外在同一物理设备、同一 `9bc45d2` 源码、同一数据/初始权重/种子/配置下完整重复 720 步。全部输入与 RNG 审计仍一致，但第 59 步开始出现 loss 差异，最终 AP 为 0.7842842896713277，与首次相差 +0.0895482944924586 点，严格对照同样失败。

这证明存在版本改动之外的同版本重现性问题，**尚不能证明全部版本差异只由某一个厂商算子造成**。既有选项只用 GPU FP32 按样本 GEMM 替换卷积权重梯度，保留原前向、输入/偏置梯度及其他操作；它不是全流程确定性保证，也不是 `portable_v1` 的 FP64 实验。此次不改运行时以强行消除差异。

## 源码与环境

以下 SHA256 对 `up/`、`configs/` 及平台 `src/up_platform/` 中声明的源码/配置文件按 CRLF 转 LF 后生成；构建产物单独校验，不把安装生成的缓存当源码差异。

| 快照 | 文件数 | 归一化 manifest SHA256 |
| --- | ---: | --- |
| 原版 UP `52aaeb9` | 618 | `35ea0ab7da9c01bb4b7fe192194e36912a084f14b3f8c1deaab8df9a7e54ca1e` |
| 简化前 `9bc45d2` | 727 | `7814ac208bd5126534cda425aa50daa91405fc0579f0816080eb63ea3f515892` |
| 当前 `f78e27a` | 727 | `cfad4bd04aa7f904f07c91ca199522f2b1bd19927e659bff7a1de91f3d586713` |

NVIDIA 两版在独立任务 venv 中使用 Python 3.12.3、Torch 2.10.0+cu126、TorchVision 0.25.0+cu126、RTX 4090、驱动 580.178.04。实际编译工具链为 CUDA toolkit 12.8.93，与 Torch 的 CUDA 12.6 有小版本差异；构建日志保留该警告，实际编译与 32 项操作对照通过，不能把这个组合推广为任意 toolkit 混装建议。

原版和当前版 NVIDIA `csrc` 的 43 个文件归一化内容完全相同。原版扩展独立编译；简化前平台仅在核对 43 个文件和二进制 SHA 后复用当前编译产物，两者不是伪装成独立编译。

C500 使用厂商容器、Python 3.12.11、Torch 2.10.0+metax3.8.1.0、TorchVision 0.25.0+metax3.8.1.0。镜像不可变 digest 为 `cr.metax-tech.com/public-library/maca-pytorch@sha256:1a10a6cc56741f5e84da5dd6f0aa9dc4cfd420d66188e43983c9a3efd9d85ab0`。`torch.version.cuda` 报告的 `11.6` 是厂商兼容字段，不能解释为使用 NVIDIA CUDA 11.6 工具链。

两版同平台进程的配置、软件环境、初始状态、输入和 RNG 分别比较。两平台之间 Python 补丁版本和厂商运行时不同，本轮并非消除这些差异后的跨厂商等价实验。

## 复现

### 1. 冻结源码和数据

从仓库根目录准备独立目录，不覆盖已有训练输出；以下 `RUN` 替换成自己的空目录：

```bash
RUN=/path/to/new-validation
mkdir -p "$RUN"
git worktree add --detach "$RUN/upstream" 52aaeb9ed27a490820696d7754d19f98cd6ecd11
git worktree add --detach "$RUN/baseline" 9bc45d21345190079aa90339e48bc1d4aeafca33
git worktree add --detach "$RUN/changed" f78e27a5f2139480a90e5f7a8cdcfa94c48ab48c
```

本轮直接复用此前冻结的完整 `data/` 包，没有重采样。数据包 SHA256 为 `7a2171e949b90b86d58b61f59f438ca4abe3fd50645520b28ecda562a313e80f`。数据和权重不在 Git 仓库中；已有数据时执行下面的验证，迁移主机仅调整路径字段，并重新验证全部图片/标注、锚框、权重和官方 DETR 源码哈希。NVIDIA 的实际路径迁移记录保留于原始证据。

```bash
DATA=/path/to/frozen/data.json
python benchmark/hardware/prepare_cross_vendor_data.py --manifest "$DATA" --verify
```

无数据时先按[历史数据准备方法](hardware/MULTISEED_ACCURACY.md#复现)准备，旧 `tools/prepare_*.py` 已迁移到 `benchmark/hardware/`。这里只复用数据准备步骤，不执行其中的多种子控制器；重新转换权重或编码图片可能产生不同文件字节，不能绕过冻结数据哈希后继续声称严格复现。

### 2. 编译 NVIDIA 扩展

先匹配上面记录的环境和业务依赖，使用同一 Python；不要同时安装两棵 UP 的 editable 包。验证驱动会优先载入所选源码及其平台包。

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2
(cd "$RUN/upstream" && python setup.py build_ext --inplace)
(cd "$RUN/baseline/packages/up-platform" && UP_BUILD_BACKEND=nvidia python setup.py build_ext --inplace)
(cd "$RUN/changed/packages/up-platform" && UP_BUILD_BACKEND=nvidia python setup.py build_ext --inplace)
```

C500 不执行 NVIDIA 构建，保留厂商 Torch/SDK。实际执行使用了单独冻结的验证工具快照，而非旧基线自带驱动；工具原字节与 SHA 随证据保存。本轮驱动修正只涉及选择对应源码的包路径和在首次初始化时选定 numerical profile，没有修改模型计算、预算或容差。

### 3. 运行完整固定预算

仅在确认 GPU 可用后设置物理编号，模型的一对运行必须用同一物理设备。NVIDIA 与 C500 分开执行，下面的循环只启动单一种子：

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0 DEFAULT_TASKS=cls:det
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_DISABLE_CUDA_CUDNN_TF32=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
GPU=0
for MODEL in resnet18 yolov5s detr vit_b16; do
  python benchmark/hardware/validate_current_accuracy.py \
    --baseline "$RUN/upstream" --changed "$RUN/changed" --data "$DATA" \
    --backend nvidia --device "$GPU" --model "$MODEL" \
    --numerical-profile native --output "$RUN/nvidia-full/$MODEL"
done
```

C500 使用匹配容器及同样环境变量，另外为无 passwd 条目的容器 UID 设置真实 `USER`/`LOGNAME`。把上面的 baseline 改为 `$RUN/baseline`、backend 改为 `metax`、profile 改为 `metax_deterministic_v1`、输出改为新的 `metax-full/`。不要把它标为 native 结果。

该入口固定 seed=131、epochs 和更新预算；`multiseed_v1/v2` 只是沿用的配方版本名，**没有启动多种子训练**。每版输出包括 `report.json`、实际配置、完整训练状态和全部 batch/RNG 哈希；比较器非零退出表示失败，不能删除目录后悄悄重跑。输出目录已存在时会拒绝覆盖。

### 4. 短程、推理与审计

```bash
CUDA_VISIBLE_DEVICES="$GPU" python benchmark/hardware/validate_platform_equivalence.py suite \
  --baseline "$RUN/baseline" --changed "$RUN/changed" --data "$DATA" \
  --backend nvidia --output "$RUN/short-native"

CUDA_VISIBLE_DEVICES="" python benchmark/hardware/audit_cross_vendor_accuracy.py \
  --runs "$RUN/nvidia-full/resnet18/baseline" "$RUN/nvidia-full/resnet18/changed" \
  --output "$RUN/checkpoint-audit.json"
CUDA_VISIBLE_DEVICES="" python benchmark/hardware/recompute_cross_vendor_metrics.py \
  --runs "$RUN/nvidia-full/resnet18/baseline" "$RUN/nvidia-full/resnet18/changed" \
  --output "$RUN/metric-audit.json"
```

对其他模型和 C500 的全部结果执行同样 CPU 审计：检查完整状态与有限值，重算最终预测/真实标注指标，核对报告和 checkpoint SHA。CPU 指标复算不是 CPU 模型代算。短程套件还逐项检查训练内 eval 的模式恢复、状态不更新、用途不切换，独立推理不创建训练对象、不加载训练用途模块。

同权重跨版本推理和同版本 DETR 完整重复属于辅助控制，不能拿它们替换原始失败。

## 验证与证据

主分支只保留[机器可读摘要](current-validation-20261022.json)；完整日志、JSON、实际执行的工具字节和二进制校验索引单独放在[本轮验证 Release](https://github.com/jingfengou/United-Perception-Hardware/releases/tag/validation-current-20261022)。这不是“所有测试通过”的稳定版本发布，包含完整未通过结果。

| 原始证据包 | 字节数 | SHA256 |
| --- | ---: | --- |
| `nvidia-current-evidence.tar.gz` | 3717417 | `8fa3121ccd711934716d51c9171e0efbf10953986119cbdb16a3f0d679b89476` |
| `metax-current-evidence.tar.gz` | 4927508 | `b35a1abf5aaa611eaf16938552b3155e249ce5ba5df3ae616102b0009814e46d` |

每包提供同名 `.index.json`。下载后已实际校验归档 SHA256，并逐文件核对内部索引：NVIDIA 245、C500 359 个文本证据文件全部通过。393 个大二进制文件记录保留在索引中，训练权重/梯度文件仍在各自任务目录，没有把数据或大权重加入 Git。归档内收尾进度是打包时快照，最终完成戳另行回读并写入发布摘要。

16 次主对照训练均完成，另有 1 次 C500 同版本 DETR 控制，共 31,120 次完整训练更新。全部 17 次结果的 checkpoint 模型/优化器/调度器完整性、有限值及最终预测 CPU 指标复算通过；这表示产物与报告一致，**不把失败的版本对照改记为通过**。全部主对照逐步输入和 RNG 审计相同。

实际驱动快照为 `drivers-v2.tar`，SHA256 为 `7d5d148a05a4aa3e0e083858bda94365febddf18d2626843b0130795f8dccb62`；初始备用 `current-drivers.tar` 未用于执行。公开工具与执行字节已按仅 CRLF/LF 归一化逐项核对；原字节及两种哈希见摘要和归档，不能把换行差异当作模型实现变更。DETR 辅助 source 审计的首版误比较了缺失字段，已保留并由 `detr-baseline-audit-v2.json` 使用实际 `normalized_source_manifest_sha256` 更正，不以首版字段作为证据。

本地最终检查：

- CPU：326 tests + 3 subtests 通过；保留 2 个既有 legacy ONNX 弃用警告。
- 31 个运行时成员、73 项操作契约无漂移，硬件边界违规 0。
- 665 个受保护算法/配置/数学实现文件和 6 个历史协议夹具未变。
- 公共平台及 UP wheel/sdist 构建通过；无 UP、无 Torch 的隔离环境可导入已安装平台 wheel。
- README 图片/本地链接检查及 900px/360px、明/暗预览通过；源码包包含新架构 SVG。

尚未覆盖：多随机种子统计、全量 ImageNet/COCO、AMP/多卡/GPU ONNX/稀疏与可选 SDK、昇腾/寒武纪真机，以及 C500 DETR 的首个不同底层算子定位。本轮没有数学修复，不删除失败、不放宽容差，也不自动恢复旧精度任务。历史冻结报告与证据仍按原标签保存。
