# LightX2V 风格重构：双平台真机回归

本轮比较同一平台的重构前后，不比较 NVIDIA 与沐曦的精度等价。运行时固定为重构前 `ad36d7728b19ae6229e22923d2918feeb6b9ad0d`、重构后 `71d98124ec6cebf66b08ff2cf2333c059108773f`。后续交付提交只增加验证工具、测试和文档，不改 `up/`、模型算法或平台数学实现。

## 结论

NVIDIA native 与 C500 显式 v2 的四模型同平台重构回归通过：16 次主训练、30,400 次更新，全部逐步 loss、最终权重、指标及最终验证/独立推理预测跨版本逐位一致。完整预算后的模型、优化器、调度器恢复、训练用途 eval 状态保护和全部留出批次独立推理均通过。

C500 默认 native 不能改记为严格训练通过；本轮保留三模型失败、同版本重复及固定同一权重的推理控制，见下表。旧多种子任务未恢复；本次交付没有修改模型、配方或平台数学实现。

## 完整预算结果

以下均为最终 epoch；Top1 单位为百分数，检测 AP 使用 0–1 数值，不跨厂商比较同一行的绝对数值。

| 模型 | NVIDIA native：前 → 后 | C500 显式 v2：前 → 后 | 两平台各自 loss / 最终权重最大绝对差 |
| --- | --- | --- | --- |
| ResNet18 | Top1 65.9000015259 → 65.9000015259 | Top1 65.0999984741 → 65.0999984741 | 0 / 0 |
| YOLOv5-S | AP 0.2742361143 → 0.2742361143 | AP 0.3253444958 → 0.3253444958 | 0 / 0 |
| DETR R50 适配器 | AP 0.7805736187 → 0.7805736187 | AP 0.7706035845 → 0.7706035845 | 0 / 0 |
| ViT-B/16 | Top1 25.75 → 25.75 | Top1 21.6499996185 → 21.6499996185 | 0 / 0 |

全部 30,400 次更新均保留；每一对版本的初始状态、逐批输入及 RNG 审计一致。表中显示位数有限，原始 JSON 保存完整精度及全部 Top5/AP/AR 分项。这里证明的是既定小数据预算的重构回归，不以绝对指标宣称大基准收敛。

最终 checkpoint 的全部验证批次均保存并在独立 inference 进程重放：每个版本每个平台 ResNet 32 批、YOLO 1 批、DETR 25 批、ViT 32 批；对应完整 2,000 / 50 / 50 / 2,000 张留出图片。恢复用途 eval、独立逐批推理及额外单图 `BaseInference.predict()` 的跨版本最大绝对差均为 0。训练/推理用途没有混用，C500 没有加载本项目 NVIDIA `_C`。

## C500 Native 辅助控制

此处是 3 步、8 train / 4 test 的短程控制，不是额外的 native 完整预算或新多种子阶段。对原生失败的模型仅按预先声明重复同一 baseline 一次，使用同一 seed、初始权重、输入和 RNG；保存所有结果，不挑选通过样本。

| 模型 | native 重构前后短程严格对照 | 同一 baseline 重复一次 | 固定 baseline 同一权重：跨版本 native 推理 |
| --- | --- | --- | --- |
| ResNet18 | 未通过：第 3 步卷积梯度及 momentum | 同样未通过；第 3 步首个超差卷积梯度张量约 2.70e-5 | 首个保存验证批次及单图 API 均逐位一致 |
| YOLOv5-S | 未通过：第 3 步卷积梯度及 momentum | 同样未通过；第 3 步首个超差卷积梯度张量约 2.25e-5 | 首批 / 单图均逐位一致 |
| DETR | 未通过：训练后 eval/预测 | 梯度/状态在原容差内但非逐位一致；同版本 eval 也超原容差，首个超差输出张量约 1.01e-3 | 首批 / 单图均逐位一致 |
| ViT-B/16 | 通过 | 按原计划未追加重复 | 首批 / 单图均逐位一致 |

DETR 同版本比较的最终 checkpoint 最大差为约 `4.08e-7`，第 3 步梯度约 `5.74e-7`；小权重变化仍可在后处理输出上超过既定判据。因此额外在 CPU 上比较已保存的 eval-1 / eval-2 / restored-output，不能只看梯度在容差内就宣称全流程一致。

这说明三个失败现象无需这次重构也可出现，不能将它们直接归因于目录/路由迁移；本轮没有逐个内核重新做因果归因，既有操作级证据见[卷积诊断](platform/METAX_DETERMINISM.md)和[官方 DETR / ATen 归因](DETR_ORIGIN_20261022.md)。**两个 native 训练产物不同导致的推理差异，不等于固定同一权重的推理不稳定。** 本轮固定权重控制通过仍不保证所有输入、形状或后续运行都确定。

## 比较边界

- NVIDIA：同一张 RTX 4090 物理 GPU1，两个版本均为 `native`。
- 沐曦：同一张 C500 物理 GPU2，两个版本均显式选择原有 `metax_deterministic_v2`；不把这一结果记成默认 native 训练通过。
- 另外保留 C500 native 短程对照。失败时，同一基线、同一 seed 再运行一次作为随机性控制；各模型还用同一份 baseline native 权重做跨版本独立推理。
- 全部模型使用 seed=131、同平台相同环境、冻结数据/初始权重/配置与数值策略。只比较最终 epoch，不挑最好权重或结果。
- 容差保持 `rtol=1e-4, atol=1e-5`，最终指标绝对容差为 `1e-6`。逐位一致是实测结果，不以修改判据获得。
- 不启用 `portable_v1`，不使用 CPU/FP64 模型代算，不启用额外厂商 BLAS/DNN 实验开关，不恢复旧多随机种子任务。

路由中的 `portable` 是同输入设备上的公共 PyTorch 实现，不是 `portable_v1` 数值实验，也不是 NVIDIA 二进制。C500 的设备名虽然为 cuda，实际使用厂商 Torch；最终独立推理另行记录是否加载了本项目的 NVIDIA `_C`。

## 验证层次

| 层次 | 实际检查 |
| --- | --- |
| NVIDIA 构建与操作 | 两棵源码分别实际编译 `_C`；32 项原生用例比较前向和所需梯度，包括既有显式 CPU 操作，不冒充 32 项全为 GPU 内核 |
| 短程真实训练 | 每版本、每模型训练 3 步；固定原划分的前 8 个训练 / 4 个测试样本，逐步核对输入、RNG、loss、完整梯度和更新后状态 |
| 训练中的 eval | 在实际训练循环中验证；平台用途与供应者保持 training，eval 不更新模型、梯度、优化器和 scaler，恢复模块模式及参数身份 |
| 完整固定预算 | 下表 16 次主训练；比较全部逐步 loss、数据/RNG 审计、最终指标和最终权重，不能据此声称保存并比较了每一步的全部梯度 |
| 最终权重恢复 | 新进程、新 Runner 恢复模型、优化器和调度器，核对完整状态；重算最终 eval，验证状态不被更新 |
| 独立推理 | 另起 inference 进程，实际调用 `BaseInference.predict()`，再重放每一个真实留出批次，与训练用途 eval 输出严格比较；不创建优化器/scaler，不加载训练 backend 模块 |
| CPU 审计 | 检查保存状态完整性及有限值；用保存的预测和真实标注重算全部最终指标。这是指标审计，不是 CPU 模型代算 |

最终恢复/推理还检查 `up` 与 `up_platform` 实际导入路径，防止错误使用另一棵 editable 安装。各用途供应者在同平台两版本间比较，不要求训练和推理本身拥有完全相同的能力集合。

这里的恢复指 `state_dict`、优化器/调度器状态及评估结果恢复，不冒充中断续训时完整 RNG 和数据迭代器位置的恢复验收。部署入口使用只含 model 的 checkpoint；含额外 Python 对象的历史整包自动恢复不列为通过项。FP32 配方没有使用 scaler，不能据此声称 AMP 恢复已验证。

## 固定预算

| 模型 | 数据 / 输入 / 配方 | 每版本更新数 |
| --- | --- | ---: |
| UP ResNet18 | CIFAR-10：10,000 train / 2,000 test；64px，batch64，20 epochs；保留原 ImageNet-style stem | 3140 |
| UP YOLOv5-S SiLU | PennFudan：120 train / 50 test；320px mosaic，batch64，300 epochs，EMA off | 600 |
| 官方 DETR R50 的外部 UP 适配器 | 同一 PennFudan；320/512px，batch2、分组采样及单图尾批；原分组学习率和梯度裁剪，12 epochs | 720 |
| UP ViT-B/16 | 同一 CIFAR-10；64px、17-token 网格，完整 B/16 网络，batch64，20 epochs | 3140 |

共 16 次主训练、30,400 次更新。DETR 不是上游 UP 原生模型；两版本使用同一份已冻结官方源码、权重及外部适配器。`multiseed_v1/v2` 是沿用的配方名，本轮只运行 seed=131。

## 环境与冻结

| 环境 | NVIDIA | 沐曦 |
| --- | --- | --- |
| Python | 3.12.3 | 3.12.11 |
| PyTorch | 2.10.0+cu126 | 2.10.0+metax3.8.1.0 |
| TorchVision | 0.25.0+cu126 | 0.25.0+metax3.8.1.0 |
| Triton | 3.6.0 | 3.6.0+metax3.8.1.0 |
| NumPy / SciPy / einops / pycocotools | 1.26.4 / 1.16.0 / 0.8.1 / 2.0.10 | 同左 |
| 数值策略 | native | 显式 v2；另保留 native 短程控制 |

环境在每个版本进程中读取并比较。同平台版本间一致；两平台的厂商构建和 Python 补丁版本不同，不能称为跨厂商环境完全相同。FP32、TF32 off、确定性请求和线程设置沿用原验证协议。实际训练 PID 的环境白名单另行保存，三个 `MCDNN_*` 实验开关均未设置。

NVIDIA 使用现有 CUDA toolkit 12.8.93 和 GCC 13.3.0；与 Torch CUDA 12.6 的小版本差异警告保留，两版本使用同一工具链并分别编译，没有复用旧 `_C`。这不是通用混装建议。

C500 保留厂商容器，镜像 digest 为 `cr.metax-tech.com/public-library/maca-pytorch@sha256:1a10a6cc56741f5e84da5dd6f0aa9dc4cfd420d66188e43983c9a3efd9d85ab0`。MACA 3.8.1.3，驱动 3.6.11；`torch.version.cuda=11.6` 是厂商兼容字段，不是 NVIDIA toolkit。v2 是既有固定 GPU FP32 计算策略，本轮没有再次修改或优化它，仍有原报告记录的性能代价。

源码 manifest 对 `up/`、`configs/` 与平台 `src/up_platform/` 中声明的源码/配置仅作 CRLF→LF 归一化，二进制另行校验：

| 快照 | manifest SHA256 |
| --- | --- |
| 重构前 ad36d77 | `18e4171327dd0de6da77db99ebe0085c09150c3a008b09b014282bf55fe997c9` |
| 重构后 71d9812 | `3b796cdf2d5b4c6044904e6e91d0e4c7edebab9cb4b7eb34bae087a4bbce9e3c` |

两台机器接收的 baseline/changed 源码包及训练驱动字节相同，源包 SHA 和实际执行工具字节保存在证据中。完整数据包沿用 `7a2171e949b90b86d58b61f59f438ca4abe3fd50645520b28ecda562a313e80f`；跨主机仅沿用已核验的路径迁移，不重新抽样、编码图片或转换权重。

## 复现

### 1. 准备两棵固定源码和原数据

在匹配环境中准备新的空目录；`TOOLS` 指向本交付版本的 `benchmark/hardware`，不是任意历史快照自带的旧工具。

```bash
RUN=/path/to/new-rearchitecture-validation
TOOLS=/path/to/United-Perception-Hardware/benchmark/hardware
DATA=/path/to/frozen/data.json
mkdir -p "$RUN/baseline" "$RUN/changed"
git archive ad36d7728b19ae6229e22923d2918feeb6b9ad0d up configs packages/up-platform |
  tar -xf - -C "$RUN/baseline"
git archive 71d98124ec6cebf66b08ff2cf2333c059108773f up configs packages/up-platform |
  tar -xf - -C "$RUN/changed"
python "$TOOLS/prepare_cross_vendor_data.py" --manifest "$DATA" --verify
```

数据、外部 DETR 和大权重不在 Git 中。原数据准备及哈希约束见[前轮复现](CURRENT_VALIDATION_20261022.md#复现)。移动数据目录时应保留图片、标注、锚框、权重及官方源码哈希；不能用重新抽样的数据冒充严格复现。

NVIDIA 两版本各自编译；C500 不执行此步骤，更不能用 NVIDIA wheel 替换厂商 Torch：

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export UP_BUILD_BACKEND=nvidia TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2
(cd "$RUN/baseline/packages/up-platform" && python setup.py build_ext --inplace)
(cd "$RUN/changed/packages/up-platform" && python setup.py build_ext --inplace)
```

### 2. 运行同平台短程及完整预算

只使用实际获配的物理 GPU；同一对版本始终使用同一张卡。以下 NVIDIA 设置改为 `BACKEND=metax PROFILE=metax_deterministic_v2 GPU=2` 即为本轮 C500 主对照。

```bash
BACKEND=nvidia PROFILE=native GPU=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0 DEFAULT_TASKS=cls:det
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_DISABLE_CUDA_CUDNN_TF32=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export CUDA_VISIBLE_DEVICES="$GPU"
# 容器 UID 无 passwd 条目时，使用其真实用户名：
# export USER=oujingfeng LOGNAME=oujingfeng
python "$TOOLS/validate_platform_equivalence.py" suite \
  --baseline "$RUN/baseline" --changed "$RUN/changed" --data "$DATA" \
  --backend "$BACKEND" --numerical-profile "$PROFILE" --output "$RUN/short-primary"
for MODEL in resnet18 yolov5s detr vit_b16; do
  python "$TOOLS/validate_current_accuracy.py" \
    --baseline "$RUN/baseline" --changed "$RUN/changed" --data "$DATA" \
    --backend "$BACKEND" --device "$GPU" --model "$MODEL" \
    --numerical-profile "$PROFILE" --output "$RUN/full/$MODEL"
done
```

这些命令拒绝覆盖既有输出。非零退出就是失败，不能删除后悄悄重跑。C500 native 辅助对照另建输出并明确使用 `--numerical-profile native`；不和 v2 的结果混合。

### 3. 最终 checkpoint 恢复、训练用途 eval 和独立推理

```bash
for MODEL in resnet18 yolov5s detr vit_b16; do
  for LABEL in baseline changed; do
    OUT="$RUN/final-checkpoints/$LABEL/$MODEL"
    python "$TOOLS/audit_rearchitecture_checkpoints.py" training \
      --source "$RUN/$LABEL" --training "$RUN/full/$MODEL/$LABEL" \
      --data "$DATA" --backend "$BACKEND" --model "$MODEL" \
      --numerical-profile "$PROFILE" --output "$OUT/training"
    python "$TOOLS/audit_rearchitecture_checkpoints.py" inference \
      --source "$RUN/$LABEL" --reference "$OUT/training" \
      --data "$DATA" --backend "$BACKEND" --model "$MODEL" \
      --numerical-profile "$PROFILE" --output "$OUT/inference"
  done
  CUDA_VISIBLE_DEVICES="" python "$TOOLS/audit_cross_vendor_accuracy.py" \
    --runs "$RUN/full/$MODEL/baseline" "$RUN/full/$MODEL/changed" \
    --output "$RUN/$MODEL-checkpoint-audit.json"
  CUDA_VISIBLE_DEVICES="" python "$TOOLS/recompute_cross_vendor_metrics.py" \
    --runs "$RUN/full/$MODEL/baseline" "$RUN/full/$MODEL/changed" \
    --output "$RUN/$MODEL-metric-audit.json"
done
```

`training` 动作是对已完成完整训练的最终状态进行恢复和 eval，不是再次训练。它保存全部真实验证批次及输出；`inference` 动作新建独立推理对象，逐批严格比较。原始执行记录还包含跨版本最终预测文件的 CPU 比较、native 同版本重复及同权重推理命令。

### 4. Native 控制单独运行

在与 C500 主对照相同的卡和环境中，先用第 2 节的 `suite` 命令另建 `short-native-control`，明确改为 `--backend metax --numerical-profile native`。它本轮非零退出，失败摘要和产物必须保留。对本轮三个失败模型进行单次 baseline 重复，再用同一 baseline native 权重运行两版本推理：

```bash
for MODEL in resnet18 yolov5s detr; do
  python "$TOOLS/validate_platform_equivalence.py" training \
    --source "$RUN/baseline" --data "$RUN/short-native-control/data/data.json" \
    --backend metax --model "$MODEL" --numerical-profile native \
    --output "$RUN/native-controls/repeat/$MODEL"
done
for MODEL in resnet18 yolov5s detr vit_b16; do
  for LABEL in baseline changed; do
    python "$TOOLS/validate_platform_equivalence.py" inference \
      --source "$RUN/$LABEL" --data "$RUN/short-native-control/data/data.json" \
      --training "$RUN/short-native-control/baseline/$MODEL/training" \
      --backend metax --model "$MODEL" --numerical-profile native \
      --output "$RUN/native-controls/same-weights/$LABEL/$MODEL"
  done
done
```

每次 training 仍是固定 3 步。用 `validate_platform_equivalence.close_trees()` 在 CPU 上比较 checkpoint、三步 gradients 和全部保存的 eval/恢复输出；原控制器、CPU 补充比较器及实际命令随证据包保留。`close_trees` 的 rtol/atol 不变，失败记录不能覆盖或删除。

## 执行问题与处理

- 首次准备时，导入路径在工具解包前加入，Python 缓存了当时不存在的目录，产生 `ModuleNotFoundError`。故障发生在 GPU 训练前；原脚本和日志保留，接续工具只在解包后加入路径，并逐文件核对原始源包，没有重新生成数据或修改冻结源码。
- NVIDIA 四个完整训练及比较子进程全部返回 0 后，控制器向已断开的标准输出管道写日志，产生 `BrokenPipeError`。原 `queue-status.json` 和未执行 GPU 审计的等待器目录保存在 `orchestration-recovery/`。恢复收据先检查所有既有训练/比较结果、源码和旧控制器退出，再确认 GPU 训练已完成；仅用原审计控制器接续最终恢复/推理，标准输出改存文件。没有重跑训练，也没有把控制器异常当成数值通过。

## 证据和限制

[机器可读结果](rearchitecture-hardware-20261023.json)由两个已封存证据索引生成；同时核对实际执行的 9 份公开工具与交付文件内容，仅允许 CRLF/LF 归一化。`up/`、`configs/` 和平台运行时与受测 `71d9812` 无改动，重构前后 UP 模型/配置也无改动。

原始文本、实际控制器/工具、冻结源包、CPU JUnit 和逐文件索引见[独立证据 Release](https://github.com/jingfengou/United-Perception-Hardware/releases/tag/validation-rearchitecture-20261023)。全部归档成员已下载并逐文件校验；大权重、输入和梯度保留在服务器，公开索引只记录大小与 SHA256，不冒充可下载的完整二进制包。

| 证据包 | 字节 | 归档 SHA256 | 包内索引文件数 / 服务器二进制索引数 |
| --- | ---: | --- | ---: |
| nvidia-rearchitecture-evidence.tar.gz | 6,831,998 | `06c3e4e47e5be82dd29add4f7a4bc1820af23ebd53b6e5472f41c9624a3c7e5b` | 341 / 344 |
| metax-rearchitecture-evidence.tar.gz | 7,193,171 | `2e9a4c15ed18bfb1387f2e8233c58a47d78d6247cc663ef3315161f4883cd390` | 451 / 498 |

文件数不含包内 `evidence-index.json` 自身；同名 `.tar.gz.index.json` 一并发布。先检查下载的源包 SHA，再提取标准库校验器，检查所有成员及内外索引一致性，不需要 UP、Torch 或 GPU：

```bash
BASE=https://github.com/jingfengou/United-Perception-Hardware/releases/download/validation-rearchitecture-20261023
EVIDENCE=/path/to/new-evidence-directory
mkdir "$EVIDENCE"
for BACKEND in nvidia metax; do
  ARCHIVE="${BACKEND}-rearchitecture-evidence.tar.gz"
  curl -fL "$BASE/$ARCHIVE" -o "$EVIDENCE/$ARCHIVE"
  curl -fL "$BASE/$ARCHIVE.index.json" -o "$EVIDENCE/$ARCHIVE.index.json"
done
sha256sum "$EVIDENCE/"*-rearchitecture-evidence.tar.gz
# 与上表核对一致后再提取已校验包中的校验器。
tar -xzf "$EVIDENCE/nvidia-rearchitecture-evidence.tar.gz" -C "$EVIDENCE" verify_evidence.py
python "$EVIDENCE/verify_evidence.py" --directory "$EVIDENCE" --output "$EVIDENCE/verified.json"
```

本地全仓 432 tests + 3 subtests（JUnit 435 case）零失败、错误与跳过，保留两条既有 ONNX 弃用 warning；665 项算法/配置保护、80 项迁移实现、六个历史协议夹具、零硬件边界违规及 31 运行时成员 / 73 操作无漂移检查均通过。CPU 检查与本轮 GPU 验收分开记录；云端 CI 以交付提交对应的工作流和 JUnit 为准。

两平台所有本任务 GPU 进程均已退出；收尾快照和锁作用域保留，未终止其他任务。首次准备失败、NVIDIA 日志管道异常、native 三模型失败及旧数值阶段没有删除或改写。

本轮不代表：全量 ImageNet/COCO 收敛、多随机种子统计、跨厂商精度等价、AMP/多卡、GPU 导出、所有专用操作/形状/布局/第三方 SDK，或昇腾/寒武纪真机通过。短程梯度、完整预算结果、原生随机性和数值策略分别记录，不相互替代。
