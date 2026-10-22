# United Perception Hardware

[![CPU contracts and packaging](https://github.com/jingfengou/United-Perception-Hardware/actions/workflows/hardware-boundaries.yml/badge.svg)](https://github.com/jingfengou/United-Perception-Hardware/actions/workflows/hardware-boundaries.yml)

**保留 UP 的算法与配方，只适配硬件差异。**

基于 [ModelTC/United-Perception](https://github.com/ModelTC/United-Perception)，将设备、专用算子和厂商 SDK 收入同仓独立包 [`up-platform`](packages/up-platform/)。训练与独立推理分开适配，公共能力只维护一份；原有 `up.accelerator`、算子导入和 `.apply()` 入口继续兼容。

[使用手册](docs/platform/README.md) · [接口定义](docs/platform/INTERFACES.md) · [新增厂商](docs/platform/ADDING_VENDOR.md) · [能力矩阵](docs/platform/CAPABILITIES.md) · [验证记录](docs/reports/README.md)

## 验证状态

验证运行时冻结于 `f78e27a`。本轮使用真实 CIFAR-10 / PennFudan 子集，四模型、单种子、完整固定预算；**不是完整 ImageNet/COCO 收敛，也不是跨厂商精度等价证明**。

| 范围 | 实测结论 |
| --- | --- |
| NVIDIA RTX 4090 | 四模型与原版 UP 完整预算结果逐位一致；32 项算子与训练/eval/恢复/独立推理短程对照通过 |
| C500 native | 同权重独立推理四模型逐位一致；训练短程数值对照仅 ViT 通过 |
| C500 可选策略 | ResNet18、YOLO、ViT 完整预算逐位一致；DETR 未通过 |
| 昇腾 / 寒武纪 | 有设备入口，专用操作与真机验收未齐全 |

C500 完整预算两版均显式启用 `metax_deterministic_v1`，不代表默认 native 训练通过。失败与同版本重复对照均保留；旧多随机种子复测仍暂停，CPU CI 不代表 GPU 精度验收。[本轮协议与结果](docs/reports/CURRENT_VALIDATION_20261022.md) · [历史结果与证据](docs/reports/README.md)

DETR 后续归因已在不经过 UP 的官方网络反向中定位到 FP32 矩阵乘法非确定性，并用纯 PyTorch 固定操作数复现；这不把原精度失败改判通过。[定位过程与复现](docs/reports/DETR_ORIGIN_20261022.md)

## 工作方式

![UP 通过兼容门面进入平台包；训练及训练内 eval、独立推理分别路由，复用 common/shared 与已选厂商实现。](assets/readme/platform.svg)

- **UP 管算法与业务。** 模型、损失、数据、学习率和训练步骤留在 UP；普通设备无关 PyTorch 计算不重复包装。
- **平台管硬件差异。** 设备运行时、AMP、通信、专用操作和层按厂商路由；缺失能力明确报错，不新增静默 CPU 模型代算。
- **用途由入口决定。** 训练内 eval 和 `train --evaluate` 保持 training 路径，退出后恢复模块状态；独立推理与导出走 inference，不依据 `.eval()` 猜测用途。

## 安装

在目标 Linux 主机或厂商容器中，先准备匹配的驱动、SDK、PyTorch/TorchVision 和 UP 业务依赖，再安装平台包和 UP。**不要用 NVIDIA wheel 覆盖厂商 PyTorch。**

```bash
git clone https://github.com/jingfengou/United-Perception-Hardware.git
cd United-Perception-Hardware
python -m pip install setuptools wheel packaging
UP_BUILD_BACKEND=none python -m pip install --no-deps --no-build-isolation -e ./packages/up-platform
python -m pip install --no-deps --no-build-isolation -e .
```

`--no-deps` 保留预先准备的环境，不会安装完整训练依赖。原 `requirements.txt` 是历史记录，不是现代环境的通用安装锁。已验证组合、依赖清单和环境限制见[环境说明](docs/platform/ENVIRONMENTS.md)。平台普通安装不探测或编译 GPU 扩展。

### NVIDIA 扩展

需要 UP 原生检测/点云操作时，在匹配的 CUDA toolkit 和编译器环境中显式构建：

```bash
# 4090 为 8.9，A800 为 8.0；按实际 GPU 设置。
UP_BUILD_BACKEND=nvidia TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 \
  python -m pip install --no-deps --no-build-isolation -e ./packages/up-platform
```

扩展归属 `up_platform.backends.nvidia._C`；UP 根目录不再编译扩展。上游保留的 `easy_setup.sh` 不是当前安装入口。

## 选择硬件与运行

编辑模型 YAML **已有的** `runtime` 段，保留其他字段：

```yaml
runtime:
  device: cuda
  fp16: false
  accelerator:
    backend: metax
```

| 显卡 | backend | device |
| --- | --- | --- |
| NVIDIA | `nvidia` | `cuda` |
| 沐曦 | `metax` | `cuda` |
| 昇腾 | `ascend` | `npu` |
| 寒武纪 | `cambricon` | `mlu` |

厂商身份不等于设备名；两家都叫 cuda 不会共用 NVIDIA 二进制。先从单卡 FP32 开始，再单独验证 AMP/多卡。

准备好数据、标注、权重与实际配置文件后：

```bash
python -m up train --config your_model.yaml --launch pytorch --ng 1 --backend dist
python -m up train --evaluate --config your_model.yaml --launch pytorch --ng 1 --backend dist

# 使用原检测推理接口的模型
python -m up inference --config your_detection_model.yaml \
  --ckpt /path/to/checkpoint.pth --img_path /path/to/images \
  --vis_dir vis_dir --launch pytorch --ng 1 --backend dist
```

评估指定权重时，在配置中设置 `saver.resume_model` 并核对 `saver.auto_resume` 的优先级。CLI 的 `--backend dist/linklink` 是通信策略，不是显卡厂商。独立使用平台时，显式指定 `purpose="training"` 或 `"inference"`，见[平台手册](docs/platform/README.md)。

## 新增厂商

**一个设备后端 + 实际训练/推理差异 + 缺失算子。** 不必先猜“支持哪些模型”，也不必重写全部 73 项操作。

1. 准备厂商匹配的 PyTorch、SDK 和通信库，记录环境组合。
2. 实现设备后端并 `register_backend(VendorDevice)`，训练/推理默认复用公共实现，不用写空类。
3. 只在确有差异时增加训练/推理实现及缺失算子，交付所需反向、状态和测试，再做真机验收。

从[逐步指南](docs/platform/ADDING_VENDOR.md)和[可安装插件示例](packages/up-platform/examples/vendor_plugin/)开始。函数签名、布局、空输入、梯度和导出要求见[完整接口索引](docs/platform/API_INDEX.md)；已有模型与配方保持不变，注册成功不等于真机通过。

默认 `native` 不变。`metax_deterministic_v1` 仅处理局部卷积权重梯度，**不能保证全流程确定性**；`portable_v1` 是另行隔离的实验选项，都不是基本接入要求。[数值策略边界](docs/platform/NUMERICAL_POLICIES.md)

## 开发与目录

```text
up/                     UP 业务与旧接口兼容
packages/up-platform/   独立平台包、厂商实现和契约测试
configs/                原有模型配方
benchmark/hardware/     真机验证、精度协议与诊断工具
tools/platform/         边界、契约、归档与发布检查
tests/                  兼容/协议/证据回归及必要小夹具
docs/platform/          当前开发手册
docs/reports/           冻结报告、归档清单和验证记录
docs/architecture/      总设计、变更台账及整理约束
```

```bash
python tools/platform/check_hardware_boundaries.py
python tools/platform/export_platform_contract.py --check
python tools/platform/export_accelerator_contract.py --check
python tools/platform/check_project.py
python -m pytest -q
```

CPU 测试、公共构建和 GPU 验收分别记录。目录迁移与旧命令见[工具入口](benchmark/hardware/README.md)，历史整理与简化见[报告索引](docs/reports/README.md)。

## 来源与证据

保留上游 Git 历史与 [Apache-2.0 LICENSE](LICENSE)，UP 基线为 `52aaeb9`。[总设计与逐文件台账](docs/architecture/PROJECT_REVIEW.md)记录相对原版的改动和原因。整理前 1,505 份原始证据已保存到[固定标签与 Release](https://github.com/jingfengou/United-Perception-Hardware/releases/tag/evidence-pre-consolidation-20261022)，主分支保留逐文件哈希、复现入口与失败结论，不改写历史。
