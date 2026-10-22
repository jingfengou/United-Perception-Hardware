# United Perception Hardware

基于 [ModelTC/United-Perception](https://github.com/ModelTC/United-Perception) 的多厂商硬件适配版。保留 UP 的模型、损失、数据和训练配方，将设备、专用算子与厂商 SDK 隔离到同仓独立包 [`up-platform`](packages/up-platform/)。

```text
UP 模型与业务 → up.accelerator 兼容门面 → up_platform
                                        ├ training：训练与训练内 eval
                                        ├ inference：独立推理与导出
                                        └ common/shared + 厂商后端
```

普通 PyTorch 计算直接复用；新增厂商只实现不兼容部分。缺少能力时明确报错，不新增静默 CPU 模型代算。训练中的 `.eval()` 不切换部署后端。

[平台手册](docs/platform/README.md) · [完整接口](docs/platform/INTERFACES.md) · [新增厂商](docs/platform/ADDING_VENDOR.md) · [能力矩阵](docs/platform/CAPABILITIES.md) · [验证与归档](docs/reports/README.md)

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
# 8.0 对应 A800；其他设备请设置其真实编译架构。
UP_BUILD_BACKEND=nvidia TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2 \
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

1. 准备厂商匹配的 PyTorch、SDK 和通信库，记录环境组合。
2. 实现设备后端，分别声明训练/推理用途；兼容操作直接复用公共实现。
3. 只覆盖缺失操作，交付所需反向、状态及契约测试，再做真机验收。

无需先猜厂商“支持哪些模型”，也不必重写全部 73 项操作。按[接口定义](docs/platform/INTERFACES.md)、[逐项索引](docs/platform/API_INDEX.md)和[可安装插件示例](packages/up-platform/examples/vendor_plugin/)实现与验证即可；已有模型和配方保持不变。

## 验证范围

以下是各自冻结版本的已有记录，不代表整理后的新版本已重新完成 GPU 验收：

| 范围 | 已有证据 / 限制 |
| --- | --- |
| NVIDIA A800 | 独立包四模型短程训练、训练 eval、恢复、独立推理及 32 项操作对照通过 |
| 沐曦 C500 native | 四模型可运行且同权重推理一致；严格训练数值对照仅 ViT 通过 |
| 沐曦局部确定性选项 | ResNet/YOLO/DETR 同种子短程对照通过，不等于全量或跨厂商精度等价 |
| 昇腾、寒武纪 | 有运行时入口，专用操作及真机验收未齐全 |

默认 `native` 不变。`portable_v1` 等[数值策略](docs/platform/NUMERICAL_POLICIES.md)显式启用、单独验收，不是基本厂商接入要求；旧多随机种子复测仍暂停。当前 CPU/构建/CI 结果见[本次整理记录](docs/reports/CONSOLIDATION_20261022.md)，历史结果见[报告索引](docs/reports/README.md)。

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

CPU 测试、公共构建和 GPU 验收分别记录，不把注册成功当成硬件支持。目录迁移与旧命令对照见[工具入口](benchmark/hardware/README.md)。

## 来源与证据

保留上游 Git 历史与 [Apache-2.0 LICENSE](LICENSE)，UP 基线为 `52aaeb9`。[总设计与逐文件台账](docs/architecture/PROJECT_REVIEW.md)记录相对原版的改动和原因。整理前 1,505 份原始证据已保存到[固定标签与 Release](https://github.com/jingfengou/United-Perception-Hardware/releases/tag/evidence-pre-consolidation-20261022)，主分支保留逐文件哈希、复现入口与失败结论，不改写历史。
