# UP Platform

<img src="up-logo.png" alt="United Perception" width="280" />

在原版 [United-Perception](https://github.com/ModelTC/United-Perception) 上重建的显式多厂商平台层。
模型、数据和训练配方留在 `up/`，设备、算子和厂商差异集中在 `up_platform/`。

**当前分支：`codex/up-platform-clean`。阶段性实现，尚未达到“两家完整适配”验收。**
NVIDIA 与沐曦已有真机记录；失败、缺口和复现步骤见 [验证报告](docs/platform/VALIDATION.md)。
旧 `main`、Release 和历史证据保持原样。本分支不包含旧数值修复或多随机种子任务。

[架构](docs/platform/README.md) · [完整接口](docs/platform/INTERFACES.md) · [接入厂商](docs/platform/ADDING_VENDOR.md) · [环境](docs/platform/ENVIRONMENT.md) · [验证](docs/platform/VALIDATION.md)

## 如何调用

```text
UP 模型 / 训练 / 独立推理
        ↓
PlatformBackend + 各算子类别父类
        ↓
ops/<类别>/nvidia.py 或 metax.py 的显式绑定
        ↓
对应厂商 PyTorch / 共享实现 / 厂商编译的内核
```

训练中的 eval 和 `train --evaluate` 使用训练适配；独立推理使用推理适配。
两者复用公共能力，不根据 `.eval()` 自动切换后端。

## 安装

先进入匹配显卡、驱动的厂商 PyTorch 环境。沐曦必须保留厂商版 PyTorch，不能用 NVIDIA wheel 覆盖。
详细版本与可选依赖见 [环境说明](docs/platform/ENVIRONMENT.md)。原版 `requirements.txt` 是历史清单，不是现代环境锁文件。

```bash
git clone --branch codex/up-platform-clean https://github.com/jingfengou/United-Perception-Hardware.git
cd United-Perception-Hardware
python -m pip install --no-deps -e .

# 仅在匹配的厂商环境中编译专用扩展，二选一：
UP_BUILD_BACKEND=nvidia python setup.py build_ext --inplace
UP_BUILD_BACKEND=metax python setup.py build_ext --inplace
```

普通安装不探测 GPU、不编译扩展。无 PyTorch/CUDA 时可以导入 `up_platform` 公共接口；运行 UP 模型仍需完整环境。

## 选择与运行

在原有 YAML 的 `runtime` 中加入：

```yaml
runtime:
  platform:
    backend: metax   # NVIDIA 使用 nvidia
    device: cuda    # 设备名可以相同，厂商身份不同
```

配置好原版配方的数据和权重路径后，继续使用 UP 命令：

```bash
python -m up train --config configs/cls/resnet/res18.yaml --launch pytorch --ng 1 \
  --opts runtime.platform.backend metax

# 训练验证路径，不切换成部署推理后端
python -m up train --evaluate --config configs/cls/resnet/res18.yaml --launch pytorch --ng 1 \
  --opts runtime.platform.backend metax

# 独立推理路径；替换为实际的配置、图片和权重
python -m up inference --config model.yaml --img_path images/ --ckpt model.pth \
  --launch pytorch --ng 1 --opts runtime.platform.backend metax
```

`--backend dist/linklink` 仍表示通信方式，不是显卡厂商。
`linklink` 真实 SDK 缺失，详见 [依赖缺口](docs/platform/LINKLINK.md)，不会用 DDP 或普通优化器冒充。

## 新增厂商

按 [接入指南](docs/platform/ADDING_VENDOR.md) 生成模板，实现设备后端、训练/推理差异和缺失算子。
兼容的厂商 PyTorch 操作可以显式复用；不兼容的操作必须提供完整实现及所需反向。
昇腾、寒武纪目前是逐项模板，不是已验证支持。

## 来源

基线为 UP `52aaeb9e`，架构参考 LightX2V `19c19614`。保留上游历史与 [Apache-2.0 LICENSE](LICENSE)。
模型和配方的原始用法见 [UP 官方文档](https://modeltc-up.readthedocs.io/en/latest/index.html)。
