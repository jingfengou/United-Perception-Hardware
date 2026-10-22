# 新增厂商操作指南

**一个设备后端 + 可选的训练/推理差异 + 缺失算子。** 已兼容的部分直接继承，不用写空的 `TrainingBackend` / `InferenceBackend`。模型、损失、训练配方均不改。

## 1. 最小接入：只实现设备

先准备真实设备、驱动、SDK、通信库和匹配的 Python / PyTorch / TorchVision，运行厂商自己的 PyTorch 示例。记录环境版本，不安装 NVIDIA wheel 覆盖厂商发行版。

```text
vendor_x_adapter/
  __init__.py       注册设备与需要覆盖的算子
  device.py         设备后端、SDK 差异
  ops.py            可选：缺失或不兼容算子
```

`device.py`（占位 SDK 和设备名替换成厂商实际值）：

```python
import importlib
from up_platform import DeviceBackend


class VendorDevice(DeviceBackend):
    name = "vendor_x"
    device_type = "vendor_device"
    communication_backend = "vendor_collective"
    # 可选：实际验证过的环境约束；省略表示不额外约束，不代表全部版本已验证。
    requirements = {"python": ">=3.10,<3.13"}

    def __init__(self, device=None):
        importlib.import_module("vendor_torch_sdk")
        super().__init__(device)
```

SDK 应先注册 PyTorch 认识的设备命名空间。兼容 `torch.<device>` 的绑定、迁移、同步、RNG、显存接口直接继承；不一致的方法才覆盖。CUDA-shaped 厂商可以使用 `device_type="cuda"`，但 `name` 仍是自己的身份，不得借此加载 NVIDIA 二进制。

`__init__.py`：

```python
from up_platform import register_backend
from .device import VendorDevice

register_backend(VendorDevice)
```

到这里已默认使用公共训练和推理实现：AMP / GradScaler / DDP、普通 PyTorch 推理和已有 ONNX 导出。**不需要另建 training.py / inference.py。** 这只是复用接口，不代表相关功能或所有模型已在真机验证。设备检查必须实际执行；注册成功和 SDK 可导入不是验收。

默认共享操作供应者为 `("portable",)`，表示在输入所在设备执行的 PyTorch 实现，不是自动 CPU 回退，也不提供全部 73 项操作。可通过设备类的 `op_fallbacks` 显式声明共享供应者；不能默认填入另一家厂商。

## 2. 有差异再增加代码

### 训练 / 推理差异

例如厂商只需不同的 GradScaler，再新增 `training.py`：

```python
from up_platform.training.backend import TrainingBackend


class Backend(TrainingBackend):
    def grad_scaler(self, **kwargs):
        from vendor_torch_sdk.amp import GradScaler
        return GradScaler(**kwargs)
```

注册改成：

```python
register_backend(VendorDevice, training="vendor_x_adapter.training:Backend")
```

推理继续复用公共实现；若确有推理差异，再设置 `inference="vendor_x_adapter.inference:Backend"`。字符串工厂只在选择该用途时导入，执行工厂接受 `context`；训练和推理模块不能互相导入。

| 注册写法 | 含义 |
| --- | --- |
| `register_backend(VendorDevice)` | 两侧均复用公共实现 |
| `register_backend(VendorDevice, training="...:Backend")` | 只替换训练侧，推理侧仍复用 |
| `register_backend(VendorDevice, inference=None)` | 明确不提供推理，选择推理时报错 |
| `@register_backend` / `@register_backend(training="...:Backend")` | 同样支持设备类装饰器 |

省略参数与显式 `None` 不同。旧 `BackendSpec` 完整规格、旧插件注册及工厂地址仍兼容；其中 `training_factory=None` / `inference_factory=None` 继续表示未支持，不改变旧语义。只有需要设备工厂本身也延迟导入等高级场景时，才需要手写 `BackendSpec`。

训练中的 eval 和 `train --evaluate` 留在训练路径，使用当前模型；临时 eval/no_grad 后恢复状态，不需要一套部署模型，也不切换推理 SDK。

### 缺失算子

只有公共实现不兼容或未提供时，才注册完整操作。例如在 `__init__.py` 增加训练专用 RoIAlign：

```python
from up_platform import register_op

register_op(
    "vendor_x", "roi_align", "vendor_x_adapter.ops:roi_align",
    scope="training", backward=True,
    dtypes=("torch.float32",),
    variants={"pooled_mode": ("AVG", "MAX"), "aligned": (False, True)},
    conventions=("aligned",), dependencies=("vendor_torch_sdk",),
)
```

实现必须符合[完整接口](INTERFACES.md)与[逐项契约](API_INDEX.md)：参数、返回值、设备、布局、空输入、状态、梯度、导出均不能只看同名函数。RoIAlign 的 `features` 需要完整反向；NMS 本来不可微则声明 `backward=False`。注册 `backward=True` 不会自动生成梯度。

明确两侧可共用时使用 `scope="common"`；只有一侧可用时填写 `training` 或 `inference`。路由只查当前用途和 common，不跨用另一用途。共同实现只有一份；不要复制整套厂商后端。专用扩展和可选 `ops/`、`csrc/` 只在确有实现时增加，由对应厂商工具链构建。

稀疏层保持参数名称、身份及 feature 图；融合优化器保持参数组和 master 状态。缺失能力明确报错，不返回常量、不伪造梯度、不新增静默 CPU 代算。数值实验 profile 不是基本接入义务。

## 3. 运行示例与配置

[可安装示例](../../packages/up-platform/examples/vendor_plugin/)仅含三个 Python 文件：设备、注册、一个 NMS 覆盖。其他操作和两侧执行接口均复用公共实现。它使用 CPU 演示，不是真实厂商验收。

```bash
python -m pip install --no-deps --no-build-isolation -e ./packages/up-platform/examples/vendor_plugin
```

```python
import torch
from up_platform import configure_platform

ctx = configure_platform({
    "backend": "example_vendor", "device": "cpu",
    "plugins": ["up_example_vendor"],
}, purpose="training")  # 独立推理时显式选 inference。
boxes = torch.tensor([[0., 0., 4., 4., .9], [0., 0., 4., 4., .8]])
assert ctx.ops.naive_nms(boxes, .5).tolist() == [0]
assert ctx.require("naive_nms")["owner"] == "example_vendor"
assert ctx.require("gpu_iou_overlap")["owner"] == "portable"
```

真实厂商安装平台、UP 和插件后，在模型配置已有的 runtime 段启用：

```yaml
runtime:
  accelerator:
    backend: vendor_x
    plugins: [vendor_x_adapter]
    numerics: native
  device: vendor_device
  fp16: false
```

继续使用 `python -m up train/inference/to_onnx`，入口决定用途。原 `--backend dist/linklink` 仍是通信策略。插件也可通过 `up_platform.backends` entry point 注册，框架只加载请求的名称。普通包导入不初始化设备；SDK 导入放设备构造或算子调用中。

## 4. 交付与验收

1. 按[接口索引](API_INDEX.md)列出复用项、覆盖项和明确未支持项，不需要先猜一张模型支持表。
2. 跑 `python -m pytest -q packages/up-platform/tests`；新增操作补参数、dtype、变体、空输入、前向、所需梯度和状态测试，不放宽既有容差。
3. 真机分别测训练、训练 eval、恢复和独立推理；验证不更新权重、optimizer 或 scaler，退出后恢复模块状态；AMP、多卡、SyncBN、导出按启用范围另测。
4. 跑两个契约导出器 `--check`、硬件边界和项目检查，命令见[平台手册](README.md#4-文档与检查)。新增语义操作需补机器契约，不只增加注册。
5. 更新[能力矩阵](CAPABILITIES.md)、实际环境和证据，区分已注册、依赖可用、已执行和已验证，不用 CPU/模拟设备测试冒充真机通过。

验收标准：增加设备后端、实际差异与注册配置即可接入，已有模型和训练配方保持不变。
