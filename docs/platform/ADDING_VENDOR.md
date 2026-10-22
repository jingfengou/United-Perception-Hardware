# 新增厂商操作指南

目标：**只增加后端和注册，已有模型与训练配方不变。** 不需要先做一张猜测的“模型支持表”；按[完整接口清单](API_INDEX.md)复用兼容部分，逐项实现差异，再根据这些能力验证具体模型。

## 1. 先准备这些东西

| 材料 | 要求 |
| --- | --- |
| 实际设备、驱动、SDK、通信库 | 记录型号、版本、容器 digest 或可复建安装方法 |
| 厂商匹配 PyTorch / Python / TorchVision | 保留厂商分发，不安装 NVIDIA wheel 代替；没有共同版本时明确记录差异 |
| 设备 API | 设备名称、绑定、迁移、同步、RNG、显存、AMP、默认通信后端的真实入口 |
| 专用操作 | 厂商已有库或源码工具链，确认参数、布局、dtype、完整反向和许可 |
| 测试材料 | 固定输入/权重及输出和梯度参考；模型数据/配置用于端到端验收，不作为开发接口的前提 |

先运行厂商自己的 PyTorch 计算示例。若 SDK 不支持当前框架依赖组合，先在厂商环境内解决并记录，平台层不自动降级、换 Python 或修复 ABI。

## 2. 建一个独立插件

推荐独立目录或插件包，不给 UP 模型添加厂商分支：

```text
vendor_x_adapter/
  __init__.py      只注册后端元数据
  device.py        共享 DeviceBackend + SDK 差异
  training.py      训练工厂及训练专用注册
  inference.py     推理工厂及推理专用注册
  common_ops.py    可证明两侧共用的操作
  ops/            完整算子及必要 autograd
  csrc/           可选专用编译源码
```

`__init__.py` 模板（占位厂商名/设备名/SDK 必须替换为真实值）：

```python
from up_platform import BackendSpec, register_backend

register_backend(BackendSpec(
    name="vendor_x",
    device_type="vendor_device",
    device_factory="vendor_x_adapter.device:VendorDevice",
    training_factory="vendor_x_adapter.training:Backend",
    inference_factory="vendor_x_adapter.inference:Backend",
    shared_providers=("portable",),
    requirements={},  # 验证后填写真实 python / torch / SDK 发行包版本约束。
))
```

只支持训练时将 inference_factory 设为 None；只支持推理时将 training_factory 设为 None。初始化未交付用途会报错，不借用另一用途实现。不能把另一个厂商名（尤其 nvidia）当默认共享供应者。

## 3. 实现 device.py

```python
import importlib
from up_platform import DeviceBackend


class VendorDevice(DeviceBackend):
    name = "vendor_x"
    device_type = "vendor_device"
    communication_backend = "vendor_collective"

    def __init__(self, device=None):
        importlib.import_module("vendor_torch_sdk")
        super().__init__(device)
```

SDK 需要先注册 PyTorch 认识的 device_type。若 API 兼容 `torch.<device>.set_device/synchronize/...`，直接继承运行时，不重写 27 个公共成员。设备或 SDK 接口不同的部分按[运行时完整契约](API_INDEX.md#runtime-migration-31-members)覆盖。厂商也可使用 cuda 设备名，但 name 仍必须是自己的身份。

不要在 SDK 包导入时分配设备。health_check 才负责真实计算；configure 不以 SDK 可导入冒充设备可用。extension_modules 可声明 sparse/link/quantization/pruning 的厂商依赖，只有启用该功能才加载。

## 4. 分别提供训练和推理工厂

training.py：

```python
from up_platform.training.backend import TrainingBackend


class Backend(TrainingBackend):
    pass  # AMP / GradScaler / DDP 兼容时直接复用。
```

inference.py：

```python
from up_platform.inference.backend import InferenceBackend


class Backend(InferenceBackend):
    pass  # 默认普通 PyTorch 推理和已有 ONNX 导出。
```

仅在对应类中覆盖不同的 autocast、grad_scaler、wrap_model 或 export_onnx；两份模块不要互相导入。共同实现放 common/shared 或插件自己的 common_ops.py。训练验证仍由 training 执行，不需要另写一套部署模型，不应把参数替换成推理专用权重表示。

## 5. 只注册不兼容的完整操作

例如为已注册的昇腾后端补充**训练专用** RoIAlign：

```python
from up_platform import register_op

register_op(
    "ascend", "roi_align", "ascend_up_adapter.ops:roi_align",
    scope="training", backward=True,
    dtypes=("torch.float32",),
    variants={"pooled_mode": ("AVG", "MAX"), "aligned": (False, True)},
    conventions=("aligned",), dependencies=("torch_npu",),
)
```

实现接受 `features, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio, pooled_mode, aligned=False`，保持 NCHW、RoI 顺序、空输入、输出形状和完整 features 反向。不能只提供 forward 或把反向留给 NVIDIA 二进制。是否可复用厂商算子需实际核对坐标、采样、边界和梯度，不仅看名称相同。

若同一实现已确认可用于训练和推理，注册 scope=common；若二者实现不同，分别在 training.py、inference.py 注册同一操作的对应 scope。scope 必须显式填写。整个操作持有 autograd context，backward=True 不会替你生成梯度。

NMS 本来不可微，backward=False 即可；不要伪造梯度。稀疏层必须保留参数名称/身份/布局和 feature 图；厂商融合优化器须保持参数组与 master 状态。缺少能力明确报错，不增加空返回、常量输出或静默 CPU 代算。

默认复用的 portable 表示**同输入设备上的 PyTorch 共享实现**，不是“所有操作自动支持”或“自动搬到 CPU”。例如 CPU/国产后端未声明 RoIAlign 时仍会报错；spconv 没有供应者时 sparse 工厂仍会报错。

## 6. 可运行的单项覆盖示例

[完整示例目录](../../packages/up-platform/examples/vendor_plugin/)包含六个实际文件和安装元数据，复用全部已允许共享实现，仅覆盖 naive_nms。示例设备为 CPU，仅演示扩展协议，不是厂商适配或私有设备执行证明。

```bash
python -m pip install --no-deps --no-build-isolation -e ./packages/up-platform/examples/vendor_plugin
```

```python
import torch
from up_platform import configure_platform

ctx = configure_platform({
    "backend": "example_vendor", "device": "cpu",
    "plugins": ["up_example_vendor"],
}, purpose="training")
boxes = torch.tensor([[0., 0., 4., 4., .9], [0., 0., 4., 4., .8]])
assert ctx.ops.naive_nms(boxes, .5).tolist() == [0]
assert ctx.capabilities()["operations"]["naive_nms"]["owner"] == "example_vendor"
assert ctx.capabilities()["operations"]["gpu_iou_overlap"]["owner"] == "portable"
```

有真实 SDK 后仅替换 device.py 和那一个不兼容操作的实现体；兼容方法继续继承。另一种测试使用 privateuseone:1 命名的模拟设备确认路由与 CUDA 无关，但没有设备分配器就不能据此宣称硬件计算通过。

## 7. 配置启用，不改模型

```yaml
runtime:
  accelerator:
    backend: vendor_x
    plugins:
      - vendor_x_adapter
    numerics: native
    requirements: {}
  device: vendor_device
  fp16: false
```

安装平台包、UP 和厂商插件后继续使用原 `python -m up train/inference/to_onnx`。入口决定 purpose，原 `--backend dist/linklink` 含义不变。插件也可通过打包的 `up_platform.backends` entry point 注册，回调应只注册元数据；框架只加载被请求的名称。

厂商原生源码在自己的工具链下构建。只有 NVIDIA 使用平台包的显式 UP_BUILD_BACKEND=nvidia；新厂商不要在公共 setup.py 中加入自动 SDK 检测或编译。portable_v1 为独立实验配置，不要求为基本接入实现。

## 8. 交付与验收

1. 跑 `python -m pytest -q packages/up-platform/tests`，包括用途隔离、单项覆盖和缺失能力测试。
2. 按全量接口清单检查设备、AMP、通信、模型所需操作与可选功能；未实现项明确未支持。
3. 对每个用到的操作比对前向、所需梯度、状态更新、空输入、布局、dtype 和变体；不可微项不要求反向。
4. 验证 train → eval → train 后上下文/供应者不变、混合子模块状态恢复、参数/optimizer/scaler 不被验证更新，以及 checkpoint 恢复。
5. 在真实硬件分别记录训练、独立推理和导出；启用 AMP/多卡/SyncBN 时另测。固定种子不替代非确定性容差或精度统计。
6. 运行两个契约导出器 --check 和硬件边界检查；新操作必须补语义记录，不能只加注册。
7. 更新[能力矩阵](CAPABILITIES.md)和环境/证据。分别写“已注册、依赖可用、已执行、已验证”，结构迁移不能提高支持等级。

验收标准不是“注册成功”或“一次 forward 没报错”，而是新增后端与配置即可复用已有模型，并有对应范围的测试证据。
