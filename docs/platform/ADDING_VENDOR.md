# 新厂商接入

## 准备材料

1. 显卡、驱动、SDK、Python、PyTorch 和可选库的匹配版本及安装来源。
2. PyTorch 已注册的设备名，例如 cuda、npu、mlu 或 privateuseone；厂商名另行管理。
3. 按 [完整接口](INTERFACES.md) 逐项选择厂商 PyTorch、共享实现或自写内核，不需要先猜支持哪些模型。
4. 专用操作的前向、所需反向、状态、布局、导出实现，以及真实测试设备。

模型能否运行取决于它实际需要的接口。模板、导入或注册成功都不是完整适配。

## 生成全部待实现函数

```bash
# 纯模板：未实现项明确报错，不覆盖已有厂商
python tools/new_platform_vendor.py --vendor acme --device privateuseone --output acme_platform

# 已确认厂商 Torch 兼容公共 API 时，显式复用
python tools/new_platform_vendor.py --vendor acme --device privateuseone \
  --output acme_torch_platform --reuse-torch
```

生成 `device.py`、`training.py`、`inference.py` 和全部类别文件。
`--reuse-torch` 只复用基础类别与训练/推理的 Torch 入口；检测、点云、稀疏、通信、量化、剪枝等仍逐项填写。
它不是自动检测支持。生成器读取父类签名，新增接口会出现在下一次生成的模板中。

## 设备后端

在 `device.py` 中延迟导入 SDK，核对厂商 Torch 身份/版本，继承 `TorchDevice` 或实现 `DeviceBackend`。
检查设备绑定、迁移、同步、RNG、内存及卷积开关。同叫 cuda 也不能跳过身份校验。
共享卷积开关使用 `torch.backends.cudnn` 兼容 API；npu/mlu 等不兼容设备必须覆盖。
模板不提供任何厂商 SDK，不能改个名字就称为支持。

## 厂商 PyTorch 方式

与已实现的 `ops/matrix/metax.py` 相同：

```python
from up_platform.ops.matrix.base import MatrixOps
from up_platform.ops.matrix import torch_impl

class AcmeMatrixOps(MatrixOps):
    dependencies = ('torch',)
    linear = staticmethod(torch_impl.linear)
    matmul = staticmethod(torch_impl.matmul)
    mm = staticmethod(torch_impl.mm)
    bmm = staticmethod(torch_impl.bmm)
    addmm = staticmethod(torch_impl.addmm)
    einsum = staticmethod(torch_impl.einsum)
    Linear = staticmethod(torch_impl.Linear)
```

`torch_impl.py` 明确绑定到 `torch.nn.functional.linear`、`torch.mm` 等，执行当前安装的厂商 Torch，而非 NVIDIA 专用二进制。
如果只有 linear 不兼容，就只覆盖 linear。完整的“复用公共实现、覆盖一个操作”示例可执行：

```bash
python -m pytest tests/platform/test_vendor_template.py -q
```

测试用 CPU 是明确的测试替身，不是实际 GPU 验证。

## SDK / 自写内核方式

[sdk_linear.py](../../up_platform/templates/sdk_linear.py) 包含完整的一阶 autograd 接线：

- 前向保存输入/权重并调用 `kernels.linear_forward`。
- 反向调用 `kernels.linear_backward` 返回输入、权重与可选 bias 梯度。
- 供应者负责批量维度、dtype、布局、设备流、空输入和 `needs_input_grad`。
- 示例只声明一阶梯度；未提供 ONNX 适配时明确报错。满足更完整契约需继续实现，不能冒充完整 Linear。
- CPU gradcheck 只证明接线，真实内核还需 GPU 前向/梯度/状态对照。

不能用同名操作冒充等价实现。RoIAlign、DeformConv 等的坐标、采样、offset、空输入与反向必须匹配。

## 注册和用途

插件已调用 `register_backend('acme', build)`；`build(config, purpose)` 显式组合类别。
把插件放到 Python 导入路径后配置：

```yaml
runtime:
  platform:
    backend: acme
    device: privateuseone
    plugins: [acme_torch_platform]
    requirements: [matrix.linear, convolution.Conv2d]
```

只交付一种用途时，另一用途工厂应明确拒绝，不自动借用。
训练内 eval 使用 `TrainingBackend.validation`。融合优化器必须保留真实状态/master weights，无 SDK 时不能换普通 AdamW。

## 没有 PyTorch 适配时

仅填几个算子不够，UP 仍依赖 PyTorch 张量、autograd、模块和基础运算。先交付设备桥接：

| 层面 | 必须交付 |
| --- | --- |
| 注册 | Dispatch key、设备名、上下文、device guard |
| 存储 | 分配器、storage、引用计数、视图生命周期和释放 |
| 传输 | CPU/设备间及设备间拷贝、异步和 pinned memory |
| 执行 | 流、事件、同步、错误传播、RNG、多设备 |
| 基础派发 | 构造、索引、reshape、转置、拷贝、算术和归约 |
| 自动求导 | 反向派发、保存张量、in-place/version、梯度累积 |
| 模块/保存 | Parameter、state_dict、优化器状态、序列化和恢复 |
| 精度/多卡 | autocast、scaler、通信后端和同步语义 |

桥接完成后再用本项目父类、模板和 SDK 示例接入。
本分支没有重写一个脱离 PyTorch 的框架，不承诺缺少桥接时 UP 可运行。

## 验收

检查父类/模板/绑定/文档一致，缺失能力明确报错。
先测 dtype、布局、边界、前向、梯度、状态和导出，再测模型训练、训练内 eval、恢复、独立推理及多卡。
保留失败和同源重复；不放宽容差、不做静默 CPU 代算、不挑最好一次。
新增厂商只增加后端和注册，已有模型算法与配方不变。
