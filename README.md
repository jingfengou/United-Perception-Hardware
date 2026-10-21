# United Perception：多厂商硬件适配版

本仓库基于 [ModelTC/United-Perception](https://github.com/ModelTC/United-Perception)，在原有模型、训练和推理框架上增加了**硬件中间层、厂商选择和算子路由**。

希望解决的问题是：换一家的显卡时，普通 PyTorch 算法继续复用，设备接口和不兼容的专用算子集中在适配层处理，不再逐个修改模型里的硬件调用。

这是保留原项目 Git 历史的独立派生仓库。上游基线为 `52aaeb9ed27a490820696d7754d19f98cd6ecd11`，原项目的 [Apache-2.0 LICENSE](LICENSE) 和代码归属保留。主要新增代码位于 [`up/accelerator/`](up/accelerator/)。

只想运行模型，阅读第 2、3、4 节；准备增加厂商适配，继续阅读第 5 节。需要逐项核对适配函数，直接看[完整接口定义](docs/hardware/INTERFACE.md)和[函数签名索引](docs/hardware/API_INDEX.md)。

原版与 NVIDIA 路由版的模型训练对照见[五模型多随机种子报告](docs/hardware/MULTISEED_ACCURACY.md)，覆盖 Faster R-CNN、YOLOv5-S、ResNet18、外部官方 DETR 适配器和 ViT-B/16。报告分别列出完整预算、最终精度、逐步数值一致性、权重恢复与未覆盖范围，不把合成输入检查当作完整数据集精度验收。

## 1. 增加了什么功能

```text
原有模型、损失函数、训练和推理流程
                  |
          统一硬件接口和路由
                  |
       共享 PyTorch 实现 / 厂商适配实现
                  |
           所选厂商的设备与 SDK
```

| 功能 | 具体作用 |
| --- | --- |
| 选择显卡厂商 | 在配置文件中选择 NVIDIA、昇腾、寒武纪、沐曦或新增的后端 |
| 统一设备操作 | 统一设备选择、模型与数据迁移、同步、随机状态和显存查询 |
| 统一混合精度 | 统一 autocast、GradScaler 等入口，厂商可覆盖不兼容部分 |
| 专用算子路由 | NMS、RoIAlign、DeformConv、Focal Loss、3D IoU、点云池化等通过统一入口调用 |
| 专用层和通信隔离 | SyncBN、稀疏卷积及其张量操作、融合优化器、分布式通信通过适配层访问 |
| 复用与局部覆盖 | 兼容的实现直接复用，只给不兼容的操作增加厂商实现 |
| 保留原有使用方式 | 保留原算子导入路径、`.apply()` 入口和模型权重参数层级 |
| 防止重新耦合 | 静态检查阻止上层代码重新直接使用厂商 SDK、专用设备接口和底层扩展 |

普通、设备无关的 PyTorch 计算继续直接使用，不包装整个 PyTorch。

**当前完成的是硬件隔离和路由，以及 NVIDIA 的首轮验证。三个国产后端已有运行时入口和部分共享实现，但专用算子尚未全部补齐，也没有完成对应真机验收。选择后端不等于该厂商已经能运行所有模型。**

| 硬件 | 配置中的 `backend` | 设备名 | 当前状态 |
| --- | --- | --- | --- |
| NVIDIA A800 | `nvidia` | `cuda` | 五模型真实数据训练/推理/恢复已完成；4 个模型各 5 对一致，Faster R-CNN 10 对平均 AP 等价但非逐步一致 |
| 昇腾 910B | `ascend` | `npu` | 运行时入口和共享实现已接入，专用算子及真机验证待完成 |
| 寒武纪 MLU590 | `cambricon` | `mlu` | 同上 |
| 沐曦 C500 | `metax` | `cuda` | 同上；不会因为设备名为 `cuda` 就自动使用 NVIDIA 二进制 |
| CPU | `cpu` | `cpu` | 用于公共接口和共享实现测试，不代表所有模型都有 CPU 实现 |

逐项情况见[能力矩阵](docs/hardware/CAPABILITIES.md)。

## 2. 框架自己有没有环境要求

**有。能使用的环境必须同时满足框架、显卡厂商和目标模型三方面的要求。**

| 要求来自哪里 | 需要核对什么 |
| --- | --- |
| 框架本身 | Python、PyTorch、匹配的 TorchVision、NumPy、ONNX、图像处理及数据集依赖 |
| 显卡厂商 | 驱动、SDK、厂商适配的 PyTorch，以及它们支持的 Python 版本组合 |
| 目标模型 | 是否还需要 RoIAlign、3D IoU、稀疏卷积等专用实现，及其前向、反向和精度支持 |

因此，使用某家的 PyTorch 容器后，还要确认本框架及目标模型在该环境中能运行。中间层不会自动解决 Python 版本冲突、第三方库 API 差异或二进制 ABI 不兼容。

### 2.1 当前有哪些已验证的环境

以下是 **2026-10-20 本轮实际执行过的环境**，不是对其他版本的支持保证：

| 范围 | 环境 | 实际验证 |
| --- | --- | --- |
| NVIDIA 算子和代表模型 | Linux、Python `3.12.3`、PyTorch `2.8.0a0+5228986c39.nv25.06`、CUDA `12.9` | 32 个算子对比样例、4 个代表模型、AMP 和双卡通信 |
| 与上述 PyTorch 配套的 TorchVision | `0.22.0a0+95f10a4e` | 使用服务器原有配套分发，不是建议与任意官方 PyTorch wheel 混装 |
| 部分普通依赖 | NumPy `1.26.4`、ONNX `1.17.0` 等 | 清单见下方，完整版本快照见验证产物 |
| 公共接口 CPU 测试 | 上述 Linux/PyTorch 环境，隐藏 CUDA 设备 | 首轮 52 项；含完整接口、成对训练/统计协议与新增模型配置检查后 117 项通过 |
| 无 CUDA 公共包构建 | Windows、Python `3.13.11`、未安装 PyTorch | wheel 和 sdist 构建通过；没有在此环境验证模型运行 |
| 昇腾、寒武纪、沐曦 | 尚未建立经真机确认的完整环境组合 | 不能据 NVIDIA 的结果推断这三家的版本支持 |

CPU CI 还配置了 Python 3.10/3.12、CPU PyTorch 2.6 的接口检查；云端执行状态见仓库的 Actions 页面。接口检查通过不等于完整模型和全部依赖都通过。

本仓库有三类环境资料：

- [`requirements.txt`](requirements.txt)：原项目的历史依赖，含 SciPy 1.1.0、ONNX <= 1.6.0 等旧约束，保留用于追溯。
- [`requirements/validated-a800.txt`](requirements/validated-a800.txt)：从首轮 A800 环境提取的应用和测试依赖子集，供该环境复验；保留配套的 PyTorch/TorchVision，COCO 数据链路另按任务报告处理。
- [`docs/hardware/validation/packages.json`](docs/hardware/validation/packages.json)：实际验证环境中可见的包名和版本快照，包含继承的系统包，不是跨机器通用的安装锁。

首轮环境清单并不保证真实数据链路也兼容：本次检测训练发现 NVIDIA 增强版 COCO 库需要任务内的标准 `pycocotools==2.0.10` 覆盖，旧官方预训练权重需要保值转换序列化格式。具体操作见[真实数据训练报告](docs/hardware/TRAINING_ACCURACY.md)，不替换系统 PyTorch/TorchVision，也不全局关闭权重安全加载。

Python 3.12/NumPy 1.26 下的回归启动器还给两边相同地恢复了旧代码依赖的 `collections.Iterable`、`np.float`、`np.int` 别名，保持原 dtype。这是已记录的共同兼容前置，不代表原始 CLI 对任意现代依赖组合自动兼容；直接使用旧入口时也需处理这些旧 API，或选择经验证的兼容环境。测试启动器与具体命令见[五模型报告](docs/hardware/MULTISEED_ACCURACY.md)。

`setup.py` 中的 `python_requires >= 3.6` 是继承的包元数据，**不能据此理解为 Python 3.6 以上任意版本都能运行全部功能**。实际运行支持以对应环境的测试记录为准。

### 2.2 厂商只支持某个 PyTorch/Python 版本，怎么办

例如，某厂商只提供了某一套 Python + PyTorch + SDK 组合，应先保留这套组合，再验证框架的依赖和模型。若存在冲突，在该厂商的环境配置或适配层中解决并复验，不把它强制换成本轮 NVIDIA 使用的版本。

不同厂商可以使用不同的容器或虚拟环境。确定可用组合后，记录镜像版本、驱动/SDK、Python/PyTorch/TorchVision、普通依赖和算子实现版本，为该厂商单独维护环境清单。

也可以在配置中加入启动时的包版本检查。例如，下面仅演示**本轮 NVIDIA 环境**的约束写法：

```yaml
runtime:
  accelerator:
    backend: nvidia
    requirements:
      python: ">=3.12,<3.13"
      torch: ">=2.8.0a0,<2.9"
```

确定国产环境的可用版本后，再填写它自己的约束。版本检查只会校验和报错，不会自动安装或切换环境。

## 3. 如何安装

以下命令在目标 Linux 机器或厂商容器内执行。安装顺序是：**先准备厂商运行环境，再补框架依赖，最后安装本仓库。**

### 3.1 先准备运行环境

1. 安装与显卡匹配的驱动和厂商 SDK，或进入厂商提供的适用容器。
2. 运行厂商自带的 PyTorch 示例，确认能在目标卡上创建张量和计算。
3. 保留厂商配套的 PyTorch 和 TorchVision。昇腾后端会加载 `torch_npu`，寒武纪后端会加载 `torch_mlu`，沐曦后端使用厂商的 PyTorch 分发。
4. 核对框架所需普通依赖，以及目标模型需要的额外库。

### 3.2 安装公共框架

```bash
git clone https://github.com/jingfengou/United-Perception-Hardware.git
cd United-Perception-Hardware
```

这是公共仓库，可直接克隆。在代码根目录补齐普通依赖：复验首轮 A800 环境时，可在保留其配套 PyTorch/TorchVision 的前提下使用下列子集；真实数据检测还需按训练报告准备兼容的 COCO 库和权重格式：

```bash
python -m pip install -r requirements/validated-a800.txt
```

此文件不是三家国产硬件的安装方案，也不能仅靠它重建原 NVIDIA 容器。国产环境需要与厂商 SDK 要求协调依赖版本，再形成自己的安装清单。

依赖准备好后，安装公共框架：

```bash
python -m pip install setuptools wheel packaging
UP_BUILD_BACKEND=none python -m pip install -e . --no-deps --no-build-isolation
```

`--no-deps` 表示保留前面准备的依赖环境；这条命令本身不会把完整训练依赖安装齐。

`UP_BUILD_BACKEND=none` 关闭本项目 NVIDIA 扩展的编译，适用于先安装公共部分以及国产后端接入。厂商专用扩展在对应的 SDK 环境中另外构建、安装和注册。

### 3.3 NVIDIA 额外编译原有扩展

NVIDIA 检测和点云模型使用原生扩展时，还需安装与 PyTorch 匹配的 CUDA toolkit 和编译器，然后编译：

```bash
# 8.0 是本轮 A800 的编译架构；其他卡应使用相应架构。
UP_BUILD_BACKEND=nvidia CUDA_HOME=/usr/local/cuda \
  TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2 \
  python setup.py build_ext --inplace
```

原 `easy_setup.sh` 保留了上游历史编译配置。新环境使用本节的构建入口，并核对自己的工具链。

## 4. 如何选择显卡并使用

### 4.1 修改模型配置

以沐曦运行分类模型为例，编辑模型 YAML 中**已有的** `runtime` 段，保留其他配置：

```yaml
runtime:
  task_names: cls
  accelerator:
    backend: metax
  device: cuda
  fp16: false
```

不要在同一个 YAML 中重复添加两个 `runtime` 段。其他硬件对应关系为：

| 显卡 | `runtime.accelerator.backend` | `runtime.device` |
| --- | --- | --- |
| NVIDIA | `nvidia` | `cuda` |
| 昇腾 | `ascend` | `npu` |
| 寒武纪 | `cambricon` | `mlu` |
| 沐曦 | `metax` | `cuda` |

设备名不能单独标识厂商。国产环境应显式选择厂商；`device` 也可省略，采用该后端的默认设备。

### 4.2 先检查设备是否可用

下面以沐曦后端为例；其他厂商替换 `metax`：

```bash
UP_ACCELERATOR_BACKEND=metax python -c \
  "from up.accelerator import get_accelerator; a=get_accelerator(); print(a.name, a.device); print(a.health_check())"
```

这个检查只验证基础设备计算和同步，不代表整个模型已经适配。配置中的厂商选择优先于环境变量。

### 4.3 单卡训练、评估和推理

先准备模型配置、数据和需要的权重。可从 [`configs/cls/resnet/res18.yaml`](configs/cls/resnet/res18.yaml) 复制配置，修改数据目录、标注文件、类别数、batch size 和上一节的后端选项。

```bash
# 单卡训练。your_model.yaml 替换为准备好的实际配置文件。
python -m up train --config your_model.yaml \
  --launch pytorch --ng 1 --backend dist

# 评估。先在配置的 saver.resume_model 中填写权重路径。
python -m up train --evaluate --config your_model.yaml \
  --launch pytorch --ng 1 --backend dist
```

使用指定评估权重时，检查 `saver.auto_resume` 的优先级；需要明确指定文件时设为 `false`。例如编辑原有 `saver` 段，保留目录设置并加入：

```yaml
saver:
  save_dir: checkpoints/my_model
  results_dir: results_dir/my_model
  auto_resume: false
  resume_model: /path/to/checkpoint.pth
```

支持原检测推理接口的模型，还可以使用：

```bash
python -m up inference --config your_detection_model.yaml \
  --ckpt /path/to/checkpoint.pth --img_path /path/to/images \
  --vis_dir vis_dir --launch pytorch --ng 1 --backend dist
```

这里的 `--backend dist` 表示使用 PyTorch 通信框架；它与 YAML 中选择厂商的 `runtime.accelerator.backend` 是两个不同参数。

首次接入先使用单卡、FP32。跑通后再验证 AMP 和多卡；多卡可将 `--ng` 改为卡数并使用 `--fork-method spawn`。三个国产后端的多卡通信仍需真机核验。

### 4.4 遇到“缺少算子”，应该怎么做

例如报错包含：

```text
backend='ascend', device='npu', operation='roi_align': no implementation registered
```

这表示框架还没有为该后端接上这个算子，需要在适配层补实现，不能靠换配置自动解决。

- 兼容的常规 PyTorch 运算及已注册共享实现继续复用。
- 模型用了 RoIAlign、DeformConv、某些 Focal Loss、3D IoU 或点云池化时，要确认该厂商已有相应实现。
- 使用稀疏卷积、linklink、量化或剪枝工具时，还要提供对应供应者。
- 不增加自动 CPU 代算或无实际计算的占位实现。

**使用者准备显卡环境、模型配置、数据和权重；适配开发者负责补齐该模型缺失的操作并验收。** 补齐后，其他经过相同接口的模型可以复用。

## 5. 如何增加一个厂商的适配

### 5.1 需要准备哪些东西

| 材料 | 用途 |
| --- | --- |
| 实际硬件和可访问的运行环境 | 验证真实计算、反向、显存及通信，不以 mock 代替真机 |
| 驱动、SDK、容器或安装包版本 | 明确目标部署环境，能够重复创建 |
| Python、PyTorch、TorchVision 版本及 SDK 扩展 | 确认框架与厂商共同可用的版本组合 |
| 设备接口说明 | 确认设备名，以及设备绑定、同步、随机状态、内存、AMP 和通信入口 |
| 目标模型、配置、数据与权重 | 确定实际需要适配哪些操作 |
| 厂商算子库或编译工具链 | 复用已有算子，或构建缺失的专用实现 |
| 数值和梯度参考结果 | 验证输出、索引、精度、反向与模型权重兼容 |

先适配目标模型需要的操作，不要求第一天就实现仓库的所有专用算子。

### 5.2 已有厂商：补算子即可

昇腾、寒武纪、沐曦的后端名已经注册。为它们补实现时，在可导入的插件模块中注册缺失算子即可，无需再创建整套后端。

例如，为昇腾增加 RoIAlign 的**注册模板**：

```python
# 文件：ascend_up_adapter/__init__.py
from up.accelerator import register_op

register_op(
    "ascend",
    "roi_align",
    "ascend_up_adapter.roi_align:roi_align",
    backward=True,
)
```

还必须在 `ascend_up_adapter/roi_align.py` 中提供真正的 `roi_align` 函数。它应接收原接口的参数：

```python
roi_align(features, rois, pooled_h, pooled_w,
          spatial_scale, sampling_ratio, pooled_mode)
```

函数可以调用厂商已有算子，也可以使用可微的 PyTorch 组合实现；需要处理输入/输出、坐标与对齐约定、支持的模式、dtype，以及训练所需的反向。

`backward=True` 只是在声明能力，不会自动生成梯度。若使用自定义 autograd Function，应注册包含完整前向、反向和必要 context 的入口。

安装该插件，或把它放到 Python 可导入的路径后，在配置中启用：

```yaml
runtime:
  accelerator:
    backend: ascend
    plugins:
      - ascend_up_adapter
  device: npu
```

这个例子中的适配函数需要实际开发，注册一条模块路径不会自动产生实现。

### 5.3 全新厂商：增加运行时后端，再覆盖差异

在一个独立插件模块中注册后端。下面是模板，SDK 模块、设备名和通信名需要替换为厂商实际提供的接口：

```python
# 文件：vendor_x_up/__init__.py
import importlib
from up.accelerator import TorchAccelerator, register_backend

@register_backend("vendor_x")
class VendorXAccelerator(TorchAccelerator):
    name = "vendor_x"
    device_type = "vendor_device"
    communication_backend = "vendor_collective"
    op_fallbacks = ("portable",)

    def __init__(self, device=None):
        importlib.import_module("vendor_torch_extension")
        super().__init__(device)
```

SDK 需先注册其实际 PyTorch 设备。若设备迁移、同步、内存或 AMP 的接口与共享运行时不同，只覆盖对应方法。厂商也可以使用 `cuda` 等已有设备名，厂商身份仍由 `backend` 单独管理。

然后按上一节的方法注册不兼容的算子，配置选择：

```yaml
runtime:
  accelerator:
    backend: vendor_x
    plugins:
      - vendor_x_up
  device: vendor_device
```

共享实现通过 `op_fallbacks` 显式复用；当前后端注册的同名操作优先。不能只凭设备名相同就默认它能使用 NVIDIA 二进制，编译和语义兼容需单独验证。

仓库还提供了可运行的 [`example_plugin.py`](up/accelerator/example_plugin.py)：它在 CPU 上复用共享实现，只覆盖一个 NMS，用于展示插件协议。它不是国产显卡的模拟实现。

### 5.4 专用层、通信和编译也要接入

| 不兼容部分 | 在哪里适配 |
| --- | --- |
| 设备、同步、RNG、内存、AMP | 后端运行时的方法 |
| NMS、RoIAlign、DeformConv、Focal Loss、点云操作 | `register_op(厂商名, 操作名, 实现)` |
| SyncBN | `sync_batch_norm_forward` / `sync_batch_norm` 注册项 |
| 稀疏卷积及张量表示 | `sparse.*` 工厂、特征读写、稠密化及容器执行接口 |
| DDP 与集合通信 | `distributed.*` 注册项和运行时的通信后端配置 |
| 融合优化器及可选工具库 | 对应操作注册项或延迟供应者 |
| 厂商二进制 | 在匹配的厂商 SDK 环境内独立构建、安装，再由后端延迟加载 |

层实现应保持原权重参数的名称和布局，避免额外嵌套模块导致 checkpoint key 改变。缺失能力应说明厂商、操作和原因。

### 5.5 怎样才算适配完成

1. 记录并固定厂商环境组合，通过基础设备检查。
2. 逐项检查目标模型需要的算子，比较输出和适用的梯度，记录精度与变体。
3. 运行模型短程训练、推理、权重恢复；需要多卡时再验证通信、DDP 和 SyncBN。
4. 运行静态检查，确保没有给上层模型和 Runner 增加厂商分支。
5. 更新能力矩阵、环境清单和真机验证记录。

**新增厂商应通过增加后端/插件和注册配置完成；已有模型和训练流程保持不变。**

## 6. 检查、验证与代码位置

在已经安装 PyTorch 的测试环境中：

```bash
python -m pip install pytest packaging setuptools wheel PyYAML numpy
python tools/check_hardware_boundaries.py
python tools/export_accelerator_contract.py --check
python -m pytest -q tests/test_accelerator.py tests/test_hardware_contract.py tests/test_training_accuracy.py tests/test_multiseed_accuracy.py
```

`up.accelerator.capabilities()` 可查询已注册的操作、供应者和声明信息。查询结果是注册清单，不会证明对应 SDK 已安装或真机已经通过。

完整测试包含多种子统计测试，需要与当前 Python 匹配的 SciPy（本轮为 `1.15.3`）。2026-10-21 的 A800 成对回归结果如下；完整预设预算不等于完整 ImageNet/COCO 训练，DETR 为外部官方单类适配器：

| 模型 | 配对种子 | 原版 / NVIDIA 路由版平均指标 | 结论 |
| --- | --- | --- | --- |
| Faster R-CNN R50-FPN | 10 | AP 60.4172 / 60.4419 | 平均 AP 通过预设 +/-1 点等价检验，非逐步一致 |
| YOLOv5-S SiLU | 5 | AP 26.0174 / 26.0174 | 逐步 loss、最终模型状态、指标相同 |
| ResNet18 | 5 | Top-1 66.2200 / 66.2200 | 同上 |
| 外部 DETR R50 | 5 | AP 77.6032 / 77.6032 | 同上 |
| ViT-B/16，64px | 5 | Top-1 22.3400 / 22.3400 | 同上；仅小数据回归，不是收敛验收 |

数据为 CIFAR-10 10,000/2,000 与 PennFudan 120/50，均值、标准差、每个种子和复现步骤见[五模型报告](docs/hardware/MULTISEED_ACCURACY.md)。不把不同预训练/预算下的绝对指标作为跨模型排行榜。

| 内容 | 位置 |
| --- | --- |
| 厂商选择和统一入口 | [`up/accelerator/__init__.py`](up/accelerator/__init__.py) |
| 后端与算子注册 | [`up/accelerator/registry.py`](up/accelerator/registry.py) |
| 共享运行时与设备方法 | [`up/accelerator/runtime.py`](up/accelerator/runtime.py) |
| 已有厂商运行时 | [`up/accelerator/backends/torch_backends.py`](up/accelerator/backends/torch_backends.py) |
| 算子、层、稀疏操作和通信 | [`up/accelerator/`](up/accelerator/) |
| 原有算子兼容入口 | [`up/extensions/python/`](up/extensions/python/) |
| 262 条上游硬件依赖记录 | [`upstream_inventory.json`](docs/hardware/upstream_inventory.json) |
| 详细接口与接入说明 | [接口说明](docs/hardware/README.md)、[新增厂商说明](docs/hardware/ADDING_BACKENDS.md) |
| 完整适配函数定义 | [中文接口契约](docs/hardware/INTERFACE.md)、[全部函数签名](docs/hardware/API_INDEX.md)、[机器可读清单](docs/hardware/interface_contract.json) |
| 真实数据训练与精度变化 | [五模型多种子报告](docs/hardware/MULTISEED_ACCURACY.md)；[早期短程对照](docs/hardware/TRAINING_ACCURACY.md)与原始差异记录继续保留 |
| 能力和实际测试范围 | [能力矩阵](docs/hardware/CAPABILITIES.md)、[验证报告](docs/hardware/VALIDATION.md) |
| 受测快照与 Git 提交的内容核对 | [`git_source_match.json`](docs/hardware/validation/git_source_match.json)，行尾差异单独记录 |
| 上游完整任务文档 | [UP 官方文档](https://modeltc-up.readthedocs.io/en/latest/index.html) |

首轮 NVIDIA 检查是有界数值与合成输入验证，最新真实数据完整 Runner 对照已完成并另见[五模型报告](docs/hardware/MULTISEED_ACCURACY.md)，区分逐步数值一致性与平均 AP 等价。它们不代表完整 ImageNet/COCO 收敛或性能验收。当前上游快照没有 DETR，外部官方适配器单列，不冒充 UP 自有模型。linklink、spconv、MQBench、MSBench 的真实执行及三家国产硬件验证仍未完成。

检测的[首次差异定位](docs/hardware/validation/accuracy_20261021/divergence/README.md)进一步确认：首步前向、输入和随机状态相同，差异从原版已有的 RoIAlign 原生反向开始；诊断中固定其返回梯度后，两版连续 3 步完全对齐。该回放不是正式训练实现或最终 AP 验收。
