# UP 平台包

`up-platform` 与 UP 在同一仓库，独立安装、独立测试，Python 导入名为 `up_platform`。按 LightX2V 风格组织设备后端、算子注册与分用途工厂，硬件算子统一按“类别 → 厂商”放置，不增加新模型算法或推理引擎。[当前总设计](../architecture/PLATFORM.md)与[逐文件迁移清单](../architecture/platform-layout.json)是现行架构；历史 GPU 结果仍只对各报告的冻结版本成立。

## 1. 调用关系

```text
UP 模型 / loss / 数据 / 配方 / Runner / CLI
                    |
             up.accelerator 旧门面
                    |
         up_platform.PlatformContext
           purpose 由入口固定
          /                  \
     training               inference
   AMP / scaler / DDP     autocast / ONNX 导出
   训练操作与层            推理操作与层
          \                  /
       registry_factory + context
        base 设备 / common 通信
       ops/<类别>/<厂商> 或 common.py
                    |
          已选厂商设备与 SDK
```

普通设备无关 PyTorch 计算保持直接调用。平台不读 YAML、CLI、`up.flags`、模型配置，不导入 `up`；学习率、优化器超参数、训练步骤、数据划分与评测仍归 UP。

训练中的验证、`train --evaluate` 使用 training 适配和当前模型；只是临时 eval/no_grad，退出时逐个恢复模块 training 标志与参数 requires_grad，异常也恢复。EMA 模型同样保留状态。它们不是部署推理，绝不因为 `.eval()`、`no_grad()`、`requires_grad=False` 切换后端。

| 入口 | 固定用途 |
| --- | --- |
| `python -m up train`，包括 `--evaluate` | `training` |
| Runner 内 evaluate、NAS/多任务验证 | 保持当前训练用途 |
| `python -m up inference`、视频推理、`BaseInference` | `inference` |
| `python -m up to_onnx` | `inference` |
| 旧 API 直接初始化，没有显式用途 | 默认 training；已有上下文则沿用，不根据 Runner.training 猜测 |
| 独立包 `configure_platform` | 必须显式传 purpose |

## 2. 目录职责

```text
packages/up-platform/
  pyproject.toml, setup.py      独立分发，默认无 GPU 编译
  src/up_platform/
    __init__.py                薄公共 API
    set_ai_device.py           显式初始化、用途固定
    context.py                 操作调用与能力查询
    registry_factory.py        后端规格、显式作用域、固定路由
    catalog.py                 73 项已有操作的延迟注册
    contracts.py               逐项语义定义和完整性检查
    common/
      policy.py                原生无操作策略与通用生命周期钩子
      precision.py             可共用的 autocast 实现
      collectives.py           传输、进程组和集合通信
      legacy_contract.json     31 个运行时成员 / 73 操作的迁移 ABI 基线
    training/
      runtime/backend/base.py  TrainingBackend：AMP、scaler、优化器入口
      runtime/backend/factory.py  所选训练实现的加载与获取
      runtime/distributed.py   DDP 实现
      runtime/legacy.py        旧 linklink 参数广播/梯度同步
      evaluation.py            训练内验证状态保护
    inference/
      backend.py, factory.py   InferenceBackend 与独立选择工厂
    base/
      base.py, factory.py      DeviceBackend 与厂商选择
      <vendor>.py              SDK 与设备运行时差异
      cuda.py                  CUDA-shaped 运行时共享开关
      nvidia_extension.py      选择 NVIDIA 时才允许加载二进制
      csrc/nvidia/pybind.cpp    聚合绑定，扩展名不变
    ops/
      __init__.py, base.py     统一门面与共用操作入口
      nms/common.py           同设备公共实现
      nms/nvidia/             厂商实现、可选 csrc/
      roi/nvidia/             RoIAlign、PSRoI 及 csrc/
      iou/, pointcloud/       IoU、点云，内部按厂商分组
      deform_conv/, focal_loss/  内部按厂商分组
      normalization/common.py, sparse/common.py  公共层与稀疏接口
      convolution/metax/, mm/metax/  显式 v1/v2，不改变 native
    backends/, shared/        旧导入兼容，不是新实现入口
    experimental/numerics/     portable_v1 算法及 policy.py 生命周期；默认不加载
  tests/                       不安装 UP 也可运行
  examples/vendor_plugin/      独立可安装、仅覆盖一项的 CPU 示例
up/accelerator/                旧入口、插件签名与 UP 配置转换
docs/platform/                当前平台架构、接口、指南、矩阵
docs/reports/                 冻结报告、证据归档索引和整理验收
```

同一操作的公共实现放在该类别的 common.py，由各用途/厂商显式复用。common 通信不拥有 DDP、梯度同步策略或训练状态；这些在 training/runtime，UP 仍负责编排。

新增厂商只需设备后端和注册入口；`register_backend(VendorDevice)` 默认选择公共执行类，缺失算子再补到对应类别。没有差异就不创建训练/推理子类。旧 backends/shared 地址只是兼容转发，不需要新厂商仿写。

数值实验通过懒加载策略接入，native 不导入 experimental。公共运行时只调用策略钩子，实验自己负责模式、RNG、局部操作注册与开关恢复，见[数值策略](NUMERICAL_POLICIES.md)。这不是新增厂商必须实现的另一套接口。

## 3. 安装与使用

先准备厂商匹配的 PyTorch/SDK/驱动及业务依赖，不自动安装或替换厂商 Torch。再从仓库根目录执行：

```bash
python -m pip install setuptools wheel packaging
python -m pip install --no-deps --no-build-isolation -e ./packages/up-platform
python -m pip install --no-deps --no-build-isolation -e .
```

仅平台包元数据/普通导入不需要 PyTorch；执行张量功能和 CPU 测试需要事先安装 CPU 或厂商 PyTorch。默认安装和 `UP_BUILD_BACKEND=auto/none` 都不探测或编译 CUDA。NVIDIA 扩展须明确在平台包构建阶段启用，见[根 README](../../README.md#nvidia-扩展)。根目录的 `setup.py build_ext` 不再生成旧二进制。

```python
from up_platform import configure_platform

ctx = configure_platform({"backend": "cpu", "device": "cpu"}, purpose="training")
ctx.runtime.manual_seed(131)
with ctx.training.autocast(enabled=False):
    pass
print(ctx.purpose, ctx.capabilities()["backend"])
```

生产进程初始化后保持厂商和用途不变；`reset=True` 仅供隔离测试或显式重新初始化，不能用于训练验证切换。延迟注册必须在初始化前完成；所选用途模块导入/构造中的注册也会在初始化结束前冻结。

UP 原 YAML 保持不变：`runtime.accelerator.backend` 选择厂商，`runtime.device` 为设备，plugins/requirements/numerics 仍在 accelerator 下。原 `--backend dist/linklink` 仍指通信策略。普通用户无需增加 purpose 配置。

## 4. 文档与检查

- [接口定义](INTERFACES.md)：公共运行时、两种执行接口和路由义务。
- [完整 73 项索引](API_INDEX.md)及[机器契约](interface_contract.json)：参数、返回、设备/布局、空输入、状态、梯度、导出、dtype、依赖。
- [新增厂商指南](ADDING_VENDOR.md)：完整模板、单用途交付、单项覆盖及测试步骤。
- [能力矩阵](CAPABILITIES.md)：注册不等于依赖可用、已执行或真机通过。
- [环境要求](ENVIRONMENTS.md)：厂商版本组合、普通依赖与历史兼容前置。
- [数值策略](NUMERICAL_POLICIES.md)：默认 native 与可选实验的隔离边界。
- [当前真机与完整预算回归](../reports/CURRENT_VALIDATION_20261022.md)：固定运行时、数据、种子、审计与未通过项。
- [历史拆包验收](../reports/platform/VALIDATION.md)：原冻结阶段的 CPU/构建结果，不代替当前 GPU 验收。

```bash
python tools/platform/check_hardware_boundaries.py
python tools/platform/export_platform_contract.py --check
python tools/platform/export_accelerator_contract.py --check
python tools/platform/check_project.py
python -m pytest -q
```

检查同时防止平台反向导入 UP、训练/推理模块相互导入和上层直接使用厂商接口。契约或注册变化需审查语义定义后执行两个导出器的 `--write`，CI 会检查漂移。
