# 当前平台架构

采用 LightX2V 风格的 **设备后端 + 按类别组织的算子 + 注册表 + 分用途工厂**。UP 模型、损失、数据、配方和 CLI 不重写；平台包仍为同仓独立安装的 `up-platform`。参考版本为 LightX2V `19c19614`，重构基线为本仓库 `ad36d77`。

## 1. 自上而下的职责

```text
UP 模型 / 损失 / 数据 / Runner / CLI
                 |
       up.accelerator 兼容与配置转换
                 |
     set_ai_device.configure_platform
       选择厂商，入口固定 purpose
          /                    \
training/runtime/backend     inference/factory
  AMP / scaler / DDP         autocast / export
          \                    /
       PlatformContext + registry_factory
                 |
     ops/<类别>/<厂商>/ 或 <类别>/common.py
                 |
     base/ 设备后端 + 厂商 PyTorch / SDK
```

普通设备无关 PyTorch 运算仍直接调用 PyTorch。平台只集中硬件边界，不重新包装整个 PyTorch。厂商原生算子兼容时无需重写；不兼容或需要优化时才增加实现。

## 2. 当前目录

```text
packages/up-platform/src/up_platform/
  __init__.py                  薄公共 API；普通导入不加载 Torch/SDK
  set_ai_device.py             显式配置与单一活动上下文
  context.py                   用途冻结、完整操作调用与能力检查
  registry_factory.py         厂商/算子注册、延迟解析
  catalog.py                  73 项既有语义操作的注册元数据
  base/
    base.py                   DeviceBackend：迁移、RNG、同步、内存等
    factory.py                厂商选择及内置设备规格
    nvidia.py, metax.py        CUDA-shaped 设备，身份分别管理
    ascend.py, cambricon.py    SDK 初始化及不兼容设备方法
    cuda.py                   已明确可复用的卷积运行时控制
    nvidia_extension.py       可选 NVIDIA 扩展加载与厂商检查
    csrc/nvidia/pybind.cpp     聚合绑定；扩展 ABI 名保持不变
  ops/
    __init__.py, base.py       统一调用门面与公共执行表面
    nms/common.py             同设备公共 NMS
    nms/nvidia/               NVIDIA NMS / TorchVision 包装及 csrc/
    roi/nvidia/               RoIAlign / PSRoIAlign / PSRoIPool 及 csrc/
    deform_conv/nvidia/       DeformConv 及 csrc/
    focal_loss/nvidia/        Focal / Cross Focal Loss 及 csrc/
    iou/common.py             同设备公共 overlap
    iou/nvidia/               2D/3D IoU、共享原生模块的 3D NMS
    pointcloud/nvidia/        点云池化、点/框查询及 csrc/
    normalization/common.py   SyncBN、原有层工厂和可选 SDK 门面
    sparse/common.py          稀疏层及张量操作契约
    convolution/metax/        显式 v1 卷积权重梯度实现
    mm/metax/                 显式 v2 FP32 GEMM 与固定内核
  training/
    runtime/backend/base.py   公共训练后端；有差异才覆盖
    runtime/backend/factory.py  只加载所选训练实现
    runtime/distributed.py    DDP 包装
    runtime/legacy.py         既有 linklink 梯度同步
    evaluation.py             训练内 eval 状态保护
  inference/
    backend.py, factory.py    公共推理/导出与单独工厂
  common/                     精度、底层通信、策略生命周期与 ABI 基线
  experimental/numerics/      portable_v1；默认不加载
  backends/, shared/          历史地址兼容，不作为新实现位置
```

同一类别的公共实现只有一份。暂未实现的厂商算子不创建空目录或假实现；没有训练/推理差异也不创建空后端子类。`base/nvidia_legacy.py` 只保存原有可选 SDK 委托，不是另一套张量计算实现。

## 3. 注册和调用

- `register_backend(VendorDevice)` 注册设备，默认复用公共训练/推理类；显式覆盖只影响指定用途，`None` 表示该用途未提供。
- `register_op(vendor, name, implementation, scope=...)` 注册完整语义操作。`common` 表示训练和推理均可使用；可微操作交付完整反向和状态，不只注册裸 forward。
- 先查当前厂商的当前用途/common，再查明确声明的共享供应者；仍缺失则报错。不会借用另一用途专用实现，也不会因两家都叫 cuda 而加载 NVIDIA 二进制。
- 模块/SDK 及所选用途按需加载；初始化结束后冻结操作供应者。注册成功、依赖可导入、实际执行、真机验收分别记录。

详见[完整接口](../platform/INTERFACES.md)和[新增厂商步骤](../platform/ADDING_VENDOR.md)。31 个旧运行时成员、73 项路由操作及旧 `.apply()` 保持兼容。

## 4. 训练与推理的分工

训练工厂负责 AMP/scaler、DDP、梯度同步及训练操作入口；独立推理工厂负责推理 autocast 和已有导出。两者共用设备基类和明确适用的操作，不互相导入。

训练中的 eval、`train --evaluate` 使用当前训练模型和 training 后端，临时 eval/no_grad 后恢复每个模块原状态；不创建部署模型、不切换 inference 算子、不更新优化器/scaler。独立图片/视频推理与导出由入口明确指定 inference。

## 5. 与 LightX2V 的对应及保留差异

| LightX2V | 本框架 |
| --- | --- |
| `lightx2v_platform/base` | `up_platform/base`：设备与 SDK 边界 |
| `ops/attn/<厂商>` 等 | `ops/nms/<厂商>` 等：针对 UP 的操作集合 |
| `registry_factory.py` | 同名注册模块，额外保留用途与契约检查 |
| `set_ai_device.py` | 同名显式初始化模块，由 UP YAML/入口调用 |
| `lightx2v_train/runtime/backend` | `training/runtime/backend`：训练基类与工厂 |

不照搬视频模型、量化算法、推理引擎或 FSDP。LightX2V 推理初始化入口具有导入副作用；本框架保留公共包无 Torch/SDK 普通导入的能力，因此必须显式初始化。既有 YAML 和通信参数不改为另一项目的环境变量约定。

## 6. 迁移与验证边界

[迁移清单](platform-layout.json)记录旧路径、新路径、基线提交和计算定义哈希。C++/CUDA 及受保护实现保留原源码内容；Python 迁移只允许导入位置变化，不重写数学定义。旧模块以模块别名共享同一实现，避免产生两套类、缓存或注册表。

NVIDIA 继续显式构建同名 `up_platform.backends.nvidia._C`，只是编译器从各算子类别收集源码和 include 路径。未新增静默 CPU 回退，native/v1/v2/portable_v1 的选择与默认值不变。

结构重构必须通过 CPU 契约、旧/新 API、用途隔离、训练 eval/恢复、ONNX、公共 wheel/sdist 与源码保护检查。历史真机结果继续只对原冻结版本成立；未重新执行 GPU 训练不能称为新架构精度验收。旧多随机种子任务仍暂停。
