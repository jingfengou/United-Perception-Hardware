# 完整接口定义

开发新厂商时，按本文的接口实现不兼容部分即可，不需要先猜“厂商支持哪些模型”。[完整索引](API_INDEX.md)逐项列出 31 个旧运行时成员的归属和 73 项路由操作；[JSON](interface_contract.json)供程序检查。实现一项能力后，所有使用这项接口的模型都能复用；是否能跑完整模型仍需确认其所需能力齐全并实测。

## 1. 公共入口

```text
configure_platform(config=None, *, purpose, reset=False) -> PlatformContext
get_platform() -> PlatformContext
register_backend(spec=None, *, training=..., inference=..., replace=False, probe=None)
register_op(backend, name, implementation=None, *, scope,
            backward=None, replace=False, dtypes=None, variants=None,
            conventions=(), dependencies=())
capabilities() -> dict
require(name, **checks) -> dict
```

`config` 为字典或 `PlatformConfig`，字段仅 backend、device、plugins、requirements、numerics。purpose 仅 training/inference，不从模型状态推断。get_platform 在尚未初始化时明确报错；旧 get_accelerator 为兼容保留自动初始化。

新增厂商推荐 `register_backend(VendorDevice)` 或设备类装饰器，默认直接使用公共 TrainingBackend / InferenceBackend，不必创建两份空类。类声明 name、device_type，可继承 DeviceBackend 的 op_fallbacks，额外环境约束放 requirements。只有确有差异时传 training / inference 工厂（可调用对象或 `module:attribute`）；省略表示公共实现，显式 None 表示该用途未支持。具体模板见[新增厂商指南](ADDING_VENDOR.md)。

旧 `BackendSpec(name, device_type, device_factory, training_factory=None, inference_factory=None, shared_providers=('portable',), requirements={})` 继续可用，**保持原 None 默认值的未支持含义**，不与简写的 training / inference 参数混用。三个工厂可为可调用对象或 `module:attribute`；设备工厂接受 `device=None`，执行工厂接受 `context`。模块路径只在该厂商该用途被选中时加载。SDK 在 device 构造或操作执行时导入，不能放在注册模块顶层。

`name` 是厂商，device_type 是 PyTorch 设备命名：NVIDIA 与 MetaX 都可为 cuda，但不会因此复用 NVIDIA 二进制。额外后端可通过 config.plugins 或 `up_platform.backends` entry point 注册；只发现用户请求的 entry point，不遍历加载所有 SDK。

## 2. 上下文与设备

PlatformContext 提供只读 purpose、runtime、当前 execution、ops、layers、sparse、capabilities/require/call。`.training` / `.inference` 访问错误用途会报 CapabilityError。设备方法可从 ctx.runtime 或兼容委托 ctx.method 调用。

DeviceBackend 保留原查询、绑定、迁移、同步、RNG、内存、环境校验、卷积开关及可选库加载语义。继承共享实现，仅覆盖厂商 API 不一致的方法。逐项参数/返回义务见[31 成员迁移表](API_INDEX.md#runtime-migration-31-members)。其中 autocast/GradScaler/call/capabilities 已分别归执行层和上下文，旧门面仍提供原名字。

现行设备基类为 `up_platform.base.base.DeviceBackend`，注册实现位于 `registry_factory.py`，初始化位于 `set_ai_device.py`；完整操作按 `ops/<类别>/<厂商>/` 放置，类别内 `common.py` 供明确兼容的厂商复用。旧模块路径继续作为别名，不产生第二份实现。整体关系见[当前总设计](../architecture/PLATFORM.md)。

设备迁移不改 dtype 和训练图；Module.to 的参数身份遵循对应 PyTorch 版本。RNG 保持原设备作用域，不扩大成所有设备；同 seed 不保证跨厂商相同随机张量。内存数值单位为字节；CPU 统计为 0。health_check 必须真实在所选设备计算。版本检查只校验显式约束，不替换用户环境。

## 3. 两侧执行接口

| 类别 | 方法/入口 | 完整义务 |
| --- | --- | --- |
| training | `autocast(enabled=True, **kwargs)` | 返回上下文管理器，校验 dtype、BF16 和实际设备可用性；可复用 common 实现 |
| training | `grad_scaler(**kwargs)` | 提供 scale/step/update/state_dict/load_state_dict 兼容对象；验证不调用 step/update |
| training | `wrap_model(model, device_ids=None, **kwargs)` | DDP 包装，保留参数身份；不接管学习率/训练循环 |
| training | `wrap_legacy_model(model, sync=True, master_rank=0)` / `reduce_gradients(model, sync=True, allow_dead_parameter=False)` | 原 linklink 参数广播/同步与异步梯度钩子；旧 DistModule/reduce_gradients 只转发，业务通信标志显式留在 UP |
| training | `build_fused_optimizer(config)` / `reload_fused_optimizer(optimizer)` | 可选厂商实现；保留参数组、master 参数和 optimizer 状态；不改超参数 |
| training | `ops` / `layers` / `sparse` / `call(name, ...)` | 可微操作必须提供完整反向及 context；NMS 等不可微操作不要求伪造反向 |
| inference | `autocast(enabled=True, **kwargs)` | 独立选择推理精度实现，默认复用公共 autocast，不创建 GradScaler |
| inference | `ops` / `layers` / `sparse` / `call(name, ...)` | 仅本用途和 common 能力；默认普通 PyTorch 模型，不自动打包、量化、换模块 |
| inference | `export_onnx(model, args, destination, **kwargs)` | 已有 ONNX 导出适配；默认保留 autograd symbolic 的 legacy 路径，支持新版时默认 dynamo=False；不替换节点、domain、属性 |

TrainingBackend 位于 `training/runtime/backend/base.py`，由同目录 factory.py 加载；InferenceBackend 位于 `inference/backend.py`，由 inference/factory.py 加载。两者不隐式导入另一侧实现。兼容的 autocast/语义操作显式共用；厂商可单独覆盖训练或推理。内置厂商直接使用公共类；原 backends/<vendor>/training.py、inference.py 仅为兼容地址。推理 context 不能获取 GradScaler、融合优化器或 DDP 操作。

底层集合通信放 common，语义与固定 PyTorch 版本一致，包含原地修改、进程组、异步 Work；DDP 及梯度同步策略属于训练。原有 Gloo 对象通信、显式 CPU BEV/点云接口和 SofterNMS CPU 内部阶段保留，不授权新增静默 CPU 模型计算。

## 4. 完整操作路由

初始化时冻结注册表。native 默认查找顺序：

1. 当前厂商的当前用途实现；没有时使用该厂商 common 实现。
2. 按 shared_providers 的显式顺序，在每个供应者查当前用途/common。
3. 报错，包含 backend、device、purpose、operation 与原因。绝不查另一用途专用实现。

`scope` 必须为 common/training/inference。工厂导入时可以进行所选用途注册，之后再注册不会更改已有 context，需新建测试 context。普通导入只创建元数据，不初始化设备或导入所有厂商。

旧 register_op 省略 scope 时，已知训练专用操作映射为 training，其他默认为 common；旧 callable / module:factory 插件也继续延迟构造。新插件必须显式注册。环境配置的 requirements 与厂商 BackendSpec 要求同时满足，用户字段不能覆盖掉厂商约束。

注册单位是完整语义操作，不是裸 forward/backward 函数指针。backward=True 仅是声明，不自动实现导数。dtype、variants 为 None 表示未声明，不是任意精度/变体已通过；require(backward=True) 会拒绝未声明完整反向的操作。

`dtypes` 检查嵌套参数中浮点 Tensor；`variants` 通过实现签名绑定位置/关键字参数后检查。接口实现本身仍需检查整数标签 dtype、设备、shape、布局和业务约束。返回值的完整性必须通过契约测试证明，注册器不伪造 Tensor 或梯度。

`conventions=('offset',)` / `('aligned',)` 声明兼容门面可以显式传入的 UP 坐标约定。新平台默认 offset=0/aligned=False，UP 门面传原 ALIGNED_FLAG 的真实值。未声明的旧插件不被追加新 kwargs。RoIAlign 把 aligned 保存到 autograd context，反向不能重新读取可变全局标志。

dependencies 为懒加载模块名，或 provider:sparse/link/quantization 等运行时可选供应者。`require('roi_align', dtype=..., backward=True, variants=..., check_dependencies=True)` 只显式检查该操作，不预加载所有依赖。实际缺失能力仍在执行时报错，不静默回退。

## 5. 全量覆盖与特殊契约

73 项逐一见[API 索引](API_INDEX.md#operations-73)，每项含 inputs、signature、legacy_signature、returns、device_and_layout、empty_input、state_updates、gradients、export、registered_providers。空输入“保留供应者限制”是明确未承诺新支持：必须复用原异常/限制，新增支持需测试后改契约，不能臆造输出。

- 20 项原生语义操作及 torchvision_nms：2D/3D NMS、RoI、DeformConv、Focal Loss、IoU、点云操作与旋转评估。
- SyncBN 与旧 linklink SyncBN：训练函数与 shared forward 分别注册，参数名称和 running stats 保持。
- 9 项 sparse.*：构造、特征、替换、稠密化、容器和类型判定；不引入 wrapper 参数前缀。
- 21 项 distributed.*：20 项 common 通信/查询，wrap_model 仅 training。
- 15 项 legacy_link.* 与融合优化器、精度注册、量化枚举：仍可懒加载，不承诺任意 SDK 版本 ABI。

保留旧算子导入和 `.apply()`，不要求所有训练期间调用都有梯度。旧 CPU 导出占位绝不能用作实际数值计算。没有 ONNX symbolic 的操作明确不保证导出，不能静默换成不等价算子。

## 6. 能力与验收不是一回事

native 的操作目录不包含实验供应者。显式数值策略通过通用生命周期钩子将局部操作合并到当前 context；不修改全局注册表，不改变 purpose，也不强迫厂商实现数值实验。完整策略边界见[数值策略](NUMERICAL_POLICIES.md)。

capabilities 每项分别记录 registered、dependencies_available（未知为 None）、executed（成功调用数）、hardware_validated。当前结构迁移不会把历史证据升级为新结构真机通过，hardware_validated 默认 False。dependencies_available=True 也仅表示声明模块已加载，不证明 SDK ABI、设备可用、梯度正确或精度收敛。

native 默认不变；experimental/numerics 的 portable_v1 只有显式选择才启用，包含 GPU FP64 中间计算和额外开销，不是新厂商基本接入义务。原失败/未验证证据继续保留；不得混用历史快照与新包结果。

沐曦另提供显式 `numerics: metax_deterministic_v1`，定义、支持变体、生命周期及同厂商证据见[专项说明](../reports/platform/METAX_DETERMINISM.md)。它只替换 GPU FP32 普通二维卷积的权重一阶梯度，不改目的、语义算子供应者或 native RNG；`numerics_state()` 报告厂商 profile 及真实执行次数。该选项不是通用确定性保证，不用于自动启用 FP64 或掩盖 native 未通过项。

`metax_deterministic_v2` 另覆盖密集二维 GPU FP32 `aten.mm.default`，含通过原 PyTorch autograd 调用的反向矩阵乘法；不按模型名、参数名或 `[4,600]` 特定形状分支。Triton 只在首次执行该操作时加载，保留输入步幅并返回同设备 FP32 矩阵，空 K 返回零；其他 GPU dtype、稀疏布局或 `mm.out` 不在该覆盖契约内，不能伪装成已验证变体。它不替换全部 BMM/addmm、全部厂商 BLAS 或导出路径，详见[专项记录](../reports/METAX_GEMM_FIX_20261022.md)。
