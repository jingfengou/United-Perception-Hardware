# 可选数值策略

基本厂商接入不依赖数值实验。默认 `numerics: native` 不进入实验 PyTorch 模式、不改变厂商 RNG，也不注册实验操作供应者。

| 策略 | 实现归属 | 用途与限制 |
| --- | --- | --- |
| `native` | `common/policy.py` | 无操作策略，保留原生设备行为 |
| `metax_deterministic_v1` | `backends/metax/` | 显式 C500 FP32 卷积权重一阶梯度替换；有临时显存/计算代价，见[报告](../reports/platform/METAX_DETERMINISM.md) |
| `portable_v1` | `experimental/numerics/policy.py` | 实验性固定 RNG、归约顺序及部分 GPU FP64 中间计算；默认关闭，不承诺跨厂商全量精度等价 |

## 隔离边界

公共运行时只认识策略生命周期：启动/关闭、RNG 保存恢复、操作供应者/局部注册表、autocast 校验、卷积开关约束和状态查询。它不导入实验实现，不包含 portable_v1 的算法、计数器或 PyTorch 模式管理。

策略通过 `up_platform.registry.register_numerical_policy(name, factory)` 注册懒加载工厂。内置工厂地址集中在后端元数据模块；只有明确选择该策略才导入实现。实验操作只合并到该上下文的注册快照，不污染 native 全局注册表。

`PortablePolicy` 自己管理 TorchFunction/TorchDispatch 模式、MHA/DNN 开关、RNG envelope 与操作计数；关闭时检查线程及模式栈，激活失败时恢复已改变的模式和全局开关。公共 AMP 与卷积实现只调用通用策略钩子。静态检查阻止 common/shared/context 再次导入 experimental。

保留旧配置字符串和诊断脚本使用的私有 `_numerics` 只读兼容入口，不作为新厂商扩展 API。训练 eval 不更换策略或 purpose。

## 使用与验收

在已有 `runtime.accelerator` 段中显式选择，例如 C500 的 `numerics: metax_deterministic_v1`。是否使用该策略由用户决定，选择厂商不会自动开启。

`portable_v1` 当前不支持 AMP，需要兼容的 CUDA-shaped 厂商运行时及厂商 Triton。其历史完整预算结果对应冻结源码，不应拿整理后的运行时续跑旧阶段或覆盖旧结果。见[历史数值协议](../reports/hardware/PORTABLE_NUMERICS.md)和[暂停阶段报告](../reports/hardware/CROSS_VENDOR_FIX.md)。

结构整理只验证生命周期与 native 隔离，不提升任何真机支持等级。后续数学修复必须新建源码与结果阶段，保持原模型、容差及原失败记录。
