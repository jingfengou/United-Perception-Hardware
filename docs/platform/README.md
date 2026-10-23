# 平台层设计

以原版 UP `52aaeb9ed27a490820696d7754d19f98cd6ecd11` 重建，参考 LightX2V `19c19614b067f1746ac24350dfd097e1c0b48591` 的设备、算子类别和训练后端。
同仓命名空间 `up_platform` 随 UP 一起安装，不反向导入 `up`，不复制旧硬件分支的兼容层和数值实验。

## 目录

```text
up/                         原版模型、损失、数据、配方和业务编排
up_platform/
  base.py                   DeviceBackend、PlatformBackend
  factory.py                显式选择、插件注册、固定用途
  contracts.py              契约和能力状态
  backends/                 设备子类、各类别的显式组合
  ops/<类别>/
    base.py                 具名接口及契约
    nvidia.py / metax.py     厂商绑定，允许显式引用共享实现
    ascend.py / cambricon.py 未启用的逐项模板
    torch_impl.py            可共用的 PyTorch 调用
  ops/_native/              扩展构建、加载与内核源码
  training/                 AMP、DDP、优化器、训练验证状态
  inference/                推理 AMP、原有 ONNX 导出
  templates/                SDK/autograd 示例
tests/platform/             无 UP 依赖的接口及路由测试
tools/*platform*             审计、构建、模板生成与复现工具
docs/platform/              接口、接入、环境、证据
```

## 调用规则

`UP → 类别接口 → 当前厂商文件中的绑定 → 厂商 PyTorch / 共享实现 / SDK`。

- 每家文件明确声明实现，没有“找不到就尝试另一家”的全局回退。
- `nvidia/cuda` 与 `metax/cuda` 身份不同；分别编译 `_nvidia_ops`、`_metax_ops`，不混用二进制。
- 共享源码由两家工具链分别编译；源码共享不意味着沐曦加载 NVIDIA 二进制。
- 用途模块按需加载，配置后固定；另一用途使用新进程，不按 `.eval()`、`requires_grad` 或 `no_grad()` 猜测。
- 普通导入不初始化设备、不加载扩展。旧 `up.extensions` 路径及 `.apply()` 仍是兼容门面。
- 基础算术、索引、reshape、张量存储、autograd、模块系统继续由 PyTorch 负责，属于设备桥接前提。
- 原版显式 CPU 数据处理/评估/序列化保留；新增缺失操作不得静默 CPU 代算。
- 原版 CPU RoIAlign 导出占位行为保留，但不能用于证明 CPU 数值正确。

## 训练与推理

| 入口 | 用途 | 约束 |
| --- | --- | --- |
| train | training | 原配方决定优化器分组、学习率、损失和训练步骤 |
| 训练内 eval、train --evaluate | training | 临时 eval/no_grad，恢复进入前每个子模块及 EMA 模式 |
| 独立图片/视频推理 | inference | 普通 PyTorch 模型，不自动替换部署引擎 |
| 原有导出 | inference | 保留 legacy ONNX symbolic |

NMS 等本来不可微的训练期操作仍不可微。适配不改 checkpoint 键、初始化、参数身份、量化规则或剪枝策略。

## 审查入口

- [原版审计](upstream_inventory.json)：从基线 Git 对象生成，包括直接调用、Tensor 候选、扩展与第三方导入。AST 不是张量类型推断器，候选不等于已验证。
- [完整接口](INTERFACES.md) / [机器契约](interfaces.json)：由父类自动生成签名、返回、梯度、状态、设备、布局、空输入及导出约束。
- [当前调用位置](call_sites.json)：静态可解析的 UP → 平台入口；动态调用和 CPU/基础桥接仍需人工审查。
- [linklink 例外](linklink_call_sites.json)：未完成的 SDK 边界，CI 不允许无审查增加。
- [变更说明](CHANGES.md)：增删改目的、兼容修复与未完成项。

```bash
python tools/platform_inventory.py
python tools/platform_contracts.py
python tools/platform_call_sites.py
python tools/check_platform_boundaries.py
python -m pytest tests/platform -q
```

`capabilities()` 的已注册、依赖可发现、已执行、已验证分别记录，不把导入成功当成通过。
真实验收只认 [验证报告](VALIDATION.md) 的协议和证据。当前不宣称两家完整适配。
