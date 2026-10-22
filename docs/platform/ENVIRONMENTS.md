# 环境要求

可用环境同时满足 **UP 业务依赖 + 厂商运行时 + 所需操作实现**。平台只校验显式版本要求，不会安装或切换 Python、替换厂商 PyTorch，不能自动消除 ABI/API 冲突。

## 已有实测组合

| 用途 | 环境 | 记录 |
| --- | --- | --- |
| 当前 NVIDIA 回归 | RTX 4090、Linux、Python 3.12.3、Torch 2.10.0+cu126、TorchVision 0.25.0+cu126 | [完整预算与严格回归](../reports/CURRENT_VALIDATION_20261022.md)，含实际 toolkit 小版本差异记录 |
| 当前 C500 回归 | 厂商容器、Python 3.12.11、Torch 2.10.0+metax3.8.1.0、TorchVision 0.25.0+metax3.8.1.0 | [完整预算、native 与未通过项](../reports/CURRENT_VALIDATION_20261022.md)；不是 native 全流程精度通过 |
| 前次平台拆包短程 | A800 / C500，各自冻结源码与厂商版本 | [历史拆分报告](../reports/platform/REAL_HARDWARE.md)、[局部修复报告](../reports/platform/METAX_DETERMINISM.md) |
| CPU 本地回归 | Windows、Python 3.13.11、Torch 2.10.0+cpu | [当前回归](../reports/CURRENT_VALIDATION_20261022.md)，不是 GPU 部署组合 |
| 首轮历史 NVIDIA | Python 3.12.3、Torch 2.8.0a0+5228986c39.nv25.06、TorchVision 0.22.0a0+95f10a4e、CUDA 12.9 | [历史报告](../reports/hardware/VALIDATION.md)，不与新组合混装 |
| 昇腾 / 寒武纪 | 尚无本仓库真机确认的完整组合 | 需厂商 SDK、匹配 PyTorch 和目标模型验收 |

容器是否必要由厂商的软件分发方式决定，平台包可安装在厂商容器内。驱动与设备映射仍需主机管理员正确配置；进入容器不等于模型已适配。

## 依赖文件如何使用

- 根 `requirements.txt` 保留上游历史约束，含 SciPy 1.1.0、ONNX <= 1.6.0，不能当作现代环境的一键安装配方。
- [`validated-a800.txt`](../../requirements/validated-a800.txt) 是首轮 A800 应用/测试依赖子集，不是完整容器锁。
- [`metax-c500-validation.txt`](../../requirements/metax-c500-validation.txt) 对应指定厂商镜像的验证组合，不覆盖其他 SDK 版本。
- 平台包和 UP 的 Python 元数据下限为 3.8，不代表任意更高 Python/PyTorch 组合都已验证。
- 安装顺序：匹配驱动/SDK/PyTorch/TorchVision → 业务依赖 → 平台包 → UP。`--no-deps` 不会补齐业务依赖。

UP 的数据/评估链路还依赖 YAML、NumPy/SciPy、图像库、ONNX、COCO 等；检测/点云任务可能需要原生扩展或 spconv，量化/剪枝需要对应可选 SDK。按目标任务准备，不强制安装未使用的厂商库。

历史真实数据回归的共同启动器处理了旧 `collections.Iterable`、NumPy 类型别名和旧权重序列化问题；这不证明原始 CLI 对任意现代环境自动兼容。COCO 库也曾需要标准 `pycocotools==2.0.10` 覆盖，详见[真实训练报告](../reports/hardware/TRAINING_ACCURACY.md)。不全局关闭权重安全加载，不用这些兼容前置改变模型 dtype。

## 新厂商环境记录

每个厂商分别记录 GPU、驱动/SDK、容器镜像及 digest、Python、Torch/TorchVision、通信库、第三方算子与普通依赖版本；训练、推理和 AMP/多卡分别验收。厂商只支持特定组合时先保留该组合，再解决框架冲突，不能强行装 NVIDIA wheel。

配置可以增加版本校验，例如：

```yaml
runtime:
  accelerator:
    backend: metax
    requirements:
      python: ">=3.12,<3.13"
      torch: ">=2.10,<2.11"
```

这只是已观察组合的约束示例，不代表该区间所有版本都经过验证。精度策略、TF32 与 DNN 开关也属于复现环境，不能只记录张量是 float32。
