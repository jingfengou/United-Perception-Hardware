# 硬件验证入口

这里是测试和诊断工具，不参与 UP 训练运行时，不改变模型算法。仅显式调用训练命令才使用硬件；CPU CI 不启动真实训练或旧多种子任务。

## 工具分组

| 分组 | 入口 | 使用边界 |
| --- | --- | --- |
| 当前平台对照 | `validate_platform_equivalence.py`、`collect_platform_evidence.py` | 独立进程对比训练、训练 eval、恢复与独立推理；不当成完整数据集精度 |
| 当前单种子完整预算 | `validate_current_accuracy.py` | 固定四模型配方与 seed=131，成对运行两个冻结版本；不是恢复旧多种子任务，见[本轮报告](../../docs/reports/CURRENT_VALIDATION_20261022.md) |
| 基础真机回归 | `validate_vendor_runtime.py`、`validate_native_ops.py`、`validate_models.py`、`validate_distributed.py`、`validate_onnx.py` | 需要真实 SDK、工具链与适配依赖，分别记录结果 |
| C500 局部诊断 | `diagnose_metax_determinism.py`、`run_metax_determinism_controls.py`、`collect_metax_determinism.py` | 固定操作数、原容差与失败记录，不能泛化为全形状确定性 |
| DETR 归因 | `diagnose_detr_origin.py` | UP tensor tape、独立官方网络、逐步梯度/ATen 追踪与纯 PyTorch 重放，见[归因报告](../../docs/reports/DETR_ORIGIN_20261022.md)；不修改模型或恢复旧多种子任务 |
| C500 GEMM 修复 | `validate_metax_gemm.py`、`validate_metax_detr_fix.py`、`collect_metax_gemm.py` | 真实步幅的固定算例及梯度、完整预算的 eval/恢复/独立推理和归档审计，见[专项报告](../../docs/reports/METAX_GEMM_FIX_20261022.md)；原容差不变 |
| C500 native/v2 成本 | `benchmark_metax_detr.py`、`collect_metax_detr_benchmark.py` | 同卡 seed=131，冷/热缓存各一组 ABBA 完整预算；排除 eval/save 的计时、分配器峰值和 CPU 指标复算，见[成本报告](../../docs/reports/METAX_NATIVE_V2_20261022.md) |
| 固定预算精度协议 | `prepare_*`、`validate_training_accuracy.py`、`run_multiseed_accuracy.py`、`run_cross_vendor_accuracy.py` | 必须显式给定计划/数据/输出；旧阶段按冻结版本复现，不恢复当前暂停任务 |
| 数值实验诊断 | `probe_*`、`trace_gpu_dispatch.py`、`validate_portable_*`、`run_portable_probes.py` | 默认关闭，仅显式实验；不是新增厂商接入步骤 |
| 证据与统计 | `audit_*`、`collect_*`、`package_*`、`summarize_*`、`recompute_*` | 校验执行版本/计划/数据，保留失败与未验证；CPU 指标复算不是模型代算 |

`collect_hardware_validation.py` 属于拆包前的历史协议，不能用于当前平台出具通过报告；当前结构用 `collect_platform_evidence.py`。历史统计器中 `tools/...` 的源码索引键故意保留，必须配套历史 source snapshot / `--repository`，不能随目录迁移改掉原哈希约定。

查看参数不会启动训练：

```bash
python benchmark/hardware/validate_platform_equivalence.py --help
python benchmark/hardware/run_cross_vendor_accuracy.py --help
```

运行前准备两棵明确版本的源码、相同输入/配置、独立输出目录与可用硬件。完整命令和结论见[报告索引](../../docs/reports/README.md)；不把 `--help` 或 CPU 测试当成 GPU 验收。

## 旧命令迁移

| 原路径 | 当前路径 |
| --- | --- |
| `tools/validate_*.py`、训练/数据/诊断/统计脚本 | 同名 `benchmark/hardware/*.py` |
| `tools/check_hardware_boundaries.py` | `tools/platform/check_hardware_boundaries.py` |
| `tools/export_accelerator_contract.py` | `tools/platform/export_accelerator_contract.py` |
| `tools/export_platform_contract.py` | `tools/platform/export_platform_contract.py` |
| `tools/build_hardware_inventory.py` | `tools/platform/build_hardware_inventory.py` |
| `tools/snapshot_hardware_source.py` | `tools/platform/snapshot_hardware_source.py` |

工具迁移没有保留几十份旧路径壳文件；当前命令按此表更新。要严格复现旧报告，使用其对应固定提交或归档源码，而不是混用当前工具。UP 的公开 Python 兼容入口、CLI 和配置字段没有因此删除。
