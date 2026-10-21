# 五模型训练证据

范围：NVIDIA A800、FP32、单卡完整 Runner；CIFAR-10 固定 10,000/2,000 子集与 PennFudan 固定 120/50 划分。不是完整 ImageNet/COCO 收敛或国产设备真机验收。解释与复现命令见[主报告](../../MULTISEED_ACCURACY.md)。

## 先看哪些文件

| 文件或目录 | 内容 |
| --- | --- |
| `five-model-summary.json` | 主五模型汇总；Faster R-CNN/ResNet 来自原计划，其余来自追加计划 |
| `five-model-pairs.csv` | 30 对主要种子的逐项指标；AP/Top-1 均已转为百分数，差值为百分点 |
| `plan.json` | 原计划：46 次完整训练，包括 6 次同版本重复及初版 DETR 诊断 |
| `extension-plan.json` | 追加计划：YOLO、ViT、修正优化器后的 DETR，各 5 对 |
| `results/` | 原计划每次配置、报告、比较与统计，含不良 DETR 配方的原始结果 |
| `five-models-v2/results/` | 追加计划每次配置、报告、比较与统计 |
| `five-models-v2/native-shapes/` | YOLO 640px/80 前景类、ViT 224px/1000 类的补充合成功能检查 |
| `checkpoint-finiteness-original.json` | 原计划 46 个保存的模型和优化器状态有限值检查 |
| `checkpoint-finiteness-extension.json` | 追加计划 30 个保存状态的有限值检查 |
| `data-integrity-recheck.json` | 图片、标注、权重和锚框与准备清单的重新校验 |
| `data-integrity-final.json` | 全部运行完成后再次核对 12,170 张图片、标注、权重和锚框 |
| `publication-source-check.json` | 407 个原版与 436 个路由版 Python 模块对冻结 Git 源码的核对 |
| `contracts-final.xml` | 117 项 CUDA 隐藏的 CPU 测试结果；先前阶段的 XML 单独保留 |
| `offline-statistics-recheck.json` | 阻止所有 Torch 导入，仅用 JSON/SciPy 复算两个完整计划，统计相同 |

五模型主要验收共 66 次完整训练、94,720 个优化器步骤，其中含 6 次 Faster R-CNN 同版本重复。初版 DETR 的 10 次完整运行另外保留，不混入主表；GPU 预检查、合成检查和失败启动也不计入主要数量。

## 不要混淆的结论

- `strict_numeric_passed` 使用既定数值阈值检查 loss、最终模型权重和指标；Faster R-CNN 的这项检查仍不通过。
- `bitwise_training_state_equal` 只比较记录到的 loss、最终模型 `state_dict` 哈希和指标，不表示记录了全部中间激活、每步梯度或逐位优化器状态。
- `equivalent_within_margin` 是固定 holdout 上跨种子的平均指标 TOST，预设范围为 +/-1 个百分点；不是每个种子的差值都小于 1 点，也不是全量基准保证。
- checkpoint 恢复检查是新建模型/优化器加载后重新评估，不包括恢复后再训练完整一个 epoch 的连续性验收。
- 初版 DETR 两边平均 AP 同为 4.2321%，只说明该错误微调配方的路由一致，不能算正常训练通过。主表 DETR 使用另外预先固定的分组学习率配方。
- ViT 20 epoch 小数据配方的绝对 Top-1 较低，不能因两边相同就宣称充分收敛。跨模型初始化和配方不同，也不能据此排名模型优劣。

## 驱动快照

| 快照 | 用途 |
| --- | --- |
| `drivers/validate_training_accuracy.py` | 原计划训练驱动，SHA-256 以 `85f81852` 开头 |
| `drivers/validate_training_accuracy-v2.py` | 追加计划驱动，SHA-256 以 `a8c0138f` 开头 |
| `drivers/validate_training_accuracy-preflight-v2a.py` | 最早 YOLO/ViT GPU 预检版本；补回 `np.int` 前的字节版本，经已记录的 `f1ed21b0` 完整哈希验证 |
| `drivers/accuracy_detr_adapter.py` | 首次三维 NLL 的失败版本，不用于成功训练 |
| `drivers/accuracy_detr_adapter-v2.py` | 成功训练的等价展平标签损失版本，SHA-256 以 `fcc37b65` 开头 |
| `drivers/run_multiseed_accuracy.py` | 原计划最初调度器 |
| `drivers/run_multiseed_accuracy-hardened-v1.py` | 加强校验后的原计划调度器 |
| `drivers/run_multiseed_accuracy-v2.py` | 追加计划调度/汇总器，SHA-256 以 `c44d8488` 开头 |
| `drivers/validate_models-native.py` | 两项原始分辨率补充功能检查 |
| `drivers/resume_yolo_schedule.py` | 本次任务的排队续接记录，不是通用生产训练工具 |

运行旧快照时需放在单独 `tools/` 目录并使用原文件名；DETR 成功训练应将 `accuracy_detr_adapter-v2.py` 放为 `accuracy_detr_adapter.py`。当前生产工具在仓库根 `tools/`。不要在正在执行的训练中修改驱动文件。

## 失败、调度与完整性

`setup-failures.json` 与原始日志保留首次 DETR 三维交叉熵的两次严格确定性失败，均未执行优化器更新。`five-models-v2/preflight/vit-v1.log` 保留旧 `np.int` 别名导致的数据加载失败，同样没有训练更新；修正只是给两边恢复相同的原始别名。

`opencv-thread-probe.json` 比较默认 48 和 2 线程，数据/RNG 相同但没有提速，正式训练未采用这个修改。`orchestration-resume.json` 记录只替换排队进程、保留两个在途训练子进程继续；未开始的 YOLO seed135 改用 GPU0，所有配对仍在同一物理卡顺序运行。

`original-evidence-index.json` 与 `extension-evidence-partial-index.json` 是不可变已完成文件的阶段性传输哈希；后者排除可变的综合汇总，保留 `vit-only-summary.json`。最终完整文件索引见 `evidence-index.json`。锚框 JSON 保留原文件 CRLF 字节，不能格式化后继续沿用旧校验值。

统计可以只用 JSON 和 SciPy 复核，不需要 GPU、图片或大 checkpoint；验证器仍检查报告哈希、配对前置条件、预算、恢复指标及跨种子来源一致性。这不是对未公开大权重的独立复算。图片、模型和优化器大文件仍留在 A800-1 的原工作目录，不提交 Git。
