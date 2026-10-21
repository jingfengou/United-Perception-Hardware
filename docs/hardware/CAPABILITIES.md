# 能力矩阵

本表描述仓库实现及本轮验证状态，不是厂商产品的完整支持矩阵。运行时可查询 `up.accelerator.capabilities()`，注册状态不等于 SDK 可执行状态或真机验证状态。

沐曦 C500 已完成[严格 FP32 四模型两步单卡功能检查](METAX_VALIDATION.md)、运行时和固定小样本 NMS/IoU 检查。默认 TF32 的 ResNet/DETR 恢复一致性失败保留；尚未验证 ImageNet/COCO 全量收敛、多卡或全部专用算子。

随后完成[四模型真实数据跨厂商对照](CROSS_VENDOR_ACCURACY.md)：CIFAR-10/PennFudan 固定预算、两平台各五种子，共 40 次训练/76,000 步，状态审计及最终预测独立复算全部通过；预设 +/-1 点平均精度等价仅 ResNet18 确认，YOLO、DETR、ViT 未确认。不将跑通训练改写为全模型精度等价或全量 ImageNet/COCO 收敛。

最新[2.10 版本对齐与差异定位](CROSS_VENDOR_DIAGNOSIS.md)追加 41 次完整训练：三模型及关闭库 DETR 的五种子对照均未确认 +/-1 点等价；同一权重两平台的 12 次评估指标一致。MetaX 卷积库开关已接入共享实现，但 C500 当前栈的卷积权重梯度样例在请求确定性后仍不重复，关闭库也不保证完整训练逐位可重复。RNG 状态相同不保证跨厂商同形状 Dropout 掩码相同。

2026-10-21 完成了 A800 的[五模型多种子训练对照](MULTISEED_ACCURACY.md)：Faster R-CNN 10 对平均 AP 通过预设等价检验但逐步检查不一致；ResNet18、YOLOv5-S、ViT-B/16 和外部官方单类 DETR 各 5 对的 loss、最终模型状态与指标相同。[早期短程对照](TRAINING_ACCURACY.md)及同版本漂移记录保留。以上不代表全量 ImageNet/COCO 收敛、全部模型配置或其他硬件已经验收。

| 能力 | NVIDIA / A800 | 昇腾 / 910B | 寒武纪 / MLU590 | 沐曦 / C500 |
| --- | --- | --- | --- | --- |
| 后端名 | `nvidia` | `ascend` | `cambricon` | `metax` |
| 设备名 | `cuda` | `npu` | `mlu` | `cuda` |
| 运行时供应者 | PyTorch CUDA | 延迟导入 `torch_npu` | 延迟导入 `torch_mlu` | 厂商 PyTorch 分发 |
| 默认通信传输配置 | `nccl`，双卡通过 | `hccl`，待真机核验 | `cncl`，待真机核验 | `nccl` 兼容接口，待真机核验 |
| 设备迁移、RNG、资源管理 | 共享 PyTorch 运行时，已测相关路径 | 已有运行时入口，未测 | 已有运行时入口，未测 | C500 固定用例通过，见运行时报告 |
| AMP / GradScaler | ResNet 短程 AMP 通过 | SDK AMP 入口，未测 | SDK AMP 入口，未测 | C500 ResNet AMP 路径可执行，未单独断言 scaler 是否跳步或验证全部精度/模型 |
| 2D NMS / IoU、IoF、IoS | 原 NVIDIA 扩展，通过对比 | 同设备 PyTorch 共享实现已注册，未测 | 同左 | 共享 NMS/IoU 的固定 FP16/FP32 用例通过；IoF/IoS 未单测 |
| Torchvision NMS | 延迟调用原 Torchvision 算子，通过对比 | 同设备共享实现，保留分数排序和 exclusive 坐标语义，未测 | 同左 | 共享实现的固定 FP32 用例通过 |
| RoIAlign / PSRoIAlign / PSRoIPool / DeformConv | 原扩展及 autograd，通过对比 | 需注册完整实现，当前缺失即报错 | 同左 | 同左，不自动继承 NVIDIA 二进制 |
| Sigmoid / Softmax / Cross Focal Loss | 原扩展及 autograd，通过对比 | 需注册实现 | 需注册实现 | 需注册实现 |
| BEV / 3D IoU 与 NMS | 原扩展，通过对比 | 需注册实现 | 需注册实现 | 需注册实现 |
| 点云查询 / RoIAware / RoIPoint 池化 | 原扩展，通过对比 | 需注册实现 | 需注册实现 | 需注册实现 |
| Numba 旋转框 IoU 评估 | NVIDIA evaluator 已路由，通过对比 | 需替换完整评估操作 | 同左 | 同左 |
| SyncBN | 共享实现，双卡训练通过 | 有共享入口，可覆盖，未测 | 同左 | 同左 |
| 稀疏卷积及张量操作 | 工厂与张量路由；spconv 未安装，未做真实稀疏执行 | 需提供依赖或覆盖操作 | 同左 | 同左 |
| linklink / 融合优化器 | 延迟供应者与路由存在；真实 SDK 未测 | 未注册供应者，使用即明确报错 | 同左 | 同左 |
| MQBench / MSBench | 延迟供应者存在，真实工具链未测 | 需显式提供并验证 | 同左 | 同左 |
| 真机状态 | 本轮通过有界验收，范围见报告 | 未验证 | 未验证 | C500 严格 FP32 功能及四模型固定预算真实训练完成；跨厂商精度等价仅 ResNet18 确认，非全量基准或多卡验收 |

通用 NMS 保留调用方已按分数排序的契约和 `ALIGNED_FLAG.offset`；通用 overlap 保留原扩展的 inclusive corner、空输出和分母截断语义，不调用 CPU 代算。共享实现以正确性和可扩展性为目标，国产设备性能仍需实际分析。

NVIDIA 已测变体见 `validation/native_comparison.final.json` 及 `validation/upstream_ops.pt` / `routed_ops.final.pt`。其中含 RoIAlign 对齐与非对齐、FP16/FP32；2D overlap 三种模式及 FP16/FP32；两种 NMS offset；RoIAware max/avg；DeformConv 三类输入梯度。不能把这些小样本检查解释为任意 dtype、形状、全部模型和数据集均已验证。

RoIPointPool3d 的原始实现没有反向，本轮不新增伪梯度。新增厂商必须保留返回元组、索引类型、坐标与对齐约定、reduction、精度和所需反向；有意不支持的变体须明确报错。原生 CPU 算子并不意味着其他 GPU 操作可自动在 CPU 回退。
