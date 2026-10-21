# 能力矩阵

本表描述仓库实现及本轮验证状态，不是厂商产品的完整支持矩阵。运行时可查询 `up.accelerator.capabilities()`，注册状态不等于 SDK 可执行状态或真机验证状态。

2026-10-21 补充了[五模型多种子训练对照](MULTISEED_ACCURACY.md)：Faster R-CNN 10 对平均 AP 通过预设等价检验但逐步检查不一致，ResNet18/ViT 各 5 对一致，其他项仍按报告进度记录。[早期短程对照](TRAINING_ACCURACY.md)及同版本漂移记录保留。表中算子或功能测试通过不应解读为所有模型精度已完成验收。

| 能力 | NVIDIA / A800 | 昇腾 / 910B | 寒武纪 / MLU590 | 沐曦 / C500 |
| --- | --- | --- | --- | --- |
| 后端名 | `nvidia` | `ascend` | `cambricon` | `metax` |
| 设备名 | `cuda` | `npu` | `mlu` | `cuda` |
| 运行时供应者 | PyTorch CUDA | 延迟导入 `torch_npu` | 延迟导入 `torch_mlu` | 厂商 PyTorch 分发 |
| 默认通信传输配置 | `nccl`，双卡通过 | `hccl`，待真机核验 | `cncl`，待真机核验 | `nccl` 兼容接口，待真机核验 |
| 设备迁移、RNG、资源管理 | 共享 PyTorch 运行时，已测相关路径 | 已有运行时入口，未测 | 已有运行时入口，未测 | 已有共享运行时入口，未测 |
| AMP / GradScaler | ResNet 短程 AMP 通过 | SDK AMP 入口，未测 | SDK AMP 入口，未测 | 共享 AMP 入口，未测 |
| 2D NMS / IoU、IoF、IoS | 原 NVIDIA 扩展，通过对比 | 同设备 PyTorch 共享实现已注册，未测 | 同左 | 同左 |
| Torchvision NMS | 延迟调用原 Torchvision 算子，通过对比 | 同设备共享实现，保留分数排序和 exclusive 坐标语义，未测 | 同左 | 同左 |
| RoIAlign / PSRoIAlign / PSRoIPool / DeformConv | 原扩展及 autograd，通过对比 | 需注册完整实现，当前缺失即报错 | 同左 | 同左，不自动继承 NVIDIA 二进制 |
| Sigmoid / Softmax / Cross Focal Loss | 原扩展及 autograd，通过对比 | 需注册实现 | 需注册实现 | 需注册实现 |
| BEV / 3D IoU 与 NMS | 原扩展，通过对比 | 需注册实现 | 需注册实现 | 需注册实现 |
| 点云查询 / RoIAware / RoIPoint 池化 | 原扩展，通过对比 | 需注册实现 | 需注册实现 | 需注册实现 |
| Numba 旋转框 IoU 评估 | NVIDIA evaluator 已路由，通过对比 | 需替换完整评估操作 | 同左 | 同左 |
| SyncBN | 共享实现，双卡训练通过 | 有共享入口，可覆盖，未测 | 同左 | 同左 |
| 稀疏卷积及张量操作 | 工厂与张量路由；spconv 未安装，未做真实稀疏执行 | 需提供依赖或覆盖操作 | 同左 | 同左 |
| linklink / 融合优化器 | 延迟供应者与路由存在；真实 SDK 未测 | 未注册供应者，使用即明确报错 | 同左 | 同左 |
| MQBench / MSBench | 延迟供应者存在，真实工具链未测 | 需显式提供并验证 | 同左 | 同左 |
| 真机状态 | 本轮通过有界验收，范围见报告 | 未验证 | 未验证 | 未验证 |

通用 NMS 保留调用方已按分数排序的契约和 `ALIGNED_FLAG.offset`；通用 overlap 保留原扩展的 inclusive corner、空输出和分母截断语义，不调用 CPU 代算。共享实现以正确性和可扩展性为目标，国产设备性能仍需实际分析。

NVIDIA 已测变体见 `validation/native_comparison.final.json` 及 `validation/upstream_ops.pt` / `routed_ops.final.pt`。其中含 RoIAlign 对齐与非对齐、FP16/FP32；2D overlap 三种模式及 FP16/FP32；两种 NMS offset；RoIAware max/avg；DeformConv 三类输入梯度。不能把这些小样本检查解释为任意 dtype、形状、全部模型和数据集均已验证。

RoIPointPool3d 的原始实现没有反向，本轮不新增伪梯度。新增厂商必须保留返回元组、索引类型、坐标与对齐约定、reduction、精度和所需反向；有意不支持的变体须明确报错。原生 CPU 算子并不意味着其他 GPU 操作可自动在 CPU 回退。
