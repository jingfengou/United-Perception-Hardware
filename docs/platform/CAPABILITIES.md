# 能力矩阵

以下记录注册边界及 **`ad36d77 → 71d9812` 的同平台重构回归**，完整协议、数值策略和失败见[本轮报告](../reports/REARCHITECTURE_HARDWARE_20261023.md)。73 项基础操作见[机器契约](interface_contract.json)；四模型执行成功不代表全部操作、dtype、AMP 或多卡均已验证。更早证据只对各自冻结快照成立。

当前代码按[LightX2V 风格总设计](../architecture/PLATFORM.md)组织。NVIDIA native、C500 显式 v2 均实际完成四模型单种子固定预算、训练内 eval、模型/优化器/调度器恢复和全部留出批次独立推理；全部 loss、最终权重、指标及预测跨版本逐位一致。不是全量大基准收敛或跨厂商精度等价证明。

前轮 [C500 GEMM 专项](../reports/METAX_GEMM_FIX_20261022.md)和[四模型旧快照](../reports/CURRENT_VALIDATION_20261022.md)继续独立保留；不升级 native 状态，也不重写 v1 失败。

| 厂商 / 设备 | 设备工厂 | 训练工厂 | 推理工厂 | 默认共享能力 | 本轮真机训练 | 本轮独立推理/导出 |
| --- | --- | --- | --- | --- | --- | --- |
| nvidia / cuda | 已迁移 | 已注册 | 已注册 | portable + NVIDIA 专用实现 | RTX 4090 native 四模型重构前后完整固定预算逐位一致；含训练内 eval/恢复；独立编译后 32 项原生操作通过 | 全部留出批次独立推理逐位一致；GPU 导出未复测 |
| ascend / npu | 已迁移，按需加载 torch_npu | 共享入口 | 共享入口 | portable；缺失专用算子明确报错 | 未验证 | 未验证 |
| cambricon / mlu | 已迁移，按需加载 torch_mlu | 共享入口 | 共享入口 | 同上 | 未验证 | 未验证 |
| metax / cuda | 已迁移，不加载 NVIDIA 二进制 | 共享入口 | 共享入口 | portable；数值选项必须显式选择 | C500 显式 v2 四模型重构前后完整预算逐位一致；native 短程仅 ViT 通过，其余三模型同版本重复也有差异 | v2 全部留出批次及同权重 native 首批/单图跨版本推理逐位一致；导出未复测 |
| cpu / cpu | 可执行 | 可执行 | 可执行 | 公共运行时/通信入口、同设备共享操作 | CPU 契约通过，不是全部模型训练 | CPU 契约/有限 ONNX 检查，不是全部模型 |

两侧共用 DeviceBackend 和明确 common 操作；AMP 的基础实现共享，GradScaler/DDP/训练优化器不可从 inference 获取。普通 PyTorch 层算力由厂商 Torch 提供，注册入口不证明所有普通算子、dtype 或第三方扩展都可用。

## 沐曦如何运行模型

平台边界迁移完成，不等于四家专用算子已补齐。执行路径分为三类：

- 普通 Conv、Linear、Attention 等直接使用厂商 PyTorch；不把所有 PyTorch 算子在平台包中重复实现。
- YOLO 的 UP NMS 接口经兼容门面路由到 `ops/nms/common.py`，张量仍在输入 GPU 上计算，不加载 NVIDIA 扩展。`portable` 在这里指公共实现，不是 `portable_v1` 数值实验。
- UP RoIAlign、DeformConv、部分 Focal/3D/点云等只有 NVIDIA 实现的操作，国产后端未补齐时明确报错。需要使用这些操作的模型不能据此视为可运行。

当前选择的 ResNet18、YOLOv5-S、普通 DETR R50 外部适配器和 ViT 主要依赖前两类路径；普通 DETR 不等于 Deformable DETR。MetaX 显式 v2 的卷积权重梯度与 mm 另外使用 `ops/convolution/metax/`、`ops/mm/metax/`，默认 native 不启用它们。

接入新厂商时，兼容的公共实现可直接复用；其余必需操作必须增加 `ops/<类别>/<厂商>/` 实现并注册，训练可微操作同时交付反向与测试。不是创建一个厂商目录就算完成，也不必为每家复制所有公共代码。具体步骤见[新增厂商](ADDING_VENDOR.md)。

## 原生和可选能力

| 类别 | 默认供应者 | 范围与限制 |
| --- | --- | --- |
| NMS / 2D overlap | portable，NVIDIA 有原生覆盖 | 同输入设备，不新增 CPU 回退；dtype/变体未声明不等于全支持 |
| RoI/DeformConv/Focal/3D/点云 | NVIDIA 完整实现迁移 | 国产缺失项必须实现；Focal Loss 仅 training；不可借用另一用途专用实现 |
| SyncBN | portable/shared forward，训练函数 training | 完整多卡语义仍需真机；原参数和 running stats 保留 |
| sparse.* | shared 门面 + runtime sparse 供应者 | 无厂商 SDK 时明确失败；不包装重命名参数 |
| distributed.* | common PyTorch 通信；wrap_model 在 training | 通信传输、DDP、对象组实际能力分别验证 |
| legacy_link / 融合优化器 / 工具库 | NVIDIA 可选 SDK | 延迟加载；未执行，不声称当前结构已验证 |
| ONNX | InferenceBackend.export_onnx | 保留 legacy symbolic；未有 symbolic 的操作不承诺导出，无新引擎 |
| portable_v1 | experimental/numerics | native 默认关闭；仅已有显式 CUDA-shaped 厂商声明，含 GPU FP64 和性能代价；不提升精度验收结论 |
| metax_deterministic_v1 | ops/convolution/metax/weight_gradient.py | 显式 C500 FP32 卷积权重梯度替换；原前向、输入/偏置梯度及 RNG 不变，无 FP64/CPU 代算；短程结果见[专项报告](../reports/platform/METAX_DETERMINISM.md)。[前轮完整预算](../reports/CURRENT_VALIDATION_20261022.md)三模型通过、DETR 同版本亦不稳定，不能保证全流程确定性或提升 native 支持等级 |
| metax_deterministic_v2 | ops/mm/metax/ | 复用 v1 卷积实现，另以厂商 Triton 固定 GPU IEEE FP32 mm；[前轮 DETR 专项](../reports/METAX_GEMM_FIX_20261022.md)及[本轮四模型重构回归](../reports/REARCHITECTURE_HARDWARE_20261023.md)分别记录。默认关闭且有性能代价；不是 v1/跨厂商等价、全形状或 AMP/DDP/导出通过 |

## 历史证据与本轮结论

- 历史 NVIDIA 算子、AMP、双卡和代表模型结果见[原验证报告](../reports/hardware/VALIDATION.md)，多种子回归见[五模型报告](../reports/hardware/MULTISEED_ACCURACY.md)。
- C500 旧快照的有限功能和真实数据对照见[沐曦报告](../reports/hardware/METAX_VALIDATION.md)及[跨厂商报告](../reports/hardware/CROSS_VENDOR_ACCURACY.md)。三模型精度差异和实验修复保留原结论。
- 昇腾/寒武纪此前没有真机证据，本轮也没有。
- 历史目录/策略整理没有重跑 GPU；本轮另建了单种子验证阶段，不恢复已暂停的多随机种子任务、不重写旧 manifest、不把 CPU mock 当硬件验收。训练内 eval 保持 training 用途，独立推理固定 inference 用途。

capabilities 的 hardware_validated 默认 False；registered 是注册存在，dependencies_available 是按需检查的声明依赖可加载性，executed 是成功调用次数。只有明确记录硬件、版本、用途、变体、容差与证据后才能在文档中提升对应验证状态。
