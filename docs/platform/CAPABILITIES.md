# 能力矩阵

以下记录注册边界及 **`f78e27a` 的有限真机回归**，完整协议和失败见[本轮报告](../reports/CURRENT_VALIDATION_20261022.md)。73 项基础操作见[机器契约](interface_contract.json)；四模型执行成功不代表全部操作、dtype、AMP 或多卡均已验证。更早证据只对各自冻结快照成立。

当前代码已按[LightX2V 风格总设计](../architecture/PLATFORM.md)重构。这里只更新实现位置，不将原冻结版 GPU 结果升级为新架构真机验收。

后续 [C500 GEMM 专项](../reports/METAX_GEMM_FIX_20261022.md)在显式 v2 策略下验证 DETR 四份单种子完整预算、训练 eval/恢复及独立推理一致；它不升级下表的 native 状态，也不重写 v1 失败。

| 厂商 / 设备 | 设备工厂 | 训练工厂 | 推理工厂 | 默认共享能力 | 本轮真机训练 | 本轮独立推理/导出 |
| --- | --- | --- | --- | --- | --- | --- |
| nvidia / cuda | 已迁移 | 已注册 | 已注册 | portable + NVIDIA 专用实现 | RTX 4090 四模型单种子完整固定预算与原版 UP 一致；含训练内 eval/恢复；32 项原生操作通过 | 简化前后四模型独立推理通过；GPU 导出未复测 |
| ascend / npu | 已迁移，按需加载 torch_npu | 共享入口 | 共享入口 | portable；缺失专用算子明确报错 | 未验证 | 未验证 |
| cambricon / mlu | 已迁移，按需加载 torch_mlu | 共享入口 | 共享入口 | 同上 | 未验证 | 未验证 |
| metax / cuda | 已迁移，不加载 NVIDIA 二进制 | 共享入口 | 共享入口 | portable；数值选项必须显式选择 | C500 native 短程仅 ViT 数值通过；局部卷积策略完整预算 ResNet/YOLO/ViT 一致，DETR 未通过 | 四模型同权重 native 独立推理逐位一致；导出未复测 |
| cpu / cpu | 可执行 | 可执行 | 可执行 | 公共运行时/通信入口、同设备共享操作 | CPU 契约通过，不是全部模型训练 | CPU 契约/有限 ONNX 检查，不是全部模型 |

两侧共用 DeviceBackend 和明确 common 操作；AMP 的基础实现共享，GradScaler/DDP/训练优化器不可从 inference 获取。普通 PyTorch 层算力由厂商 Torch 提供，注册入口不证明所有普通算子、dtype 或第三方扩展都可用。

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
| metax_deterministic_v1 | ops/convolution/metax/weight_gradient.py | 显式 C500 FP32 卷积权重梯度替换；原前向、输入/偏置梯度及 RNG 不变，无 FP64/CPU 代算；短程结果见[专项报告](../reports/platform/METAX_DETERMINISM.md)。[本轮完整预算](../reports/CURRENT_VALIDATION_20261022.md)三模型通过、DETR 同版本亦不稳定，不能保证全流程确定性或提升 native 支持等级 |
| metax_deterministic_v2 | ops/mm/metax/ | 复用 v1 卷积实现，另以厂商 Triton 固定 GPU IEEE FP32 mm；DETR 同策略四份 720 步、恢复和 50 图独立推理一致。默认关闭；不是 v1/跨厂商等价、全形状或 AMP/DDP/导出通过，见[完整验收](../reports/METAX_GEMM_FIX_20261022.md) |

## 历史证据与本轮结论

- 历史 NVIDIA 算子、AMP、双卡和代表模型结果见[原验证报告](../reports/hardware/VALIDATION.md)，多种子回归见[五模型报告](../reports/hardware/MULTISEED_ACCURACY.md)。
- C500 旧快照的有限功能和真实数据对照见[沐曦报告](../reports/hardware/METAX_VALIDATION.md)及[跨厂商报告](../reports/hardware/CROSS_VENDOR_ACCURACY.md)。三模型精度差异和实验修复保留原结论。
- 昇腾/寒武纪此前没有真机证据，本轮也没有。
- 历史目录/策略整理没有重跑 GPU；本轮另建了单种子验证阶段，不恢复已暂停的多随机种子任务、不重写旧 manifest、不把 CPU mock 当硬件验收。训练内 eval 保持 training 用途，独立推理固定 inference 用途。

capabilities 的 hardware_validated 默认 False；registered 是注册存在，dependencies_available 是按需检查的声明依赖可加载性，executed 是成功调用次数。只有明确记录硬件、版本、用途、变体、容差与证据后才能在文档中提升对应验证状态。
