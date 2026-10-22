# 能力矩阵

以下分别记录**包拆分后的注册状态与有限真机结果**。73 项当前注册见[机器契约](interface_contract.json)。历史结果属于各自冻结快照，不能自动转记为当前源码通过；本次短程对照见[真机报告](REAL_HARDWARE.md)。

| 厂商 / 设备 | 设备工厂 | 训练工厂 | 推理工厂 | 默认共享能力 | 当前结构真机训练 | 当前结构独立推理/导出 |
| --- | --- | --- | --- | --- | --- | --- |
| nvidia / cuda | 已迁移 | 已注册 | 已注册 | portable + NVIDIA 专用实现 | A800 四模型各 3 步，含训练内 eval/恢复，数值通过；32 项原生操作通过 | 四模型独立推理通过；GPU 导出未复测 |
| ascend / npu | 已迁移，按需加载 torch_npu | 共享入口 | 共享入口 | portable；缺失专用算子明确报错 | 未验证 | 未验证 |
| cambricon / mlu | 已迁移，按需加载 torch_mlu | 共享入口 | 共享入口 | 同上 | 未验证 | 未验证 |
| metax / cuda | 已迁移，不加载 NVIDIA 二进制 | 共享入口 | 共享入口 | portable；可显式选择既有 experimental numerics | C500 四模型功能通过；native 严格数值比较仅 ViT 通过，三模型有差异 | 四模型独立推理通过，同权重新旧输出逐位相同；导出未复测 |
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
| metax_deterministic_v1 | backends/metax/determinism.py | 显式 C500 FP32 卷积权重梯度替换；原前向、输入/偏置梯度及 RNG 不变，无 FP64/CPU 代算；ResNet/YOLO/DETR 同种子三步新旧及重复检查通过，见[专项报告](METAX_DETERMINISM.md)；不提升 native 或完整精度结论 |

## 历史证据与本轮结论

- 历史 NVIDIA 算子、AMP、双卡和代表模型结果见[原验证报告](../hardware/VALIDATION.md)，多种子回归见[五模型报告](../hardware/MULTISEED_ACCURACY.md)。
- C500 旧快照的有限功能和真实数据对照见[沐曦报告](../hardware/METAX_VALIDATION.md)及[跨厂商报告](../hardware/CROSS_VENDOR_ACCURACY.md)。三模型精度差异和实验修复保留原结论。
- 昇腾/寒武纪此前没有真机证据，本轮也没有。
- 本次只做短程真机回归，不恢复已暂停的多随机种子精度任务，不重写旧冻结 manifest，不把 CPU mock 当硬件验收。本地检查见[验收记录](VALIDATION.md)。

capabilities 的 hardware_validated 默认 False；registered 是注册存在，dependencies_available 是按需检查的声明依赖可加载性，executed 是成功调用次数。只有明确记录硬件、版本、用途、变体、容差与证据后才能在文档中提升对应验证状态。
