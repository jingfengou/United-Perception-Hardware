# 诊断证据的范围

- [summary.json](summary.json)核对三组短程对照、GPU 数学用例、随机流协议及修复前的逐算子分叉，不声明完整训练精度等价。
- `evidence.tar.gz` 包含 25 份原始 JSON：卷积前后向 provider/profiler、Dropout 舍入对照、共同算子完整哈希、RNG 恢复、短程模型报告和逐算子 trace；`evidence.tar.gz.index.json` 提供归档、成员和收集工具 SHA-256。不包含 tensor payload、模型权重、数据图片、SDK 或凭据。
- `*-v8-source.tar.gz` 是 DETR/ViT 进一步归约修复前的源码；`*-v10-source.tar.gz` 是 DETR 64 步、ViT 32 步对照使用的源码。两平台同阶段规范化 UP manifest 一致，原始成员哈希分别保留。
- 最终完整复测使用 `../fixed-order/*-source.tar.gz` 的 v11 源码。v10 到 v11 的 UP 改动仅为 `numerics.py` / `runtime.py` 中的能力错误后端上下文；共享数值 kernel 文件哈希全部不变。仍保留各自实际受测版本，不把短程报告改写成 v11 运行。
- YOLO 4 步对照使用 v8；早期 YOLO 首处分叉 trace 来自更早候选版本，没有零符号归一字段，因此只标为原始字节差异。DETR/ViT 新 trace 同时提供 FP32 零符号归一哈希，区分 `+0/-0` 与非零值差异。CPU profiler event 名称不是 CPU 数值回退证据，另有实际 accelerator event 列表。
- v4/v5 profiler 报告的源码阶段来自执行命令记录，不带内嵌源码 manifest，不能冒充与最终冻结版同级别的源码绑定。RNG 单元恢复验证也不冒充完整 Runner 中断续训的连续性验收。

[最终 GPU 数学报告](../fixed-order/nvidia-math-contracts.json)及[沐曦报告](../fixed-order/metax-math-contracts.json)包含 15 个用例的完整输出/梯度哈希，并与各自平台原生 FP64 参考按明确容差比较。两边的输入及完整输出/梯度哈希均相同，仅针对受测用例，不推广为任意输入或所有算子的逐位保证。CPU oracle 仅用于验证，生产运算仍在 GPU。

重新打包可使用 `tools/package_portable_diagnostics.py --root PHASE_ROOT --working RAW_JSON_DIRECTORY`，输入文件名见该工具的显式清单；它校验源文件快照及 trace 对应关系，并拒绝覆盖已有归档或摘要。`evidence-tools.xml` 记录 12 项本地离线工具完整性测试，与两平台各 202 项运行时契约分开统计。
