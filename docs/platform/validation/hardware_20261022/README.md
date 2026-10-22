# 2026-10-22 短程真机证据

阅读入口：[结论、边界和复现方法](../../REAL_HARDWARE.md)。本目录不是多随机种子精度验收，也不更新旧冻结精度队列。

| 证据 | 内容 |
| --- | --- |
| [local.json](local.json) | 发布汇总、260 项 CPU 测试、源码/工具/wheel 哈希、通过与未通过项 |
| [runtime-source-manifest.json](runtime-source-manifest.json) | 实际执行平台/UP/配置源码的归一化清单，两平台相同 |
| [CPU JUnit](cpu-junit.xml) | 本轮全部 CPU 回归 |
| [NVIDIA 最终比较](nvidia/run-03/summary.json) | 四模型 loss、梯度、checkpoint、eval 和独立推理比较 |
| [NVIDIA 原生操作](nvidia/native-ops-comparison.json) | 新旧扩展重编译后的 32 项前向/所需梯度比较 |
| [沐曦最终比较](metax/run-03/summary.json) | 功能均可运行，但 ResNet/YOLO/DETR 严格比较有失败 |
| [沐曦旧版自身重复](metax/repeat-controls.json) | 与拆分前后对照使用相同种子、输入、RNG、环境的重复控制 |
| [沐曦同权重推理](metax/same-weights/summary.json) | 四模型同一旧版 checkpoint 在新旧推理路径逐位相同 |
| [NVIDIA 文件索引](nvidia-artifact-index.json)、[沐曦文件索引](metax-artifact-index.json) | 原始文本、远端保留的大体积张量文件大小/SHA256、源码与图片完整性审计 |
| [NVIDIA 原始证据包](nvidia-evidence-v2.tar.gz)、[沐曦原始证据包](metax-evidence.tar.gz) | 与可读展开目录相同的原始日志、配置、报告和执行驱动 |

`run-01` 保留全部最初训练及失败推理，`run-02/03` 只补推理并引用已完成的训练证据，未改原 checkpoint。最终目录中复用的文件以 summary 的 `reused_training/reused_inference` 指针及 SHA256 为准；展开包不复制链接指向的大体积文件。

每份 `run-01/{baseline,changed}/{model}/training/report.json` 包含环境、loss、梯度/状态摘要、两次 eval 的状态保护和用途审计。`inference/report.json` 记录独立图片预测、同 batch 的训练验证/推理比较及 checkpoint 身份；失败报告保留完整异常。

证据包内 `tools/` 是实际使用过的驱动、补跑和审计脚本快照。发布的 `tools/validate_platform_equivalence.py` 补齐了初次推理配置、tensor-only 权重、坐标配置及子集 manifest 汇总字段。请用每份报告记录的驱动哈希识别版本，勿假定当前工具字节与每个最初失败进程相同。

模型算法源文件 202 个内容未变；43 个 NVIDIA 原生源码仅移动路径。五个其他任务 Runner 的差异是前次结构迁移时的训练验证状态保护，列在索引中，未在本轮执行其完整流程。

大体积权重、梯度和预测 `.pt` 留在两台服务器本任务专用目录，Git 只保存哈希索引；不包含数据集图片、私钥或访问凭据。`parent_summary_fields_retained` 标记初版子集 manifest 沿用父集合汇总字段的问题，选中元数据/图片/批次已另行验证，原文件未被改写。
