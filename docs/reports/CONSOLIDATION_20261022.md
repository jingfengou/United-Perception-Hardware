# 仓库收敛记录

整理基线：`a3e2752f454c72d6b46e0b64d0debeef0d7105a3`。这是目录、文档、策略边界和 CI 的收敛，不是模型算法或数值公式修改，也不恢复暂停的多随机种子任务。

## 已完成的结构整理

| 改动 | 目的与兼容边界 |
| --- | --- |
| 根 README 收敛为安装/选择硬件/运行/接入/限制 | 历史过程不再占据产品首屏；详细环境单独说明 |
| 41 个执行/诊断工具归 `benchmark/hardware`，5 个原维护工具归 `tools/platform` | 同名工具有[迁移表](../../benchmark/hardware/README.md)，UP 公开 Python 接口与 CLI 不删 |
| 12 份历史报告归 `docs/reports` | 当前平台手册与历史结论分开；旧材料链接固定到原提交 |
| 1,505 份证据从主工作树转为标签/Release/逐文件清单 | 已公开下载归档并验证全部文件；不改写历史、不丢失败阶段 |
| 六个原协议小夹具保留在 tests | 离线 CPU 测试无需下载 59 MB 归档，并与原清单核对 |
| common/context 只依赖策略接口，portable_v1 自管生命周期 | 默认 native 不导入实验，实验操作不污染全局 native 注册表 |
| 策略激活失败恢复模式和开关 | 避免异常残留改变后续 native 行为；未修改数学实现 |
| 静态检查扩展到实验依赖边界 | 禁止公共核心再次直接导入 experimental |
| 源码快照声明新目录布局，读取器显式兼容新旧两种布局 | 新平台/benchmark/csrc 不漏记；旧快照白名单和哈希校验保留，未知布局拒绝 |
| pytest 自动发现全部三处测试 | 包含此前遗漏的 MetaX 确定性、平台对照和证据检查 |
| CPU CI 覆盖原有 Torch 2.6 组合并增加 2.10；独立无 Torch 构建作业 | CPU 契约与公共安装分开，不将其作为 GPU 验收 |

## 证据保护

归档的每个成员均验证为原提交的 Git blob 原字节，避免 Windows CRLF 转换改变旧日志。公开 Release 下载件再次验证通过：1,505 文件，原内容 89,491,967 字节。下载与恢复见[归档索引](README.md)。

[受保护清单](../architecture/consolidation-protected.json)固定全部 UP 业务文件、模型配置、原 NVIDIA 内核、MetaX 卷积数学实现和已有 portable 数学模块。检查仅对声明的文本后缀归一化 CRLF，不改变算法、阈值、模型结构、优化配方或冻结协议。

旧 `.apply()`、配置和权重键不变；数值策略配置字符串不变。原私有 `_numerics` 只读入口继续供诊断脚本使用，但路由不再依赖它。工具文件的目录迁移不是改变历史报告的脚本哈希约定，严格历史复现仍需冻结源码。

按 Git 跟踪文件树统计，整理基线为 2,442 文件 / 95,391,045 字节；首个整理提交 `4569318` 为 959 文件 / 6,429,314 字节。后续小型修复/报告更新不改变目录规模。这里统计默认工作树的跟踪内容，不是 `.git` 历史大小。

## 审查项闭环

| 原编号 | 本轮状态 |
| --- | --- |
| G01 / G02 | 已整理产品入口、唯一当前手册、报告目录、归档及历史链接 |
| G03 | 已改自动发现并增加安装后 wheel、无 Torch 构建；实际 CI 结果见下表 |
| G04 | 已隔离 portable_v1 生命周期、RNG/操作覆盖/开关；native 禁止导入实验 |
| G05 | 旧入口保留并明确用途/目录迁移；暂不设弃用时间，不恢复或掩盖此前分析埋点、健康检查的已说明差异 |
| G06 | 已澄清 shared portable 与实验 portable_v1，不为改名破坏旧入口 |
| G07 | 已统一环境说明，仍不宣称未实测的版本组合受支持 |
| G08 / G09 | 硬件能力、AMP/多卡与性能验收继续保留未验证，不属于目录整理的完成证明 |
| G10 | 保留全局上下文固定用途的约束，独立部署分进程，不新增多上下文框架 |

原路线 A/B/C 的审计、目录归档、边界与 CI 收敛已执行；D 的新真机候选验收未执行，不因整理完成自动记为通过。

## 验证

本地检查已实际完成；当前不申请 GPU，也不把历史真机结果转记为整理后版本已通过。

| 检查 | 结果 |
| --- | --- |
| 全量 CPU/兼容/维护测试自动发现 | 294 passed，0 failed/skipped；另 3 个子测试通过，2 条既有 ONNX 弃用提示 |
| 安装 wheel、仓库外、无 UP 的独立包测试 | 30 passed，0 failed/skipped；同样 2 条 ONNX 提示 |
| 无 Torch/UP 公共构建与普通导入 | 两包 wheel/sdist 构建成功；安装平台 wheel 后以 `-I` 导入，Torch/UP 均未安装或加载 |
| 原生扩展源码分发 | 平台 sdist 保留全部 43 份 NVIDIA csrc 文件，公共构建不编译 |
| 模型与配置保护 | 665 个固定文件核对通过，`up/`、`configs/` 无 Git 内容差异 |
| 边界与契约 | 0 违规，31 成员 / 73 操作索引无漂移 |
| 归档与夹具 | Release 下载件 1,505 文件通过；6 个原字节夹具通过 |
| 当前文档本地链接 | 179 项通过；历史证据链接固定到原提交 |
| 云端 CI | 实现验收提交 `cca572f` 的三个 CPU 组合与独立 public-packaging 均实际成功，[运行记录](https://github.com/jingfengou/United-Perception-Hardware/actions/runs/39746051459) |

本地环境：Windows / Python 3.13.11 / Torch 2.10.0+cpu。无 Torch 构建使用独立虚拟环境；未替换 GPU 环境或厂商 PyTorch。构建产物保留在任务忽略目录 `dist/consolidation-20261022/build/`，不把生成文件混进源码仓库。

云端 JUnit 已下载核对：Python 3.10 / Torch 2.6、Python 3.12 / Torch 2.6、Python 3.12 / Torch 2.10 三组均为 297 个 XML case（294 测试 + 3 子测试），零失败/错误/跳过，仓库外安装后 wheel 各 30 项通过。三组均确认包含此前漏掉的 MetaX 确定性、平台对照、平台证据回归。验收后的提交仅补记本报告与策略名说明，不改运行时、工具或测试。

首次全量检查暴露的是已拆离实验注册导致的生成契约漂移；审查后重新生成两个索引，未改测试容差。归档构建还验证了 `core.autocrlf` 的影响，最终归档强制使用原 Git blob 字节并逐个匹配，未把换行转换后的副本冒充原始证据。

```bash
python tools/platform/check_project.py
python tools/platform/check_hardware_boundaries.py
python tools/platform/export_accelerator_contract.py --check
python tools/platform/export_platform_contract.py --check
python -m pytest -q
```

## 仍然保留的限制

- 当前结构的 NVIDIA/MetaX 新真机训练/训练 eval/独立推理未在这轮重跑；已有结论见[历史平台报告](platform/REAL_HARDWARE.md)。
- 昇腾/寒武纪、国产专用算子、AMP/多卡和可选 spconv/linklink/量化剪枝仍按原矩阵记未验证或未支持。
- MetaX 局部策略只覆盖已说明的一阶 FP32 场景；portable_v1 含 FP64 及性能代价，不把隔离成功当成精度或性能提升。
- 上游历史业务依赖与现代环境的兼容前置仍需按[环境说明](../platform/ENVIRONMENTS.md)准备。
- 全局上下文固定厂商与用途、旧门面暂不弃用；没有新增复杂多上下文或推理引擎。
- Git 历史不改写，归档只减少默认工作树，不声称普通完整克隆也变为小体积。

后续按“问题编号 → 契约 → 最小改动 → 测试 → 固定源码证据 → 能力矩阵 → 发布”继续；模型数学修复与厂商扩展另立阶段，不混入目录整理。
