# 验证报告与证据归档

当前开发文档见[平台手册](../platform/README.md)。历史报告只对文内冻结的源码、配置、环境和预算成立，不把历史精度、CPU 契约或注册状态自动转记为当前版本真机通过。

## 当前整理

- [2026-10-22 厂商入口简化](VENDOR_SIMPLIFICATION_20261022.md)：设备类直接注册、公共执行复用、旧地址兼容和验证。
- [2026-10-22 仓库收敛记录](CONSOLIDATION_20261022.md)：目录、策略隔离、CI、验证和未覆盖范围。
- [整体设计与原版 UP 差异](../architecture/PROJECT_REVIEW.md)：固定版本的逐文件变更台账。

## 历史报告

| 阶段 | 报告 | 边界 |
| --- | --- | --- |
| 首轮 NVIDIA | [算子/模型/AMP/双卡](hardware/VALIDATION.md) | 旧路由结构、指定 A800 环境 |
| NVIDIA 真实数据 | [训练对照](hardware/TRAINING_ACCURACY.md)、[五模型多种子](hardware/MULTISEED_ACCURACY.md) | 原版/路由版及固定小数据预算，不是全量 ImageNet/COCO |
| C500 接入 | [环境与有限功能](hardware/METAX_VALIDATION.md) | 厂商 Docker 和显式精度设置 |
| 跨厂商精度 | [40 次训练](hardware/CROSS_VENDOR_ACCURACY.md)、[环境与随机性定位](hardware/CROSS_VENDOR_DIAGNOSIS.md) | 多模型精度等价未全部通过 |
| 实验数值策略 | [协议](hardware/PORTABLE_NUMERICS.md)、[修复阶段](hardware/CROSS_VENDOR_FIX.md) | NVIDIA 最终完整复测暂停，含 FP64 与额外开销 |
| 平台拆包 | [CPU/构建](platform/VALIDATION.md)、[短程真机](platform/REAL_HARDWARE.md) | training / 训练 eval / 独立 inference；保留未通过项 |
| C500 局部修复 | [卷积非确定性](platform/METAX_DETERMINISM.md) | 同厂商、同种子短程通过，不代表跨厂商全量精度等价 |

历史结果没有因整理而重跑、覆盖或挑选。旧多随机种子任务未恢复。

## 原始证据

整理前两个 validation 目录中的 **1,505 个已跟踪文件 / 89,491,967 字节**已完整归档，包括失败、未完成阶段、日志、原始 JSON、源码快照和索引。

- [固定 Git 标签](https://github.com/jingfengou/United-Perception-Hardware/tree/evidence-pre-consolidation-20261022)：指向 `a3e2752f454c72d6b46e0b64d0debeef0d7105a3`，包含整理前完整仓库。
- [Release 下载](https://github.com/jingfengou/United-Perception-Hardware/releases/tag/evidence-pre-consolidation-20261022)：归档与同名 manifest，下载后已实际逐文件校验。
- [主分支逐文件清单](evidence.manifest.json)：原路径、大小、SHA256 和 Git blob ID。
- 归档文件：`evidence-pre-consolidation-20261022.tar.gz`，59,190,175 字节。
- SHA256：`bdb2886358d6ba67039b13147fc5da279bcb86985cf9d2d9a986ceb798d233ae`。

归档保存所有原先已跟踪的证据，不包含原本仅保留在服务器的大权重或数据；它们原有的路径、大小和校验索引仍在。现有历史链接固定到归档提交，可直接查看单个文件，不要求先下载全部归档。

```bash
curl -fL -o /tmp/up-evidence.tar.gz \
  https://github.com/jingfengou/United-Perception-Hardware/releases/download/evidence-pre-consolidation-20261022/evidence-pre-consolidation-20261022.tar.gz
python tools/platform/archive_evidence.py verify \
  --archive /tmp/up-evidence.tar.gz --manifest docs/reports/evidence.manifest.json
mkdir -p /tmp/up-evidence
tar -xzf /tmp/up-evidence.tar.gz -C /tmp/up-evidence
```

主分支只保留六个原字节协议夹具供离线 CPU 测试，位置为 `tests/fixtures/protocols/`，CI 按归档清单校验。它们不是重新制定的训练协议。Git 历史未改写，默认工作树更小不代表历史克隆体积自动缩小。
