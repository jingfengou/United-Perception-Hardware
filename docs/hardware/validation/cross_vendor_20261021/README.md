# 跨厂商训练证据

正文、边界与复现入口见 [CROSS_VENDOR_ACCURACY.md](../../CROSS_VENDOR_ACCURACY.md)。

- `plan.json`：启动前固定的四模型、五种子、预算和等价判据。
- `summary.json` / `pairs.csv`：全部 20 对种子，差值方向为沐曦减 NVIDIA，单位百分点。
- `coverage.json`：40 次/76,000 步、40 份状态审计、40 份独立指标复算及源代码核对。
- `statistics-recheck.json`：独立 SciPy 单侧检验与汇总 TOST 的核对。
- `nvidia-reproduction.json`：本轮 20 次 NVIDIA 训练与上一轮对应 NVIDIA 路由版逐位复现。
- `nvidia/` / `metax/`：供直接浏览的 JSON/XML 镜像，含每个种子的 `run-summary.json`、有效配置、状态审计、指标复算与环境；完整 `report.json` 的逐步轨迹与大清单保留在压缩证据包，摘要提供其成员路径和原始哈希。
- `*-evidence.tar.gz` / `*.index.json`：完整日志、最终预测、标签、报告及逐文件 SHA-256。NVIDIA 包 146 个文件，沐曦包 148 个文件；所有成员在下载后逐项核验。
- `hardware-final.log` / `metax-hardware-final.log`：结束时无 GPU 计算进程。
- `scheduling.json`：物理 GPU 分配、沐曦自建容器 CPU 配额恢复、约 260 秒的服务器时钟偏差。

解压完整证据后可从预测和包内标签离线复算，不需要显卡或图片：

```bash
tar -xzf nvidia-evidence.tar.gz
tar -xzf metax-evidence.tar.gz
python /path/to/repo/tools/recompute_cross_vendor_metrics.py \
  --runs metax/results/*-seed*/ --data-root metax/data --output metax-reevaluated.json
```

分类需 PyTorch CPU 张量计算，检测需标准 pycocotools 2.0.10。`all_models_equivalent` 为 `false`；只有 ResNet18 达到预设 +/-1 点平均精度等价。其余未确认结果没有被丢弃或重新选择种子。完整 checkpoint 和数据图片保留在服务器任务目录，不进入证据包或公共 Git 仓库。
