# 验证报告

验证日期：2026-10-20。上游基线：`52aaeb9ed27a490820696d7754d19f98cd6ecd11`。本目录保存原始验证报告、数值快照和源代码哈希；根目录 README 提供安装、环境选择和厂商接入说明。

本轮完成了中间层和全仓上层迁移，并对 NVIDIA 做有界验收。昇腾 910B、寒武纪 MLU590、沐曦 C500 的状态是运行时入口和共享实现已接入、专用缺失能力明确报错、尚未真机验证，不能解读为三个厂商已完整支持现有全部模型。

以上是 2026-10-20 的历史状态。2026-10-21 沐曦 C500 已追加[严格 FP32 四模型单卡功能检查](METAX_VALIDATION.md)，保留默认 TF32 的失败对照；没有因此获得全量精度、多卡或全部算子的支持声明。以下首轮表格仍按原验证日期保留。

2026-10-21 又完成了[沐曦/NVIDIA 真实数据跨厂商对照](CROSS_VENDOR_ACCURACY.md)：沿用用户选择的 CIFAR-10/PennFudan 完整预算，四模型、每个平台五个种子，共 40 次/76,000 步，40 份保存状态审计和最终预测独立复算全部通过。预设 +/-1 点平均精度等价仅 ResNet18 确认，YOLO、DETR、ViT 未确认；两平台 CPU 契约/协议/证据测试各 164 项通过。下文未测描述保留为首轮历史，不替代这份最新分模型结论。

2026-10-21 新增了独立原版/改造版进程的真实数据 Runner 对照，最新见[五模型多随机种子报告](MULTISEED_ACCURACY.md)，早期结果保留在[短程训练报告](TRAINING_ACCURACY.md)。它与本页保存的首轮合成检查分别记录，不将小规模数据回归解读为 ImageNet/COCO 全量收敛验收。

追加的五模型预算现已完成：主要对照 66 次、94,720 步（含 6 次同版本重复），4 个模型各 5 对一致，Faster R-CNN 的平均 AP 等价但非逐步一致。另保留 10 次初版 DETR 失败微调配方的诊断。最新 CPU 契约/协议测试 117 项通过，代码提交的 Python 3.10/3.12、CPU PyTorch 2.6 GitHub CI 也已通过；本页以下表格仍是 2026-10-20 的首轮历史记录。

## 实际执行

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| 全仓 Python 解析 | 446 个文件通过 | 本地 `ast.parse`，含应用、中间层、工具和测试 |
| 硬件边界 AST 检查 | 0 违规 | `upstream_inventory.json` 和汇总报告 |
| CPU 契约测试 | 52 通过、0 失败 | `validation/unit.final.xml`、`unit.final.log` |
| 原生扩展 | 上游、改造版分别编译 | `validation/baseline_build.log`、`changed_build.final.log` |
| NVIDIA 算子前向、梯度 | 32 个上游/改造对比样例通过，无上游数值检查缺口 | `validation/native_comparison.final.json`、两份 `.pt` 快照 |
| 代表模型 | ResNet18、Faster R-CNN、PointPillars、HRNet18 通过 | `validation/models.final.json` |
| 混合精度 | ResNet18 autocast、GradScaler 训练通过 | 同上 |
| 双卡 | NCCL all-reduce、DDP 参数一致、SyncBN 训练、对象广播、Gloo 对象 all-gather 通过 | `validation/distributed.final.json` |
| ONNX 符号 | 7 个成功导出样例的节点及属性与上游相同；另有 1 个上游 CPU 导出缺口 | `validation/onnx_comparison.json` |
| 无 CUDA 公共构建 | 未安装 PyTorch 的本地 Python 3.13.11 构建 wheel、sdist 成功 | `validation/local_build.json` |
| 代码一致性 | 本地与服务器受测代码逐文件 SHA-256 核对 | `validation/source_match.json`、`summary.json` |

CPU 测试在 A800-1 的 PyTorch 环境中隐藏 CUDA 设备执行。公共构建在 Windows、本地未安装 PyTorch 的环境执行。本报告采集时，Python 3.10/3.12、CPU PyTorch 2.6 的 CI 矩阵尚未运行；发布后的云端执行结果以仓库 Actions 记录为准。

## 硬件与环境

服务器：SSH 别名 `A800-1`，四张 NVIDIA A800-SXM4-80GB。本轮数值算子和 ONNX 使用物理 GPU 1，代表模型使用 GPU 0，双卡通信使用物理 GPU 2/3。Python 3.12.3，PyTorch `2.8.0a0+5228986c39.nv25.06`，CUDA 12.9，编译架构 `sm_80`，`MAX_JOBS=2`。

任务环境位于 `/workspace/up-hardware-routing-20261020/`，包含独立 `baseline`、`changed`、`venv`、`results`。虚拟环境复用服务器既有系统包，额外依赖仅安装在任务环境；未更改服务器全局 PyTorch。环境版本、设备信息、原生源码/二进制哈希和报告哈希记录在 `validation/summary.json`。

原生 C++/CUDA 源码保持上游内容；跨 Windows/Linux 比对源码哈希时只归一化 CRLF/LF。源代码逐文件清单采用实际文件字节；本地与服务器哈希一致性独立检查，不以文件名或 Git 提交号替代。

发布时另以 Git blob 对照受测快照，结果见 `validation/git_source_match.json`：449 个文件中 165 个字节完全一致，284 个仅有 Windows CRLF 与 Git LF 行尾差异，没有其他内容差异。原始快照哈希和行尾核对记录分别保留，不能把不同平台的原始文件字节哈希直接混为一项检查。该记录指向功能提交；后续仅修改说明或补充验证记录的提交不改变相应代码内容。

## 检查范围与限制

32 个数值样例覆盖 Torchvision NMS 旧门面、2D NMS 两种 offset、Softer NMS、IoU/IoF/IoS 的 FP16/FP32、空 overlap、RoIAlign 对齐开关和 FP16/FP32、PSRoIAlign、PSRoIPool、DeformConv 输入/偏移/权重梯度、三种 Focal Loss、3D/BEV IoU 和 NMS、点在框内查询、RoIAware max/avg、RoIPoint 池化及 Numba 旋转 IoU 评估。可微操作比较适用的输入梯度；非可微操作不声明梯度支持。

基线算子使用独立编译的上游扩展，在独立进程中加载原始算子源文件；测试加载器隔离无关应用依赖。比较容差为 `rtol=1e-4, atol=1e-5`。这不是完整上游应用的逐步执行对比。

代表模型使用合成输入，每个执行两步训练，检查 loss 和已有梯度有限、执行推理、序列化并恢复模型和优化器、比较恢复前后推理。恢复使用 `ModelHelper.cuda(device=torch.device(...))` 兼容入口。还加载原始 `ModelHelper` 类，使用当前注册的组件检查参数 key 与输出兼容。输入尺寸、点云网格等缩小，配置改动明确记录于 JSON。这些检查不等于完整数据集、完整 Runner、精度收敛、性能或长期稳定性验收。

ONNX 检查覆盖四个自定义算子的 GPU 导出，以及 RoIAlign、PSRoIPool、DeformConv 的 CPU 导出，比较节点类型、domain 和属性。上游 PSRoIAlign CPU 导出引用不存在的 `forward_cpu`；本轮改为说明后端和原因的错误，保留其 GPU 符号。不执行导出的 ONNX 模型，也不承诺任意动态形状或新版 dynamo 导出器兼容。

linklink 延迟层继承、稀疏供应者及外来稀疏张量表示使用 mock 契约测试；环境未安装这些真实供应者，未做 linklink、spconv、MQBench、MSBench 实际运行。私有设备名测试只验证不改模型的路由协议，不模拟设备数值执行。三家国产设备、驱动、SDK、各自版本组合和性能均待对应真机验收。

## 复现入口

公共检查见主说明。GPU 检查脚本在已构建扩展、安装对应任务依赖的环境执行：

```bash
python tools/validate_native_ops.py --baseline ../baseline --device 1 --output ../results/upstream_ops.pt
python tools/validate_native_ops.py --device 1 --output ../results/routed_ops.final.pt
python tools/validate_native_ops.py --compare ../results/upstream_ops.pt ../results/routed_ops.final.pt \
  --output ../results/native_comparison.final.json
python tools/validate_models.py --baseline ../baseline --device 0 --output ../results/models.final.json
CUDA_VISIBLE_DEVICES=2,3 UP_VALIDATION_OUTPUT=../results/distributed.final.json \
  python -m torch.distributed.run --standalone --nproc_per_node=2 tools/validate_distributed.py
python tools/validate_onnx.py --baseline ../baseline --output ../results/onnx_upstream.json
python tools/validate_onnx.py --output ../results/onnx_routed.json
python tools/validate_onnx.py --compare ../results/onnx_upstream.json ../results/onnx_routed.json \
  --output ../results/onnx_comparison.json
```

ONNX 两个采集命令会因上述已知 CPU 缺口返回非零并保存诊断；比较报告区分上游缺口和新增回归。本轮原始缺口保留在报告中，没有把它作为通过的数值样例。
