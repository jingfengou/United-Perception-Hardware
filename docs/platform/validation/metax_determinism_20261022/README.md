# C500 卷积非确定性专项证据

日期：2026-10-22。说明与启用方法见[专项报告](../../METAX_DETERMINISM.md)。这是同一厂商环境的固定种子短程回归，不是多随机种子精度任务。

## 结论与范围

- [最终审计](audit.json)：三模型、每模型拆分前一份与拆分后两份独立进程；共 9 次三步训练、18 次训练 eval、9 次独立推理；两组新旧对照的梯度、checkpoint 张量、预测均逐位一致，loss/指标相同。
- [固定真实操作数重放](candidate-replay-controller.json)：12 个卷积反向案例各重复十次，局部修复全部逐位一致，与原生 DNN 的梯度均通过未修改的 `rtol=1e-4, atol=1e-5`。
- [环境](environment.json)：容器内 MACA 3.8.1.3 / 驱动 3.6.11，C500，采集时本任务 GPU 计算已结束。厂商 Torch/TorchVision 版本及依赖在各训练/推理 report 的 environment 中。
- [完整运行命令](fixed-controls-controller.json)、[原生重放命令](replay-controller.json)、[DNN-off 控制命令](dnn-controls-controller.json)。控制器的子进程 exit 0 只表示命令完成，数值是否通过必须读取 comparison report 的 passed；最终审计已逐项检查。

| 模型 | 拆分后第一次对照 | 拆分后独立重复对照 |
| --- | --- | --- |
| ResNet18 | [comparison](fixed-v1/resnet18/compare-changed-0/report.json) | [comparison](fixed-v1/resnet18/compare-changed-1/report.json) |
| YOLOv5-S | [comparison](fixed-v1/yolov5s/compare-changed-0/report.json) | [comparison](fixed-v1/yolov5s/compare-changed-1/report.json) |
| DETR R50 适配器 | [comparison](fixed-v1/detr/compare-changed-0/report.json) | [comparison](fixed-v1/detr/compare-changed-1/report.json) |

## 归因与失败证据

- 最初真实卷积形状的捕获：[ResNet](capture/resnet18/report.json)、[YOLO](capture/yolov5s/report.json)、[DETR](capture/detr/report.json)。DNN 开启时固定同一操作数仍有权重梯度变化。
- 纯 PyTorch 六类操作控制：[DNN on](pure-torch-native.json)、[DNN off](pure-torch-no-dnn.json)。这个有限样例不能覆盖所有模型形状。
- [ResNet 全部实际反向排查](extended-v2/resnet18/report.json)包含融合 BN、激活、池化等；未观察到这些案例除卷积权重梯度外的变化。早期 `extended/` 没有覆盖融合 BN，保留以区分排查范围。
- DNN-off [ResNet](dnn-controls/resnet18/comparison/report.json)、[YOLO](dnn-controls/yolov5s/comparison/report.json)仍有超差。
- DNN-off [DETR 恢复预测失败](dnn-controls/detr/0/failure.json)；后续第二次 DETR 控制没有启动。该阶段没有被标成完整通过。
- [DETR 固定反向案例](replay/detr/case-1/report.json)在关闭 DNN 后输入梯度仍会变化。因此修复保留 DNN 输入梯度，只替换权重梯度归约。

不能据此断言厂商内部某条原子指令或芯片物理缺陷是唯一根因。本轮证明的是当前软件栈、所测操作数与执行路径的非确定性及针对性处理效果。

## 源码与数据身份

- 拆分前源：`0c9a43c7493563f837ad9dd27761cb00bf5fa1ce`，归一化 manifest `76ba5041ccfc32722c6336ef96a1754f6fe93729842cdee5bf65283e64fca0b2`。
- 修复执行源：`c6e8d3a22ebfa726f516a41f3fb699b60e3f29da` 加本次三个平台源码文件变动，归一化 manifest `a6c3077dedee98f9b83841f97f0c470a4d38a2a43afcd6d813f048c127f23154`。与发布运行时代码核对，不将后续文档提交时间冒充执行源码身份。
- 202 个模型/算法源码文件与拆分前相同；模型算法不改。三份训练使用同初始权重、逐步 batch 与 RNG，comparison 均核实哈希。
- 完整对照使用本次修正后的 [subset selection](fixed-v1/data/selection.json)与 [manifest](fixed-v1/data/data.json)，文件、图像、标签、权重、DETR 外部代码均通过原 manifest 校验。
- 最初 `capture/` 沿用上一轮的旧子集 manifest，其父集合 summary 字段过时的限制见[上一轮证据](../hardware_20261022/README.md)。实际选中图片及加载方式相同；本轮完整对照和最终审计不用这些旧 summary 字段。
- 拆分前对照通过测试入口进入同一个独立平台模式，[训练入口快照](diagnose-v4.py)完整保留。旧 UP 代码不改，但不能称为“未修改硬件路径的原始 native”。原始 native 超差未被覆盖。

主阶段使用 `diagnose-v4.py`；最后最小重放使用 [diagnose_metax_determinism.py](diagnose_metax_determinism.py)。初期捕获只保留工具哈希，最初脚本没有逐字节快照；后续版本、完整操作数哈希与可独立重放的二进制仍保留。发布工具增加了审计检查，不冒充所有阶段都使用同一工具哈希。

## 文件与复现

原始 [证据压缩包](metax-determinism-evidence.tar.gz)为 1,730,354 字节，SHA256：

```text
37b32dcb18647418a68063275ec08dd957735426f3e6d6ed82598ff51757bb9c
```

[压缩包索引](metax-determinism-evidence.tar.gz.index.json)记录 223 个原始文本文件和 189 个留在任务盘的二进制文件的大小/SHA256。本文与本地 CPU JUnit 是压缩包完成后的说明/验证，不伪装成原始包内容。原始日志和源码快照采用 `-text` 保留字节，不统一换行。

[CPU JUnit](cpu-junit.xml)记录 277 passed / 0 failed / 0 skipped；[发布核对](publication-audit.json)确认 223 个原始文件与 Git 暂存字节一致，725 个归一化运行时源码与真机快照匹配。[核对脚本](verify-publication.py)可以在仓库根目录执行。无 Torch 的公共 wheel 构建和正常安装后普通导入另行通过，产物保留在忽略的 `dist/platform-determinism-20261022/wheels/`；不将直接从 ZIP 导入当作 wheel 的正常安装方式。

大体积 checkpoint、三步完整梯度和最小重放操作数未放进 Git，仍位于沐曦任务挂载 `/workspace/up-run/platform-determinism-20261022/`（宿主机对应 `/mnt/afs/oujingfeng/up-hardware-metax-20261021/platform-determinism-20261022/`）。没有删除任何失败文件、旧任务或模型权重。

复现本轮短程对照使用专项报告的 `fixed-controls` 命令。重放已有真实操作数时使用 `candidate-replay`；原始厂商控制 `replay` 不启用平台模式。只用实际获配设备，不恢复旧多种子控制器。

没有验证：全量 ImageNet/COCO 收敛、跨厂商平均精度、AMP、分布式、图编译、二阶梯度、其他 SDK/芯片和所有卷积变体；额外临时显存与运行时间没有当作正式性能基准。
