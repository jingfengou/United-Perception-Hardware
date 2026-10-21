# 首次训练差异的因果定位

## 目的与边界

2026-10-21 在 A800-1 的物理 GPU 2 上进一步检查：为什么选择 NVIDIA 后，Faster R-CNN 仍与原版出现训练差异？

结论：**本次真实批次前 3 步的差异起点是原版已有的原生 RoIAlign AVG 反向，不是前向公式、初始权重、输入数据或随机状态不同。** 只固定该反向算子的返回梯度，两版在这 3 步中重新逐位一致。这是有干预的早期训练定位，不是完整训练或最终 AP 等价验收。

不修改 `up/`、厂商后端或 CUDA 内核，不加入正式训练的梯度替换、舍入或 CPU 回退。源码仍对应之前训练报告中的上游 `52aaeb9` 与改造版 `c7be549` 的运行时内容。

## 观察结果

两版使用同一数据清单、有效配置、种子 131、初始权重、初始验证指标和诊断驱动。原来的 PyTorch/cuDNN 确定性设置保留，另明确设 `PYTHONHASHSEED=0`。观察钩子会同步并读取张量，因此这是新诊断，不要求非确定性的后续 loss 重现历史数值。

| 检查项 | 原版与路由版首步 |
| --- | --- |
| 训练 batch，包括图像和标注 | 逐位相同 |
| Python、NumPy、CPU/CUDA Torch 随机状态 | 所有记录点相同 |
| Backbone、FPN、RPN、RoI 预处理、BBox head 和后处理的前向输出 | 全部逐位相同 |
| 总 loss | 同为 `1.4564250707626343` |
| 3 次非空 RoIAlign AVG 反向的所有输入，包括上游梯度与 RoIs | 全部逐位相同 |
| 这 3 次原生反向返回的特征梯度 | 全部有逐位差异 |
| RPN head、BBox head 的参数梯度 | 全部逐位相同 |
| FPN、Backbone 的参数梯度 | 139 个张量有差异，最大绝对差 `3.3527612686157227e-8` |
| 首次 SGD 更新后的权重 | 已有逐位差异，下一步前向/loss 随之分离 |

完整逐步对照：[原版与路由版](natural-comparison.json)。作为独立对照，同一原版再运行一次，也出现首步前向/随机状态相同、RoIAlign 反向输入相同而返回梯度不同的现象：[原版与原版](self-comparison.json)。

## 为什么原版也有差异

原生 [`roi_align_kernel.cu`](../../../../../up/extensions/csrc/roi_align/roi_align_kernel.cu) 的 AVG backward 在第 269-272 行对共享的特征梯度执行浮点 `atomicAdd`。多个 GPU 线程向同一位置累加时，原子操作避免写入丢失，但不固定线程之间的累加顺序；浮点加法不满足结合律，顺序不同会改变末位。这不是不同随机种子产生的差异。

两边该内核的源码没有修改，之前的固定输入、同一图重复 backward 诊断也已测出原版自身的非确定性。`torch.use_deterministic_algorithms(True)` 不是任意第三方自定义内核的改写器；该 UP 扩展没有自行提供确定性实现或接入非确定性告警，开关不会消除这里的 atomicAdd。

原理参考：[NVIDIA 的浮点原子归约说明](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)、[PyTorch 算子确定性接入说明](https://github.com/pytorch/pytorch/wiki/How-to-support-%60torch.set_deterministic%28%29%60-in-PyTorch-operators)。

微小的首次梯度差异会进入优化器状态和权重。检测训练还包含阈值匹配、排序、NMS 和采样等离散选择，可能进一步放大差异；本诊断没有逐项归因这些后续放大环节，不能把历史最终 AP 分差全部定量归因于其中某一个环节。

## 干预实验

1. 原版运行 3 步，记录每次原生 RoIAlign AVG backward 的输入和返回梯度。
2. 路由版独立启动。每次先正常执行自己的原生 backward，再严格核对该调用的全部输入与记录逐位相同，仅在本诊断进程中将返回梯度替换为原版的已记录值。任何输入不一致立即报错，不能继续回放。
3. 对比每个阶段输出、loss、随机状态、全部参数梯度及更新后的模型状态哈希。

结果：[3 步回放对照](replay-comparison.json)中，这些项目全部逐位相同；路由版正常计算的原生 backward 输出仍有差异，但回放后实际交给 autograd 的输出相同。

这证明 **RoIAlign 反向的输出差异足以解释本次前 3 步的轨迹分离**，在这段轨迹上没有发现额外的模型/优化器语义变化。不代表后续 177 步或历史 AP 分差已全部完成因果排查，更不代表完整 COCO 统计精度等价已经通过。回放依赖已记录的参考梯度，不能作为真实训练的修复办法或验收成绩。

## 原始记录与复现

- [原版观测记录](baseline.json)、[路由版观测记录](routed.json)、[原版重复记录](baseline-repeat.json)、[路由版回放记录](routed-replay.json)。张量只记录形状、dtype、SHA-256；实际梯度张量留在服务器。
- [诊断工具](../../../../../tools/diagnose_training_divergence.py)和[对应测试](../../../../../tests/test_training_accuracy.py)。每份原始记录含驱动文件 SHA-256。
- 任务环境、数据和运行时源码依据见[完整训练报告](../../../TRAINING_ACCURACY.md)。本诊断没有重新下载数据、构建扩展或更换环境。

在已有对照环境中，分别传入上游和改造版目录；物理设备号需要按实际空闲情况选择：

```bash
export PYTHONPATH=/workspace/up-hardware-routing-20261021-accuracy/compat${PYTHONPATH:+:$PYTHONPATH}
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 PYTHONHASHSEED=0
python tools/diagnose_training_divergence.py \
  --source /workspace/up-baseline --data /workspace/up-accuracy-data/data.json \
  --steps 3 --output /workspace/diagnostic/baseline
python tools/diagnose_training_divergence.py \
  --source /workspace/up-routed --data /workspace/up-accuracy-data/data.json \
  --steps 3 --output /workspace/diagnostic/routed
python tools/diagnose_training_divergence.py \
  --source /workspace/up-routed --data /workspace/up-accuracy-data/data.json \
  --steps 3 --replay-roi /workspace/diagnostic/baseline \
  --output /workspace/diagnostic/routed-replay
python tools/diagnose_training_divergence.py \
  --compare /workspace/diagnostic/baseline /workspace/diagnostic/routed-replay \
  --output /workspace/diagnostic/replay-comparison.json
```

将第二条命令的 source 改为同一原版目录、output 改为空的新目录，可做同版本重复对照。compare 只输出定位信息，不作为正式精度验收的退出码。

服务器原始张量和日志位于 `/workspace/up-hardware-routing-20261021-accuracy/results/divergence-*`，不提交大张量或数据图片到 Git。
