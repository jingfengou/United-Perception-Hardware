# 沐曦卷积反向非确定性定位与适配

本轮继续排查 C500 上 ResNet18、YOLOv5-S SiLU、外部官方 DETR R50 适配器的[短程数值差异](REAL_HARDWARE.md)。**没有恢复多随机种子精度复测，也不将三步训练称为收敛或数据集精度等价。**

## 1. 排查边界

- 使用此前同一 C500、厂商容器、PyTorch 2.10.0+metax3.8.1.0 / TorchVision 0.25.0+metax3.8.1.0、Python 3.12.11。
- 固定种子 131、batch 2、CIFAR-10 / PennFudan 原划分前 8 张训练图 / 4 张验证图、原有小图配方、三次更新及第二/三步后的真实 Runner eval。
- 固定 `rtol=1e-4, atol=1e-5`；结构、dtype、形状、整数和非有限值检查不放宽。
- TF32 关闭，FP32，`deterministic_algorithms=True`、`cudnn.deterministic=True`、`benchmark=False`。未使用 `portable_v1`、GPU FP64 替代、AMP 或 CPU 模型代算。

## 2. 如何判断是不是算子的问题

`tools/diagnose_metax_determinism.py capture` 在真实训练的第一个 batch 中截获卷积输入、权重、上游梯度和完整 stride/padding/dilation/groups/output_mask；前向重复两次，反向重复十次。**每次算子的操作数完全不变，调用前后还检查操作数哈希。** 返回给模型的是第一次真实结果，不用其他运行的梯度替换模型梯度。

再将真实操作数保存为 CPU 审计文件，在只导入 PyTorch、没有 UP、损失函数、数据增强、优化器或随机数生成的独立进程中重放同一个 ATen 算子。CPU 文件保存和结果比较不是 CPU 模型计算回退。

已经直接复现：DNN 开启时，三个模型的部分 `aten.convolution_backward` **只有权重梯度变化**；同次测试的前向和输入梯度可重复。ResNet 扩展排查还覆盖实际调用的融合 BN、池化、激活等反向，未观察到它们在这些固定操作数下变化。

这证明了当前厂商软件栈的该操作存在非确定性，不只是“种子可能没设好”。**没有厂商内部实现或内核级调度证据，不能进一步断言某一个原子指令、驱动缺陷或物理芯片是唯一根因，也不能外推所有形状。**

关闭 DNN 不是完整修复：两个 DETR 实际案例转而出现输入梯度非确定性；ResNet、YOLO 的 DNN-off 短程重复仍超差，DETR 在同权重恢复预测检查即超差。失败与未启动的后续控制分别保留，不选择性删除。

## 本轮结果

**启用下述显式修复后，三个模型的同厂商拆分前后及独立重复对照均通过。** 每模型三份独立进程训练，合计 9 次 / 27 次更新、18 次训练内 eval、9 次独立推理。不是三个随机种子，也不是旧版 native 与修复 profile 跨路径逐位相同。

| 模型 | 三步梯度、权重/BN/优化器状态张量 | 两次训练 eval / 恢复预测 / 独立推理 | loss 与指标 |
| --- | --- | --- | --- |
| UP ResNet18 | 两组新旧对照逐位一致，最大绝对差 0 | 逐位一致，最大绝对差 0 | 完全一致 |
| UP YOLOv5-S SiLU | 同上 | 同上 | 完全一致 |
| 外部官方 DETR R50 适配器 | 同上 | 同上 | 完全一致 |

另在 12 个保存的真实卷积反向案例中，每个修复实现重复 10 次，全部逐位一致；与同操作数的原生 DNN 结果均满足原 `rtol=1e-4, atol=1e-5`。各模型四案例的最大绝对差分别是 ResNet `3.27825546e-7`、YOLO `5.12227416e-9`、DETR `1.90734863e-6`。原生重复仍能复现权重梯度变化。

每次修复后训练实际替换的权重梯度操作数为 ResNet 60、YOLO 219、DETR 129；独立推理为 0。由此确认配置实际进入所测分支，而不是只注册了名字。CPU 契约与回归 **277 项通过**，两条 warning 为既有 legacy ONNX 弃用提示；静态硬件边界与两个契约索引检查通过，无 Torch/UP 的公共包普通导入通过。

可审计原始记录、失败对照、环境、冻结源码及文件索引见[本轮证据](validation/metax_determinism_20261022/README.md)。**这证明所测同一 C500 环境的短程训练可重复性及平台拆分一致性；没有证明全量训练精度、跨厂商平均精度等价或所有厂商操作的确定性。** 本轮没有申请或使用新的 NVIDIA GPU。

## 3. 只在平台层处理

新增显式配置 `metax_deterministic_v1`，实现位于：

```text
packages/up-platform/src/up_platform/backends/metax/
  device.py          配置、生命周期、状态查询
  determinism.py     卷积权重梯度的局部替换
```

只截获 GPU FP32 二维普通卷积的反向：

1. 前向继续使用原生 PyTorch/DNN。
2. 输入和偏置梯度继续使用原生 DNN；不请求原生权重梯度。
3. 权重梯度使用同一 GPU 上的 `unfold -> 按样本/组 bmm -> batch sum`。张量及中间计算仍为 FP32，不扩大精度，不修改卷积公式。
4. 不修改模型模块、参数、checkpoint 键、损失、优化器、学习率、数据、随机掩码或训练预算。

这个选项不替换整个 PyTorch，不切换语义算子供应者，不影响 native RNG，也不修改 DNN、TF32 或 MHA 的全局开关。推理用途可以保持同一配置，但不会执行这项反向替换；训练 eval 仍属于 training。

**默认 `native` 行为不变；只选择 `backend: metax` 不会自动启用该选项。** NVIDIA 不接受这个沐曦专用配置，其默认路径不变。

代价是额外的 im2col 临时显存和矩阵乘法/归约开销。本轮不把它宣传为性能优化，也不声称确定性开关覆盖所有厂商原生算子。

## 4. 如何使用

在已有模型 YAML 中只添加平台配置：

```yaml
runtime:
  device: cuda
  fp16: false
  accelerator:
    backend: metax
    numerics: metax_deterministic_v1
```

入口用途仍由 UP 命令固定，不给模型加厂商分支。独立包中对应：

```python
import torch
from up_platform import configure_platform

# 本报告使用的严格 FP32 / 确定性实验控制。
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
ctx = configure_platform(
    {"backend": "metax", "numerics": "metax_deterministic_v1"},
    purpose="training",  # 独立部署进程使用 inference。
)
print(ctx.numerics_state())
```

进程初始化后不能悄悄更换 profile。嵌套 dispatch mode 未退出时不允许 reset；reset 必须由创建该模式的线程执行。用完显式重置会退出本模式，不弹出别人的模式。

本轮验收保持 DNN 开启，**不要同时使用 `--nocudnn` 或手动禁用 DNN**；DNN-off 的输入梯度路径已发现非确定性。该 profile 不会替调用者修改全局精度/算法开关，仍须保留上述严格 FP32 与确定性控制。

当前局部替换仅承诺非空二维、非转置的密集 FP32 卷积权重一阶梯度。FP16/BF16、转置卷积等不支持的反向明确报错，不偷偷借用未验证内核或 CPU。AMP、分布式、图编译、二阶梯度、其他 GPU/驱动版本和大数据集收敛尚未验收。

## 5. 对照与复现

`tools/run_metax_determinism_controls.py fixed-controls` 运行三份同种子三步训练：

- 拆分前源码不变，测试入口显式进入同一个独立平台供应者的修复模式。
- 拆分后源码通过正式平台配置启用修复模式，重复两个独立进程。

**两侧都使用同一修复实现**，用于分离“同一硬件适配后的结构迁移”与“原始厂商实现的非确定性”。不是把修复后的结果冒充未改硬件路径的旧 native 结果。原始 native 和 DNN-off 失败继续保留。

每次使用真实 `BaseRunner.train/evaluate`，保存三步完整梯度、模型/BN/优化器/调度器状态，重构恢复后验证；再在新进程使用实际 `BaseInference.predict()`，比较同一批次和真实图片输出。原版 eval 不恢复训练标志的已知问题仍由测试调用者仅恢复标志，模型数据不变。

```bash
# BEFORE: 拆分前 0c9a43c7493563f837ad9dd27761cb00bf5fa1ce
# AFTER: 本次平台源码。先依厂商环境安装平台与 UP，不能换成 NVIDIA wheel。
export CUDA_VISIBLE_DEVICES="$ALLOCATED_GPU"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0 DEFAULT_TASKS=cls:det
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_DISABLE_CUDA_CUDNN_TF32=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
python tools/run_metax_determinism_controls.py fixed-controls \
  --root /path/to/new-task --source "$AFTER" --baseline "$BEFORE" \
  --data /path/to/frozen/data.json
```

原始算子重放：

```bash
python tools/diagnose_metax_determinism.py capture --extended \
  --source "$AFTER" --data /path/to/subset/data.json --model resnet18 \
  --output /path/to/new-capture
python tools/diagnose_metax_determinism.py replay \
  --case /path/to/new-capture/case-aten-convolution_backward-default-0.pt \
  --source "$AFTER" --profile metax_deterministic_v1 --output /path/to/new-replay
```

这些命令要求新输出目录；失败时不能删目录再挑选重跑。容差是工具常量，没有放宽容差的 CLI 参数。专门的原始厂商重复控制与本次修复后比较都使用同一判据。
