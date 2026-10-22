# DETR 差异归因：官方网络与纯 ATen 对照

## 问题与边界

本轮不修改 UP 模型算法、平台数学实现、训练配方或容差。目标是区分 [前轮 DETR 未通过](CURRENT_VALIDATION_20261022.md#detr-失败与重复对照) 的 UP 编排因素与厂商 PyTorch 计算路径，而不是重新定义精度通过标准。

已在独立于 UP 的官方 DETR 反向中找到固定操作数仍变化的 `aten.mm.default`。同一案例在仅导入 PyTorch 的独立进程中可复现，在所测 RTX 4090 上没有复现。**UP 不是这个问题发生的必要条件；结论定位到所测沐曦 PyTorch / GPU 矩阵乘法路径，不等于物理芯片故障，也没有证明全部历史 AP 差异只由这一处造成。**

三份完整训练均完成 720 次更新，两份独立官方网络之间也不重复。未经布局适配的官方损失另有确定性检查拒绝，保留失败，不改开关使它跑通。

## 对照设计

沿用冻结 UP `f78e27a5f2139480a90e5f7a8cdcfa94c48ab48c`、官方 DETR `29901c51d7fe8712168b8d0d64351170bc0f83e0`、seed=131、PennFudan 120 train / 50 test、320/512px、原 batch2 分组采样、12 epochs / 720 updates、AdamW 分组学习率及 0.1 梯度裁剪。张量仍为 `rtol=1e-4, atol=1e-5`；旧指标判据 `metric_atol=1e-6` 不变。

1. **UP 记录。** 运行真实 `BaseRunner`，观察官方 detector / criterion 实际接收的张量，保存每次前向前的 Python、NumPy、CPU/GPU RNG、实际学习率、原始及裁剪后梯度摘要、更新后权重摘要。前 70 步保存完整原始梯度；关键步保存更新前模型和优化器。观察返回原结果，不替换训练输出。
2. **独立官方网络。** 新进程直接构建冻结官方 `DETR`、`Transformer`、骨干与 criterion；恢复同一初始化状态。重放同一输入张量、RNG 和学习率，使用 PyTorch AdamW / 裁剪，不调用 UP Runner、模型门面、数据加载或平台运行时。学习率/RNG 重放用于隔离编排，不是新的训练配方。
3. **单步追踪。** 从同一第 58 步更新前状态开始，记录实际 ATen 调用及输入/输出 SHA256。追踪不替换输出；未初始化分配的无意义字节、全写入操作的旧目的缓冲区不作为数值差异。发现首个不同输出时保存实际操作数、dtype、形状与步幅。
4. **纯算子。** 新进程只调用 PyTorch，禁止导入 `up` 和 `up_platform`。恢复实际矩阵布局，重复同一个 ATen 操作 100 次，并验证输入前后哈希不变。另在 RTX 4090 使用同一份案例文件。

### 什么叫“不经过 UP”

- 网络计算类来自冻结的官方 DETR 源码，不执行 `DetrForUP.forward()`。离线构造使用官方类，随后完整加载已经保存的初始权重，不触发 Backbone 的隐式网络下载。
- 与前轮严格匹配时，保留既有的单行人类别 head、辅助损失、归一化目标框，以及已冻结的二维展开分类损失布局；没有把它们偷偷换成官方 COCO 默认配方。损失布局复用旧验证辅助函数，不依赖 UP。
- `metax_deterministic_v1` 匹配控制只加载独立平台包中已经冻结的卷积权重梯度 dispatch mode，不使用设备注册、训练/推理路由或 UP。该控制标记为“无 UP、保留局部卷积 mode”，**不能称作无平台包**。
- `--profile native --criterion official` 控制使用未经子类覆盖的官方 `SetCriterion`，禁止两包导入。纯 ATen 重放同样禁止两包导入，不包含模型、损失、随机采样或优化器。

官方训练步骤的原始定义见 [固定版本 engine.py](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/engine.py#L15)。本轮使用便于逐步审计的普通 PyTorch 循环，不调用 UP 或 `engine.train_one_epoch`；保持前向、加权损失、反向、裁剪、AdamW 的原顺序。**这不是原封不动的官方 COCO 默认数据管线训练。**

## 完整预算结果

UP 记录一次、独立官方网络重复两次，共 **2,160 次完整预算更新**。三者初始模型相同，两个独立过程的全部 720 个实际输入均与 UP 记录匹配；每次前向前重放同一组 RNG 和实际学习率。两份独立过程均确认 `up_imported=false`，仅保留显式局部卷积 mode。

| 成对比较 | 首个梯度/权重差异 | 首个 loss 差异 | loss 首次超原容差 | 最终权重最大绝对差 |
| --- | ---: | ---: | ---: | ---: |
| UP / 独立官方 0 | 58 | 59 | 65 | 0.017172399908304214 |
| UP / 独立官方 1 | 58 | 59 | 65 | 0.016109943389892578 |
| 独立官方 0 / 独立官方 1 | 58 | 59 | 70 | 0.016840625554323196 |

三份最终模型及优化器状态均有限，最终模型 SHA256 与逐步记录一致。两个“不使用 UP”的过程之间仍有同类分叉，排除了“只有经过 UP 才会发生”这一解释。**表格是训练轨迹归因，不是新的 AP 等价验收或修复通过。** 前轮 AP 失败不变，本轮没有用新 AP 数字替换旧结果。

## 已定位的首个差异

前 70 步诊断中，前 57 次更新的原始梯度、裁剪后梯度与更新后权重均逐位一致。第 58 步前向 loss 仍相同，原始梯度首次不同，涉及 221 个参数，最大绝对差 `4.76837158203125e-6`，当步尚无元素超出原容差；第 59 步已有梯度超差。观察与前轮第 59 步首次 loss 分叉相衔接，不把 loss 分叉当成首个算子差异。

同状态两次官方网络追踪各有 6,185 个 ATen 记录。第 3,788 个调用（零基索引 3,787）首次出现相同操作数、不同输出：

```text
aten.mm.default
A: FP32 [4, 600], stride [1, 4]
B: FP32 [600, 256], stride [256, 1]
Y: FP32 [4, 256]
inputs SHA256:
08acee0aa315e7a97f0f5e2d0e0a3ec81c6a47d1e7b965ee6f0d33d304ebdb58
```

它位于 bbox head 最后一层 Linear 的权重反向；第三次同状态追踪进一步验证该操作返回值与 `bbox_embed.layers.2.weight` 的原始梯度逐位相同。实际激活为 `[6,1,100,256]`，图像张量为 `[1,3,384,320]`；该批次是原分组采样产生的单图批次，没有修改 batch 或丢弃尾批。完整 720 步记录的第 58 步模型、输入和 RNG 与单步追踪均按哈希核对相同。

| 同一实际案例，每进程 100 次固定操作数重放 | 两个独立进程各自不同于首次的次数 | 最大绝对差 | 超原容差元素 |
| --- | ---: | ---: | ---: |
| MetaX C500 / Torch `2.10.0+metax3.8.1.0` | 99 / 99 | `3.814697265625e-6` | 0 |
| RTX 4090 / Torch `2.10.0+cu126` | 0 / 0 | 0 | 0 |

两边均启用 `torch.use_deterministic_algorithms(True)`，关闭 TF32 / DNN benchmark，保留 DNN，使用 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。这些是实际执行设置，不据此假定厂商一定完整实现 NVIDIA 环境变量的语义。

CPU FP64 仅对这两个保存的矩阵做数值参考审计，两平台首次结果的最大绝对差均为 `1.9073486328125e-6`，均在原容差内。**这是可重复性问题的证据，不是该单次 GEMM 数值错误或超差的证据。** CPU 没有替代模型训练计算。

## 未经布局适配的官方损失

`--profile native --criterion official` 不导入 UP 或平台包，使用原始 `SetCriterion` 的转置分类损失布局，但前向在严格确定性模式下被拒绝：

```text
nll_loss2d_forward_out_cuda_template does not have a deterministic implementation
```

原控制失败日志保留。另建目录，仅增加异常时保存部分 trace 的代码，以相同配置重现拒绝；部分 trace 确认两包均未导入。没有关闭确定性、改成 `warn_only`、覆盖官方损失，或把该控制算作完整训练成功；原计划的第二次该控制未启动。这是当前严格配置下的执行限制，**不能单凭这个 PyTorch 拒绝认定为沐曦特有缺陷**，也不拿它替代已经独立重放的 GEMM 证据。

## 复现

使用与前轮一致的厂商容器及冻结源码、数据；不要安装 NVIDIA Torch 覆盖厂商 wheel。实际 C500 环境为 Python 3.12.11、Torch/TorchVision `2.10.0+metax3.8.1.0` / `0.25.0+metax3.8.1.0`，沿用不可变镜像 digest，详见[环境记录](CURRENT_VALIDATION_20261022.md#源码与环境)。

所有输出目录必须是新目录。以下 `SOURCE` 是冻结 UP 快照，`DATA` 是已经验证过哈希的原数据 manifest；`RUN` 是独立实验根目录：

```bash
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0 DEFAULT_TASKS=cls:det
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_DISABLE_CUDA_CUDNN_TF32=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
DRIVER=benchmark/hardware/diagnose_detr_origin.py

# 保留原 720 步学习率配方，记录实际 tensor/RNG/LR tape。
python "$DRIVER" capture --source "$SOURCE" --data "$DATA" \
  --steps 720 --save-gradients 70 --output "$RUN/capture-full"

# 与冻结策略一致，但不调用 UP；两次是同种子独立重复，不是多种子任务。
for REPEAT in 0 1; do
  python "$DRIVER" standalone --source "$SOURCE" --tape "$RUN/capture-full" \
    --steps 720 --profile metax_deterministic_v1 --criterion matched \
    --output "$RUN/official-full-$REPEAT"
done

# 同一模型状态/输入/RNG，观察首个不同操作，不替换结果。
python "$DRIVER" trace_step --source "$SOURCE" --tape "$RUN/capture-full" \
  --step 58 --output "$RUN/trace0"
python "$DRIVER" trace_step --source "$SOURCE" --tape "$RUN/capture-full" \
  --step 58 --reference "$RUN/trace0/trace.json" --output "$RUN/trace1"

# 不需要 SOURCE 或平台包；仅需匹配的 PyTorch 和保存的算子案例。
python "$DRIVER" replay --case "$RUN/trace1/first-different-operation.pt" \
  --repeats 100 --output "$RUN/pure-mm"
```

不同运行可能产生不同的最后位、哈希或首次观测位置；应以自己的完整 trace 找首个差异，不硬编码指定操作后宣布命中。`trace_step` 使用保存的关键步快照；若自己的首次分叉不在已保存步骤中，需要另建阶段增加所需快照，不能拿不同状态做固定输入归因。native 官方 criterion 控制须分别加 `--profile native --criterion official`，不能与 matched 路径冒充同一训练轨迹。

仅加载自己生成或已经校验 SHA256 的可信 tensor tape / checkpoint；完整 RNG 状态需要受信任的 PyTorch 反序列化。

### 无 UP 的最小算子复现

不必安装 UP、平台包、官方 DETR 或数据集。使用厂商 PyTorch，下载已经保存的实际矩阵，核对 SHA256 后执行本仓库的单文件工具：

```bash
curl -fL -o "$RUN/first-different-operation.pt" \
  https://github.com/jingfengou/United-Perception-Hardware/releases/download/detr-origin-20261022/first-different-operation.pt
sha256sum "$RUN/first-different-operation.pt"
# 预期：0bcbe92b0db33291da6d7d1b232e53e53e2ee9613f77c8ca1e0dc3788b882ec1
python benchmark/hardware/diagnose_detr_origin.py replay \
  --case "$RUN/first-different-operation.pt" --repeats 100 --output "$RUN/new-pure-mm"
```

仍保留上面的严格环境变量。工具内部保持确定性请求及 FP32，校验操作数前后不变，重建原步幅，并检查两包确实没有导入。`report.json` 中的 `exact=false` 表示观察到不逐位重复；同时查看 `outside_tolerance`，不要把这两个判据混用。

## 证据与维护验证

- [机器可读摘要](detr-origin-20261022.json)：全部成对结果、原生官方损失的拒绝、环境、源码核对和产物校验。
- [固定 Release](https://github.com/jingfengou/United-Perception-Hardware/releases/tag/detr-origin-20261022)：54 个原始文本/工具记录、同名逐文件索引、小型实际算子案例。归档已逐成员校验，不把受控失败改记为成功。
- 索引另记录 **993 个 `.pt` 张量/快照/最终状态文件，共 39,582,535,431 字节**，保留在本任务目录；除了小型算子案例，不把数据或大权重加入 Git / Release。Runner 原生 `.pth` 输出也保留在服务器，不属于这个 `.pt` 清单。

| 下载文件 | 字节 | SHA256 |
| --- | ---: | --- |
| `detr-origin-evidence.tar.gz` | 25778361 | `5463403b2e9cef3c5b482d9c15aa2bece6555046167cdcaf32c5c4e21a180748` |
| `evidence-index.json` | 190021 | `2c5c06f7ceb5df16fac1997949becab3efeaef4bb176897d3473d97dc0a29584` |
| `first-different-operation.pt` | 630787 | `0bcbe92b0db33291da6d7d1b232e53e53e2ee9613f77c8ca1e0dc3788b882ec1` |

先行 70 步控制复用了验证适配器的离线构造，但未调用其 UP 前向；最终完整训练与单步追踪直接使用官方类。诊断驱动各次快照全部归档。完整 UP 记录、独立训练和 NVIDIA 重放实际使用的函数与提交工具按 AST 核对一致；最终版只另增加了异常时保存部分 trace 的处理。最终执行的工具字节与提交文件一致，不把驱动记录改动伪装为运行时数值修复。

本地维护验证：**339 tests + 3 subtests 通过**，两条 warning 为既有 legacy ONNX 弃用提示；31 个运行时成员 / 73 项操作契约无漂移，硬件边界违规 0；665 个受保护算法/配置文件、6 个历史协议夹具未变。归档后再次检查，C500 无本任务活动进程，4090 无计算进程；共享容器和历史结果未删除。

## 尚未证明

- 没有内核内部或驱动级因果证据，不能断言某条 atomic 指令、具体 BLAS 算法或物理芯片是唯一根因。
- 首个差异与完整训练非重复的证据，不等于量化解释了全部历史 `-0.88634 AP` 差值；不把原严格精度失败改成通过。
- 本轮没有修复运行时、替换整套 GEMM、强制 CPU 模型代算、扩大精度、修改损失或删除尾批。没有恢复旧多随机种子任务。
- 单设备、所列版本和固定小数据预算不能外推 AMP、多卡、其他 shape/驱动版本或全量 COCO 收敛。
