# 可选的跨厂商数值配置

`portable_v1` 是硬件适配层中的显式、实验性配置，不修改模型模块、损失公式、数据增强、优化器配置或 checkpoint 参数名。默认仍为 `native`，既有 NVIDIA 路径不自动切换。它不是已获跨厂商精度等价保证的通用模式，完整训练结果另行记录，不能仅以接口测试通过作为精度结论。

## 使用方式

```yaml
runtime:
  device: cuda
  accelerator:
    backend: metax
    numerics: portable_v1
```

NVIDIA 对照使用同样的配置，将 `backend` 改为 `nvidia`。不能一边使用此配置、另一边使用原生计算，再称为相同数值配置。启用前需要匹配厂商 PyTorch 的可用 Triton，以及设备上的 FP64 算子实现；缺少实现报错，不自动把运算搬到 CPU。首轮接入环境为 PyTorch 2.10 / Triton 3.6 及相应厂商构建，不代表其他版本都已验证。

```python
from up.accelerator import get_accelerator
print(get_accelerator().numerics_state())
```

返回配置名称、随机流协议、seed、计数器和实际经过适配的操作计数。厂商身份仍由 `backend` 管理，不由 `cuda` 设备名称推断。

配置跟随后端实例。在同一进程中省略该字段会复用已经启用的配置，不会自动关闭；不要在训练中切换。需要另一种配置时使用独立进程，隔离测试才使用 `configure_accelerator(..., reset=True)` 并重新设种子。

完整对照还使用共同的确定性/TF32 设置，这些不是 YAML 中 `numerics` 字段的隐含效果。验证启动器会在建立 Runner 前设置：

```python
import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)
```

进程启动时同时设置 `PYTORCH_DISABLE_CUDA_CUDNN_TF32=1`、`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0` 和 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。仅设置相同 seed 或只复制 YAML，不等于复现了全部受测运行条件。

## 处理内容

- GPU 上的 uniform、rand、rand_like、Bernoulli 和 Dropout 使用 Triton 的 Philox4x32-10。同一 seed 和计数器对应同一序列，计数器使用 64 位偏移，不在 2^32 处回绕。Dropout 的概率、缩放、训练/评估语义不变；缩放及对应的 native dropout 反向使用共同的 GPU 宽精度计算，不能只替换掩码后仍让反向使用厂商的缩放内核。
- Linear/Attention 中的矩阵乘在 GPU FP64 中按固定 K 顺序执行显式 FMA；归约采用明确的 256 元素蝶形加法及分层顺序。不能只用 FP64 或只设置 `enable_fp_fusion=False`，再假定两种 GPU 编译器会生成相同的乘加与归约顺序。
- 2D 标准、分组、空洞卷积以完整的前向/反向为路由单位：unfold、共享矩阵乘、固定顺序求和及 fold 均在原 GPU 上执行。BatchNorm、LayerNorm、Softmax 也同时覆盖前向与反向；BatchNorm running buffers 按原有语义回写，包括 running variance 的无偏更新，不增加参数前缀或改变 state_dict 键。没有宣称转置卷积、其他维度卷积或所有变体已经统一。
- DETR 的 L1 `cdist` 及其广播反向、L2 gradient norm/foreach norm 使用共同的宽精度实现；ViT 广播产生的 autograd 内部 FP32 `sum` 也会被拦截，宽精度归约后转换回原 dtype。仅包装 Python 的 `torch.sum` 无法覆盖此反向路径。
- 部分激活与损失基础函数使用 GPU FP64 内部计算。参数、checkpoint 以及原为 FP32 的操作输出和梯度边界保持 FP32；这仍然**不是全程 FP32 运算**，不能与原生 FP32 的速度、显存或逐位轨迹混为一谈。
- Attention 固定 math 路径；卷积/BN 的选定实现不使用厂商 DNN 库。配置存续期间关闭卷积库，覆盖前向及反向；只在前向临时关闭不够，因为 PyTorch 的卷积反向会重新读取全局开关。通过适配层请求启用库不会越过此策略。重置硬件配置时恢复原卷积开关及 MHA fastpath，进程中不应并行混用其他数值策略。
- NMS/IoU 使用同一份适配层实现，内部采用共同的宽精度路径。TorchVision NMS 的同分数情况明确采用原输入顺序，非同分数排序、坐标约定和阈值语义不变。

这些是数值实现选择，不是冻结随机层、替换模型、舍入/量化梯度、选择更好的 seed 或静默 CPU 回退。随机流与旧的原生随机序列不同，因此也不能宣称新配置训练结果必须等于原生 NVIDIA 的每个数值。顺序矩阵乘、FP64、展开卷积和额外转换可能明显增加计算和显存成本；当前定位为精度/复现诊断模式，不是高性能训练建议。共享服务器上的并发诊断耗时也不能充当隔离的性能基准。

诊断时同时保存原始字节哈希和将 `+0/-0` 统一后的 FP32 哈希。原始哈希不同可能仅来自零的符号，不能单凭该差异就归因为非零数值错误，更不能直接归因为最终 AP 差异。

## 状态与边界

`get_rng_state()` / `set_rng_state()` 同时保存、恢复厂商原生状态和此配置的 seed/计数器，使用带版本标识的 CPU uint8 状态向量。CPU 这里只保存 RNG 元数据，不生成模型随机掩码或执行模型计算。不可把该状态当成原生 `torch.cuda` RNG 向量混用，也不可在未恢复计数器时声称训练轨迹连续。模型 tensor-only 权重的布局不变。

重新设种子应使用 `get_accelerator().manual_seed(seed)`；直接调用 `torch.manual_seed()` 不会替你重置此配置的独立计数器。UP 原有设种子流程已经经过运行时接口。

配置使用线程局部的 PyTorch Function/Dispatch Modes，生命周期属于初始化硬件的线程。当前限定 eager、单设备/进程、FP32 边界；AMP、CUDA Graph/编译、激活 checkpoint、DataParallel、多卡通信及其他模型的组合不在已验收范围。退出嵌套 PyTorch mode 后才能重置硬件配置。显式 GPU Generator 和列为不支持的分布，例如 randn、randint、normal、multinomial，会报错；不应将未列出的随机 API 或第三方随机内核视为已经统一。CPU 数据增强的原有随机调用不变。

`numerical_profiles` 是后端的显式能力声明。默认只声明 `native`；兼容 CUDA 设备语义、Triton 和选定宽精度操作的新厂商可以声明 `("native", "portable_v1")` 以复用这套实现，但仍需完成真机验证。普通 PyTorch 操作继续直接调用，未列出的计算不会因此自动获得跨厂商逐位一致保证。

## 验证入口

```bash
CUDA_VISIBLE_DEVICES=0 python tools/validate_portable_numerics.py \
  --backend metax --source "$PWD" --output /workspace/new-profile-contracts
python -m pytest -q tests/test_portable_numerics.py
```

GPU 检查保存完整张量哈希、Dropout 梯度和 RNG 恢复结果；跨平台应进一步比较这些证据，再按冻结计划运行多种子完整训练。基础算子通过不等于完整精度验收，也不保证所有浮点结果逐位相同。

完整前后向数学检查另用 `tools/validate_portable_math.py --backend metax --source "$PWD" --output NEW_DIRECTORY`。它以同平台原生 FP64 运算为参考，检查输出和梯度，并为跨平台比较保存完整张量哈希。检查里的 CPU FP64 oracle 仅作离线验证，不是生产实现中的回退。修复经过与完整预算训练的不同阶段见[修复报告](CROSS_VENDOR_FIX.md)。
