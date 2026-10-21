# 硬件隔离与后端路由

需要逐项核对硬件适配接口时，先读[完整接口定义](INTERFACE.md)，再查[全部函数签名](API_INDEX.md)或[机器可读清单](interface_contract.json)。这些清单与实际注册项通过自动检查保持一致。

NVIDIA 原版/路由版训练对照见[五模型多随机种子报告](MULTISEED_ACCURACY.md)。它与能力注册、合成输入功能检查及国产设备真机验收是不同层面的证据。

第一轮[沐曦/NVIDIA 四模型真实数据对照](CROSS_VENDOR_ACCURACY.md)已完成 40 次训练、76,000 步及状态/指标复核；预设 +/-1 点平均精度等价仅 ResNet18 通过，YOLO、DETR、ViT 未能确认。此前的[版本对齐与差异定位](CROSS_VENDOR_DIAGNOSIS.md)另完成 40 次对照及一次完整重复，定位随机掩码/厂商栈确定性缺口并修复 MetaX 卷积开关；三模型及关闭库 DETR 仍未确认等价。相同权重的 12 次交叉评估指标一致，不推广为训练精度等价。

继续的[数值修复与复测](CROSS_VENDOR_FIX.md)仅修改适配层，提供显式实验配置 [`portable_v1`](PORTABLE_NUMERICS.md)，不改变默认原生路径。首版 30 次完整训练未确认等价；固定完整操作及归约顺序后的短程对照已对齐，修复版沐曦 15 次 / 22,300 步及全部状态/指标审计已完成，NVIDIA 完整重测待空闲 GPU，跨厂商等价仍未评估。宽精度和顺序内核有额外成本，不作为全 FP32 或性能支持声明。

上层模型、损失函数、Runner 和推理 API 通过 `up.accelerator` 访问硬件。普通 PyTorch 计算、`new_zeros()`、继承输入设备的张量构造继续使用 PyTorch。

```text
模型 / 损失函数 / Runner / 兼容算子入口
                  |
        up.accelerator 统一边界
                  |
     当前后端覆盖 -> 显式共享实现
                  |
   PyTorch 设备运行时 / 厂商 SDK / 专用扩展
```

本轮以 NVIDIA 为兼容基线，增加昇腾、寒武纪和沐曦的运行时入口。沐曦 C500 已完成[严格 FP32 下四模型单卡功能检查](METAX_VALIDATION.md)，不代表全量数据精度或所有算子可用；昇腾和寒武纪尚未真机验证。具体情况见[能力矩阵](CAPABILITIES.md)和[验证报告](VALIDATION.md)。

## 配置与初始化

在原模型配置中增加 `runtime.accelerator`：

```yaml
runtime:
  accelerator:
    backend: nvidia
    plugins: []
  device: cuda
```

`backend` 可选 `auto`、`cpu`、`nvidia`、`ascend`、`cambricon`、`metax`，也可使用插件注册的名称。对应的设备名称分别为 `cpu`、`cuda`、`npu`、`mlu`、`cuda`；厂商身份与设备名称分别管理。切换到沐曦时同时选用 `backend: metax` 和 `device: cuda`；切换昇腾时使用 `backend: ascend` 和 `device: npu`。

`device` 可以省略，采用后端默认设备；可以带索引，如 `npu:0`。多进程启动由中间层按进程本地 rank 绑定设备。原来的命令行 `--backend dist|linklink` 仍表示通信框架，不是厂商选择；`runtime.accelerator.backend` 才表示硬件后端。标准 PyTorch 通信的默认传输由后端提供，linklink 路径需要对应供应者。

配置优先于环境变量 `UP_ACCELERATOR_BACKEND`，环境变量优先于自动选择。`auto` 保持原 NVIDIA 环境的选择方式；识别到扩展设备命名空间、已知非 NVIDIA 标记或多个插件探测命中时要求明确选择。仅凭 `cuda` 无法可靠判断厂商，国产环境应显式配置后端。

```python
import up
from up.accelerator import configure_accelerator, get_accelerator, capabilities

configure_accelerator({"runtime": {"accelerator": {"backend": "nvidia"}}})
up.initialize()  # 直接作为库使用时，注册任务、模型和命令
accelerator = get_accelerator()
model = accelerator.to_device(model)
batch = accelerator.to_device(batch)
print(capabilities())
```

CLI、ModelHelper 和现有 Runner 已接入初始化。仅 `import up` 或 `import up.accelerator` 不再自动加载全部任务或厂商 SDK。后端在工作进程初始化后保持固定；`reset=True` 供隔离测试使用，不用于运行中切换已创建设备的模型。

## 统一接口

| 边界 | 接口 | 语义 |
| --- | --- | --- |
| 设备 | `get_accelerator().device`、`resolve_device()`、`set_device()`、`current_device()`、`device_count()`、`is_available()` | 厂商与设备命名隔离，设备参数兼容整数、字符串与 `torch.device`，绑定在进程内完成 |
| 迁移 | `to_device(value, device=None, non_blocking=False)` | 迁移模块、张量及嵌套 mapping/list/tuple；普通 `.to(accelerator.device)` 复用 PyTorch 通用迁移 |
| RNG | `manual_seed()`、`get_rng_state()`、`set_rng_state()` | 设备 RNG；上层仍保留 Python、NumPy 和 PyTorch 主机 RNG |
| 资源 | `synchronize()`、`empty_cache()`、`memory_allocated()`、`max_memory_allocated()`、`reset_peak_memory_stats()` | 运行时同步与内存管理 |
| 精度 | `autocast()`、`grad_scaler()` | 共享统一 AMP API，扩展设备可使用其 SDK AMP；可覆盖后端方法 |
| 卷积运行时 | `configure_convolution()`、`convolution_flags()` | cuDNN 等厂商特性仅在后端出现 |
| 自定义算子 | `ops.roi_align()`、`ops.naive_nms()`、`ops.torchvision_nms()`、`ops.call(name, ...)` 等 | 按完整算子路由，供应者负责 autograd 与必要状态 |
| 同步 BN | `layers.SyncBatchNorm`、`ops.sync_batch_norm()`、`sync_batch_norm_forward` | 保留权重路径，允许独立替换执行实现 |
| 旧 linklink 层 | `layers.LinkSyncBatchNorm`、`layers.link` | 构造时才加载供应者，保留原类继承和参数层级 |
| 稀疏层与张量 | `sparse.SubMConv3d()`、`SparseConv3d()`、`SparseInverseConv3d()`、`SparseConvTensor()`、`SparseSequential`；`features()`、`replace_feature()`、`to_dense()`、`is_sparse_convolution()` | 返回厂商原生对象；张量读取、变换和容器执行同样可覆盖 |
| 分布式 | `distributed.init_process_group()`、`wrap_model()`、集合通信函数 | 绑定当前设备、选择传输、构建 DDP；集合通信通过 `distributed.*` 算子注册项路由 |
| 可选依赖 | `dependency()`、`layers.OptionalModule`、`optional_symbol()` | 稀疏、量化、剪枝、linklink 延迟加载 |
| 能力 | `capabilities()` | 返回算子 owner、反向声明、dtype/variant 元数据、依赖供应者；不触发 CUDA 二进制加载 |

能力查询描述“已注册”，并不证明 SDK 已安装或真机可执行；返回状态为 `registered_not_hardware_validated`。`backward` 为 `None` 表示未声明，为 `False` 表示无反向契约；`dtypes/variants` 为 `None` 表示未声明，不能解释为支持任意类型或变体。插件可用 `register_op(..., dtypes=..., variants=...)` 声明能力。调用时检查已声明的首个浮点张量 dtype，其余形状、参数、变体及多输入约束由实现验证。AMP 拒绝非法精度和不可用设备，后端提供 BF16 检查时也会调用该检查。

缺少算子或依赖会报告后端、操作/供应者和原因，不增加自动迁移到 CPU 的数值实现。原有明确命名的 CPU 算子仍保留原语义。CPU 公共层可导入、安装和运行通用测试，不代表所有模型都具有 CPU 实现。

## 兼容边界

`up.extensions.python.*`、`up.extensions` 的原名称和 `.apply()` 保留为门面，内部转发到 `ops`。NVIDIA 的 Function 前向、反向及 ONNX symbolic 在 `backends/nvidia_ops/`；新厂商应注册完整语义操作，避免只替换前向、却沿用 NVIDIA 反向保存状态。

不增加模型参数嵌套层，不重命名 checkpoint key，不改变原算子的返回结构。`ModelHelper.cuda()` 保留为兼容入口，转发到当前后端；模型内部已使用统一设备。checkpoint 先映射主机内存再恢复，避免绑定原设备编号。旧配置里的匹配策略 `mode='cuda'` 只是历史策略标签，已列入静态检查的精确白名单。

低层 `up.extensions.ext` 仅为外部旧代码保留，直接使用它会进入 NVIDIA 二进制。仓库内上层代码禁止导入该低层门面或后端实现。原 CPU 导出占位函数仅在已有导出路径允许，不注册为可执行 CPU 算子；PSRoIAlign 原扩展缺少 CPU 导出函数，使用 GPU 张量导出。

## PyTorch 与 Python 版本

各厂商在各自支持的 Python、PyTorch、驱动和 SDK 组合内运行。中间层不会把任意官方 PyTorch wheel 变成厂商适配版，也不会动态安装、降级、转换二进制 ABI。每个厂商应使用独立虚拟环境或厂商容器，在其支持范围内选择本项目可用版本；自定义扩展在该环境重新编译。

根目录 `requirements.txt` 保留上游历史依赖锁定；本轮现代环境的实际版本以验证产物为准。公共契约检查只需 PyTorch、pytest、setuptools、wheel、packaging；运行完整任务还需对应数据、模型和工具依赖，按目标环境验证，不能把历史依赖文件视为已验证的现代环境安装锁。

可以在后端配置中设置部署环境的包约束，初始化时按 PEP 440 校验。例如，下面只约束本轮验证所用的 Python/PyTorch 版本区间，不是三家厂商的支持声明：

```yaml
runtime:
  accelerator:
    backend: nvidia
    requirements:
      python: ">=3.12,<3.13"
      torch: ">=2.8.0a0,<2.9"
```

`requirements` 的键是已安装 distribution 名称，`python` 是特殊项；厂商插件包也可加入。版本不满足会明确报错。厂商不支持所需版本时，选择双方兼容的版本，或由后端插件/厂商 SDK 完成适配并验证；公共模型流程不增加厂商版本分支。不能仅凭“支持 PyTorch”推断自定义 CUDA 扩展、linklink、稀疏库和量化工具同样可用。

## 构建

```bash
# 公共 Python 包：无需安装 PyTorch 或 CUDA 即可构建
UP_BUILD_BACKEND=none python -m pip wheel --no-deps --no-build-isolation .

# NVIDIA 扩展：在匹配的 PyTorch、CUDA toolkit 和编译器环境内构建
UP_BUILD_BACKEND=nvidia CUDA_HOME=/usr/local/cuda \
  TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2 python setup.py build_ext --inplace
```

`UP_BUILD_BACKEND=auto` 为默认值：明确选择非 NVIDIA 后端，或缺少 CUDA PyTorch/toolkit 时不构建 NVIDIA 扩展。国产环境建议显式使用 `none`，SDK 和专用二进制由对应插件独立构建、安装、加载。公共 sdist 通过 `MANIFEST.in` 保留原生源码。NVIDIA 加载器只允许明确选用 NVIDIA 实现的后端加载 `_C`。

## 清单、检查与验收

[逐项硬件清单](upstream_inventory.json)从固定上游提交 `52aaeb9ed27a490820696d7754d19f98cd6ecd11` 生成，记录原调用位置、源代码、统一路由、迁移位置和状态；另含原生源码族和兼容入口说明。

```bash
python tools/check_hardware_boundaries.py
python tools/build_hardware_inventory.py
python -m pytest -q tests/test_accelerator.py
```

AST 检查禁止上层直接引入设备运行时、设备谓词、硬件常量、SDK 导入、底层扩展、AMP、DDP 和集合通信；覆盖常用导入别名及已知动态 SDK 导入，也检查既有稀疏模型的张量访问。它是维护规则，不是能够分析任意反射或动态代码的安全沙箱。白名单仅保留明确的兼容路径和两个历史配置默认值。CI 在 Python 3.10/3.12、CPU PyTorch 2.6 环境运行契约检查与公共构建；实际运行记录以验证报告为准。

新增厂商参见[接入指南与单算子覆盖示例](ADDING_BACKENDS.md)。扩展验收要求新增后端/插件和注册配置即可完成接入；模型、损失和训练流程无需新增厂商判断。
