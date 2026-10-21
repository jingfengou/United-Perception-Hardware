# 新增厂商

开始实现前先查阅[完整接口定义](INTERFACE.md)、[全部函数签名](API_INDEX.md)和[机器可读清单](interface_contract.json)。共享运行时不兼容的方法需要覆盖，专用操作按目标模型需求补齐，不要求重新实现普通 PyTorch 或未启用的可选 SDK。

## 复用共享实现，只覆盖差异

仓库内的 `up/accelerator/example_plugin.py` 是可运行的 CPU 示例：继承 `TorchAccelerator`，复用通用 overlap、运行时和其他已注册共享操作，仅覆盖 `naive_nms`。它用于测试插件协议，不模拟国产显卡。

```python
from up.accelerator import configure_accelerator, ops, capabilities
import torch

configure_accelerator({"runtime": {
    "device": "cpu",
    "accelerator": {
        "backend": "example_cpu",
        "plugins": ["up.accelerator.example_plugin"],
    },
}})
indices = ops.naive_nms(torch.tensor([
    [0., 0., 4., 4., .9], [1., 1., 3., 3., .8],
]), .5)
assert capabilities()["operations"]["naive_nms"]["owner"] == "example_cpu"
assert capabilities()["operations"]["gpu_iou_overlap"]["owner"] == "portable"
```

已有厂商兼容时可以使用同一运行时类，按注册名称覆盖一个算子：

```python
from up.accelerator import TorchAccelerator, register_backend, register_op

register_backend("new_vendor", TorchAccelerator)  # 示例运行在 CPU
register_op("new_vendor", "naive_nms", "new_vendor_up.ops:nms", backward=False,
            dtypes=("torch.float32",), variants={"coordinate_offset": [0, 1]})
```

`register_backend(name, factory, replace=False, probe=None)` 支持可调用工厂、装饰器或 `"module:attribute"` 延迟路径。`register_op(backend, name, implementation, backward=None, replace=False, dtypes=None, variants=None)` 同样支持装饰器和延迟路径。重复注册默认报错，替换必须显式指定 `replace=True`。可选 `probe` 在自动选择时运行，不应分配设备；多个命中会报配置歧义，生产环境优先显式选后端。

## 真实设备插件

下面是接口示意，名称和 SDK 需替换为目标厂商实际提供的实现，不应直接作为可执行驱动：

```python
from up.accelerator import TorchAccelerator, register_backend

@register_backend("vendor_x")
class VendorX(TorchAccelerator):
    name = "vendor_x"
    device_type = "vendor_device"  # SDK 注册的实际 PyTorch 设备名称
    communication_backend = "vendor_collective"
    op_fallbacks = ("portable",)

    def __init__(self, device=None):
        import vendor_torch_extension
        super().__init__(device)
```

继承共享运行时方法，只有设备/AMP/内存接口不兼容时才覆盖对应方法。设备名称可以与其他厂商相同，不能用它推断 SDK 或扩展兼容性。新插件通过 `plugins: ["vendor_x_up"]` 导入并注册，配置选 `backend: vendor_x`，模型无需修改。

对于已经提供的厂商后端，可以在外部插件中直接 `register_op("ascend", ...)` 等补齐缺失操作，不必复制整套后端。CUDA 形状的 API 兼容并不自动授权加载 NVIDIA `_C`。只有完成对应二进制构建和语义验证后，才允许显式把兼容 owner 加入 `op_fallbacks`；优先逐操作注册已验证的兼容实现。

查找顺序为 `(当前注册后端名,) + op_fallbacks`，按第一个匹配项执行。`portable` 是共享实现集合，不是设备类型。注册的新算子用 `ops.call("operation_name", ...)` 调用；已有命名入口继续使用 `ops.roi_align()` 等。`replace=True` 适合明确替换已有注册项，不应用于掩盖重名错误。

## autograd、层和通信

注册可微算子时，供应者应提供完整 Function 的 `.apply`，其内部处理前向、反向和 context。也可使用可微 PyTorch 组合实现。`backward=True` 是能力声明，不会自动生成反向。保持原门面签名、返回结构及 ONNX 符号契约；精度转换、张量连续性及不支持的变体在实现内明确处理。

层应直接返回兼容参数布局的原生模块，不额外包裹 `nn.Module` 并增加 checkpoint 前缀。同步 BN 可覆盖 `sync_batch_norm_forward`，任务同步 BN 另有 `sync_batch_norm` Function 入口。稀疏卷积既可提供具备既有接口的 `extension_modules["sparse"]`，也可逐项覆盖 `sparse.SubMConv3d`、`sparse.SparseConvTensor`、`sparse.features`、`sparse.replace_feature`、`sparse.to_dense`、`sparse.is_sparse_convolution` 和 `sparse.sequential_forward`，适配不同的原生张量表示。

标准通信调用使用 `distributed.*` 注册项。例如插件可覆盖 `distributed.wrap_model` 或 `distributed.all_reduce`，无需更改 Runner。当前设备绑定与默认通信名也由运行时管理。`link_sync_batch_norm`、`legacy_link.*`、`build_fused_optimizer`、`reload_fused_optimizer`、`register_float_modules` 是独立可覆盖操作；没有 linklink 兼容实现时不假装支持旧 linklink 路径。

供应者独立构建、安装其原生二进制；导入公共包或查询能力时不应编译代码、加载全部 SDK，或产生自动 CPU 回退。约束部署环境的 Python、PyTorch 与 SDK distribution 版本，跨版本重新编译扩展。

## 扩展验收

1. CPU 契约测试必须通过，包括设备命名不同的 `privateuseone` 路由测试、共享实现单操作覆盖、版本错误、缺失能力、导入隔离和参数层级。私有设备测试只验证路由，不声称执行了不存在的设备。
2. 在真机检查设备迁移、同步、随机状态、显存、AMP、每个依赖算子的输入输出与反向；逐精度和变体记录，明确未测项。
3. 对代表模型执行短程训练、推理、checkpoint 恢复和至少双卡通信。使用供应者实际部署的版本组合并保存环境与测试产物。
4. 运行静态检查，确认没有在模型、损失、Runner 中新增厂商名或 SDK 分支；将能力与真机状态分别更新到矩阵和验证报告。

可在独立插件包完成这些工作。已有模型与训练流程保持不变，是新增后端的验收要求。
