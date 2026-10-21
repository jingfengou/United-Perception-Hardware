# 硬件适配接口定义

本文回答：**接一家新厂商，需要适配哪些函数？参数、返回值、梯度和验收要求在哪里？**

| 文档 | 用途 |
| --- | --- |
| [完整函数索引](API_INDEX.md) | 30 个运行时成员、73 个路由操作、统一入口的完整清单及签名 |
| [机器可读清单](interface_contract.json) | 逐项记录签名、返回、梯度范围、类别、源码和已注册供应者 |
| [新增厂商步骤](ADDING_BACKENDS.md) | 编写插件、注册和配置的操作指南 |
| [能力矩阵](CAPABILITIES.md) | 实现状态和真机验证状态，二者分别记录 |
| [上游硬件依赖清单](upstream_inventory.json) | 262 个调用位置及迁移记录；它不是函数签名清单 |

## 1. 是否需要重新实现全部函数

**不需要。继承共享运行时，检查目标模型用到的操作，只覆盖不兼容部分。**

| 范围 | 接入方的责任 |
| --- | --- |
| 普通 PyTorch | 卷积、矩阵乘法、激活、普通优化器等由厂商 PyTorch 适配，不重复包装 |
| 运行时 | 必须可用；兼容标准 PyTorch 设备 API 的方法直接继承 |
| 专用操作 | 目标模型需要的操作必须可用；复用同设备共享实现或逐项注册 |
| 多卡通信 | 需要多卡时验证通信、DDP 和使用到的 SyncBN；单卡验收不算多卡验收 |
| 稀疏、旧 linklink、量化、剪枝 | 只在启用相应模型或功能时要求，不应阻塞普通模型导入 |
| 构建与加载 | 提供匹配厂商 SDK 的插件/二进制；不能默认加载 NVIDIA 扩展 |

例如，ResNet18 分类配置主要需要运行时和普通 PyTorch；Faster R-CNN 还需要 `naive_nms`、`gpu_iou_overlap`、`roi_align`；其他检测和 3D 模型可能需要 Focal Loss、3D IoU、点云池化或稀疏卷积。实际依赖随配置变化，这不是自动生成的模型依赖分析。

未实现能力必须明确报错，不允许空函数、固定输出、伪梯度或静默 CPU 计算。声明“注册成功”不等于“该厂商已支持全部模型”。

## 2. 后端与注册契约

厂商类继承 `up.accelerator.TorchAccelerator`，其构造函数接受 `device=None`。

| 属性 | 定义 |
| --- | --- |
| `name` | 后端身份，由注册名称确定，不能从设备名推断 |
| `device_type` | 厂商 PyTorch 认识的设备命名空间，例如 cuda、npu、mlu |
| `communication_backend` | 未显式指定传输时使用的默认通信名 |
| `op_fallbacks` | 显式允许复用的实现集合；portable 是同设备共享实现，不是 CPU |
| `extension_modules` | 可选供应者映射，例如 sparse、link、quantization、pruning 到模块导入路径 |

`register_backend(name, factory=None, replace=False, probe=None)` 支持装饰器、工厂和 `module:attribute` 延迟路径。`probe` 不应分配设备；多个探测结果命中时应报歧义，生产环境推荐显式配置。

`register_op(backend, name, implementation=None, backward=None, replace=False, dtypes=None, variants=None)` 注册**整个语义操作**。可微操作由实现持有前向、反向及 context；`backward=True` 不会自动生成梯度。重名默认报错，仅显式 `replace=True` 才替换。

查找顺序是 `(当前后端名,) + op_fallbacks`。设备名相同不表示 SDK、二进制或语义兼容。`dtypes=None` 和 `variants=None` 只表示未声明，不代表所有精度、形状和变体都支持。

`configure_accelerator(config=None, backend=None, device=None, reset=False)` 选择后端、加载指定插件并校验环境。`get_accelerator()` 返回当前实例。训练中不要切换后端；reset 主要用于隔离测试。`prepare_args()` 和 `activate_args()` 服务于配置读取和启动器。

## 3. 运行时接口

完整签名见[运行时索引](API_INDEX.md#runtime)，以下定义实现义务。兼容的方法直接继承。

| 类别 | 成员 | 输入、返回及要求 |
| --- | --- | --- |
| 构造与属性 | `__init__`、`api`、`device` | 构造接受设备字符串/对象，拒绝外来设备类型；api 为供应者运行时，CPU 为 None；device 返回含逻辑索引的 torch.device |
| 设备查询 | `is_available`、`device_count`、`current_device` | 分别返回真实可用性、可见数量和当前逻辑索引，不能只判断 SDK 能否导入 |
| 设备绑定 | `resolve_device`、`set_device`、`is_accelerator_tensor` | 支持设备对象/字符串/逻辑索引；遵守可见设备映射；CPU 张量不算加速器张量 |
| 环境 | `validate_environment` | 校验 Python/发行包版本约束；不自动安装、降级或替换厂商 PyTorch |
| 迁移 | `to_device` | 迁移 Tensor/Module，递归处理字典、列表、元组；保持容器、dtype、梯度；遵守 non_blocking |
| 旧式构造 | `float_tensor`、`int_tensor` | 保持尺寸构造/数据构造区别；torch.Size 是尺寸；分别产生当前设备 float32/int32 张量 |
| RNG | `manual_seed`、`get_rng_state`、`set_rng_state` | 设置并恢复相应设备 RNG；设备范围与旧 API 一致，不擅自扩大为所有设备；CPU/Python/NumPy 种子仍由框架原流程处理 |
| 同步 | `synchronize` | 等待相应设备工作完成；CPU 可无操作 |
| 内存 | `empty_cache`、`reset_peak_memory_stats`、`memory_allocated`、`max_memory_allocated` | 缓存释放不破坏活跃张量；内存统计单位为字节；CPU 共享统计为 0 |
| 属性 | `get_device_properties` | 返回调用方所需属性，例如 total_memory；CPU 明确报无加速器属性 |
| AMP | `autocast`、`grad_scaler` | 前者为上下文管理器；后者兼容 scale/step/update/state_dict/load_state_dict；不支持的精度和不可用设备明确报错 |
| 卷积设置 | `configure_convolution`、`convolution_flags` | 管理厂商卷积开关；临时上下文退出后恢复原状态，不在模型中出现 cuDNN/厂商分支 |
| 健康检查 | `health_check` | 在目标设备创建张量、计算并同步，返回 True 或明确报错 |
| 路由 | `call`、`dependency`、`capabilities` | 调用完整操作、延迟加载指定可选库、查询注册情况；缺失能力报后端和原因；能力查询不是硬件测试 |

随机状态接口不承诺不同厂商在同一个 seed/state 下生成逐位相同的随机张量或 Dropout 掩码。跨厂商比较必须检查实际数据/掩码，不能只比较状态哈希；确定性请求也必须以重复前向、反向实测确认。C500 上已定位的 Dropout 映射和卷积反向重复性问题见[差异定位](CROSS_VENDOR_DIAGNOSIS.md)。CUDA-shaped 设备的卷积库开关由 NVIDIA/MetaX 共享实现管理，UP CLI 使用 `--nocudnn`，直接构造 Runner 则设 `config['args']['nocudnn']=True`；不新增框架 CPU 回退。默认不关闭，不能把它宣称为所有模型的精度保证。

## 4. 专用算子的输入输出和梯度

统一入口为 `ops.<操作名>()` 或 `ops.call("操作名", ...)`。以下 20 项的参数顺序及默认值直接从原函数/Function.forward 提取，见[签名索引](API_INDEX.md#routed-operations)。旧 `.apply()` 门面继续有效，不能只替换前向而把反向留在原厂商扩展。

| 操作 | 输入和关键语义 | 返回和梯度 |
| --- | --- | --- |
| `roi_align` | features 为 NCHW，rois 为 `[R,5]`，首列 batch index；保持空间比例、采样、AVG/MAX、aligned 语义 | `[R,C,Hout,Wout]`；features 梯度，RoI 坐标没有反向契约 |
| `psroi_align` | 保留位置敏感通道、group_size、output_dim、sampling_ratio | `[R,output_dim,G,G]`；features 梯度 |
| `psroi_pool` | 保留位置敏感池化及离散分桶规则 | 同上；features 梯度 |
| `deformable_conv` | input、offset、weight、卷积参数，含 groups 和 deformable_groups | 卷积输出；input、offset、weight 三类梯度 |
| `sigmoid_focal_loss` | logits `[N,C]`、标签 `[N]`、weight_pos、gamma、alpha、类别和 reduction | none 时 `[N,C]`，否则 `[1]`，不能改成 0 维标量；preds 梯度 |
| `softmax_focal_loss` | 保留类别、归一化及忽略标签规则 | `[1]`；preds 梯度 |
| `cross_focal_loss` | 还包含 gt_class_to_avoid，保持排除类别和 reduction 规则 | 与原门面同形状；preds 梯度 |
| `naive_nms` | `[N,5]` 的框坐标和分数；调用方已降序排序；保留框架坐标 offset | 当前设备 int64 保留索引；无反向声明 |
| `softer_nms` | `[N,9]`，含方差与分数，保留各列和更新规则 | 更新框及索引元组；无反向声明 |
| `gpu_iou_overlap` | `[N,4]`、`[M,4]`；IoU/IoF/IoS；inclusive 坐标、分母下界 1 | `[N,M]`；上游空输入返回 `[0]`，不得擅改；无反向声明 |
| `boxes_bev_iou_cpu` | 明确命名的原有 CPU BEV 接口；框为 x,y,z,dx,dy,dz,heading | CPU Tensor/NumPy `[N,M]`；不是新增 GPU CPU 回退 |
| `boxes_iou_bev` | 同设备 BEV 旋转框重叠 | `[N,M]`；无完整反向契约 |
| `boxes_iou3d_gpu` | 同设备 3D 框，保持角度和高度约定 | `[N,M]`；BEV 内核没有完整反向，不承诺可微训练 |
| `nms_gpu` | 3D 框、scores、thresh、pre_maxsize；排序截断保持一致 | `(int64 索引, None)`；无反向声明 |
| `nms_normal_gpu` | 原有非旋转 NMS 规则，不能与前一项混用 | 同上 |
| `points_in_boxes_cpu` | points `[N,3]`、boxes `[M,7]` | CPU int32 `[M,N]` 命中矩阵 |
| `points_in_boxes_gpu` | points `[B,N,3]`、boxes `[B,M,7]` | 当前设备 int32 `[B,N]`，框索引或 -1 |
| `roiaware_pool3d` | rois、pts、pts_feature、out_size、max_pts_each_voxel、max/avg | `[R,X,Y,Z,C]`；pts_feature 梯度及池化所需状态 |
| `roipoint_pool3d` | points、features、boxes3d、扩大宽度、采样点数 S | `(features[B,M,S,3+C], int32 空标记[B,M])`；上游无反向，不新增伪梯度 |
| `rotate_iou_eval` | NumPy 旋转框与 criterion；以整个评估操作为边界 | NumPy float32 `[N,K]`；不在上层暴露 Numba/CUDA |

另有 `torchvision_nms(boxes, scores, iou_threshold)`：统一使用三个位置参数，供应者内部的第三参数名可能不同；内部按分数排序、使用 exclusive 坐标，返回当前设备 int64 索引。其契约不同于 `naive_nms`，不能直接互换。

NVIDIA `softer_nms` 保留上游已有的显式 CPU 内部阶段及返回原设备行为；这不授权新后端为其他数值算子增加静默 CPU 代算。

所有实现必须保持输入输出设备、索引 dtype、空输入、连续性要求、坐标和角度、返回元组、异常及不支持变体。允许的精度和形状需逐项声明、测试。高阶梯度不自动保证。

已有 ONNX symbolic 的节点、domain、属性应保持；无法导出时明确报错，不能把 CPU 导出占位函数当数值实现。已测变体见[验证报告](VALIDATION.md)。

**确定性不是自动保证。** 本轮实测原版和改造版的原生 RoIAlign 反向均使用 atomicAdd：即使 PyTorch 全局启用 deterministic algorithms，固定输入和固定上游梯度重复反向仍有微小差异。应分别记录算子容差一致、逐位重复性和训练精度统计，不能把固定随机种子当作逐位确定的证明。相关真实数据对照和诊断见[训练报告](TRAINING_ACCURACY.md)；这也不证明所有检测差异都仅由该算子造成。

## 5. 专用层和稀疏张量

`sync_batch_norm` 是完整 SyncBatchNorm Function；`sync_batch_norm_forward(module, value)` 需更新 running statistics，并保留 affine 参数、输入形状和梯度。`link_sync_batch_norm` 只用于旧 linklink 路径，需完整 SDK autograd。

| 路由项 | 契约 |
| --- | --- |
| `sparse.SubMConv3d`、`sparse.SparseConv3d`、`sparse.SparseInverseConv3d` | 返回供应者模块；保持旧构造参数、参数布局和 checkpoint key |
| `sparse.SparseConvTensor` | 保留 feature、索引顺序、空间尺寸和 batch_size 的构造契约 |
| `sparse.features(value)` | 返回 `[有效点数,通道数]` 特征，不断开 autograd |
| `sparse.replace_feature(value, new_features)` | 返回替换特征后的表示，保留几何和索引 |
| `sparse.to_dense(value)` | 返回供应者原有布局的稠密张量，保持梯度 |
| `sparse.is_sparse_convolution(module)` | 返回 bool |
| `sparse.sequential_forward(sequence, value)` | 顺序执行模块；普通模块作用于特征，不丢失稀疏几何 |

外部门面还包括 `layers.SyncBatchNorm`、`layers.LinkSyncBatchNorm`、`sparse.SparseModule`、`sparse.SparseSequential`。不要为了路由额外嵌套参数模块而改变权重名称。旧 linklink 层构造仍需 `link` 供应者，仅注册前向不能替代 SDK 类。

## 6. 分布式

21 个 `distributed.*` 路由项全部列在[索引](API_INDEX.md)。接口保留对应 PyTorch 版本的参数、原地更新、异步 Work 和进程组语义。索引列出核心签名；版本新增参数仍需相应供应者支持。

- 生命周期/查询：init_process_group、new_group、destroy_process_group、is_available、is_initialized、get_rank、get_world_size、get_backend。
- 张量通信：all_reduce、broadcast、all_gather、gather、barrier、reduce、reduce_scatter、all_gather_into_tensor。
- 对象通信：broadcast_object_list、all_gather_object。
- 框架扩展：wrap_model、collective_device、cpu_group。

`init_process_group` 先绑定 local rank，backend 为 None 时使用后端默认通信名。`wrap_model` 返回 DDP 兼容对象，不改变参数身份。`collective_device` 对 Gloo 返回 CPU，对设备通信返回当前设备；`cpu_group` 是原有对象通信所需 Gloo 组，不是数值回退。

原命令 `--backend dist/linklink` 仍选择框架通信方式；配置 `runtime.accelerator.backend` 选择硬件厂商，二者不是一个参数。

## 7. 可选工具链、旧 SDK 和构建

| 路由项 | 契约 |
| --- | --- |
| `build_fused_optimizer(config)` | 接受 type/kwargs，返回保留参数组和 state_dict 契约的优化器 |
| `reload_fused_optimizer(optimizer)` | 恢复 FP16/master 参数绑定，原地更新 |
| `register_float_modules(modules, initialize=False)` | 注册需保留浮点精度的模块及可选初始化 |
| `quantization_backend_types()` | 返回平台名称到供应者 BackendType 的映射 |

`legacy_link.*` 共 15 项：allreduce、broadcast、allgather、gather、is_initialized、get_rank、get_world_size、barrier、allreduce_async、new_group、get_local_rank、initialize、synchronize、get_backend、finalize。

旧 SDK 路径按固定版本的供应者接口委托，不伪造“适用于所有 SDK 版本”的强类型签名。启用方必须同时固定 SDK 版本及其文档；新厂商走标准 PyTorch 通信/优化器时无需实现旧 linklink。

`layers.OptionalModule(provider, path='')`、`layers.optional_symbol(provider, module, name)` 延迟访问可选库，不是数值操作的注册方式。

构建入口 `up.accelerator.build.get_extensions()`：`UP_BUILD_BACKEND=none` 构建公共包，nvidia 构建原扩展，auto 根据构建环境选择。其他厂商插件独立构建安装，不把所有 SDK 放进公共 setup.py。`native.load_extension()` 是 NVIDIA 内部边界，不是通用厂商接口。

## 8. 完整性检查与验收

安装了 CPU 或厂商 PyTorch 即可运行，不需要 CUDA、GPU 或可选 SDK：

```bash
python tools/export_accelerator_contract.py --check
```

新增操作但缺少语义定义、签名变化、注册项或运行时成员变化却未更新清单，检查会失败。修改经过审查后重新生成：

```bash
python tools/export_accelerator_contract.py --write
```

这是**接口覆盖检查**，不是厂商实现正确性证明。最终需比较前向、所需梯度、真实数据训练指标、checkpoint 恢复，以及启用功能的双卡/导出行为，分别记录未实现、已注册、已执行和已验证状态。
