# 相对原版 UP 的变更

基线和约束见 [baseline.json](baseline.json)。本分支不是复制旧 `main` 后删目录。

| 增删改 | 目的 | 保持不变 |
| --- | --- | --- |
| 新增平台父类/工厂/类别/厂商绑定 | 集中差异，显式检查能力 | 模型公式与配方 |
| 替换已纳入的设备/算子/模块构造入口 | UP 通过平台选择供应者 | 参数名、初始化、返回结构、参数分组 |
| 原生源码移到平台 | 构建/加载归平台管理 | 原有内核算法；移动不等于删除 |
| 保留旧算子路径和 apply | 外部导入、配置、ONNX 兼容 | 原 symbolic |
| 训练与推理用途分开 | 训练 eval 不切换部署后端 | 训练步骤与普通 PyTorch 推理 |
| eval 模式恢复保护 | 恢复子模块/EMA 模式，异常也恢复 | 验证不更新权重/优化器/scaler |
| 安装与 SDK 延迟加载 | 无 CUDA 可安装公共部分 | GPU 运行仍需真实 SDK |
| 清单、模板、CI、真机工具 | 可追踪、可复现 | 注册/导入不冒充硬件通过 |

## 遇到的问题

| 问题 | 处理 | 边界 |
| --- | --- | --- |
| Python 3.12 移除 collections.Iterable | 使用 collections.abc；原版对照补同一别名 | 不改算法 |
| NumPy 移除 float/int/bool 别名 | 使用原本等价的内置类型 | 不把 float64 改为 float32 |
| 沐曦 PSRoIPool 整数 scratch 为 channels-last | 厂商适配固定 scratch contiguous | 原版失败保留，修后参考单列 |
| SparseInverseConv3d 参数位置不同 | 传递 indice_key 关键字 | 不把它当成 stride |
| 沐曦 sparse dense 为 N...C | 厂商适配转换为 UP 所需 NC... | 输出与梯度单测 |
| MQBench/MSBench 与现代 API 不匹配 | 固定版本、任务目录补丁、SHA 记录 | 不改量化/剪枝规则 |
| 公开 MSBench 不含原版 UP 配方的 FakeSparse | 保留 Amba 两路径失败，单列依赖/API 缺口 | 不用 DefaultFakeSparse 冒充 |
| ONNX 新版默认 exporter 改变 | 保留 legacy exporter | 不宣称部署引擎通过 |
| SegFormer 默认 Dropout 在导入时调用平台 | 延迟到选择厂商后解析，缓存实际模块 | 保留共享默认对象、RNG、参数键；CLI 和真机前向/梯度单独回归 |
| Open3D 在普通点云导入时被强制加载 | 延迟到 PCD 读取函数 | 不替换读取算法 |
| 机械迁移误识别两处 NumPy matmul | 恢复 NumPy 调用，加边界回归 | 原 CPU 几何计算不变 |
| 原版自身重复运行超容差 | 保留重复与失败，不降低阈值 | 不能直接归因重构，也不能判通过 |
| KITTI Numba 精确边界行为难匹配 | 保留候选和失败，沐曦公开入口拒绝 | 不换几何算法 |
| linklink 缺少真实 SDK | 单列接口与缺口 | 不换成 DDP/普通 BN/普通优化器 |

上游历史、LICENSE、配方、旧 `main`、Release 和证据不删除。
大 checkpoint、梯度、编译日志留在证据目录，仓库保存小报告、源码/依赖与 SHA256 索引。
旧多种子和 v1/v2/portable 实验未恢复。本轮不是全数据集精度或全部第三方变体验收，状态见 [验证报告](VALIDATION.md)。
