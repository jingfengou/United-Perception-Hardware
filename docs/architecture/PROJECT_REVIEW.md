# 平台解耦总设计与仓库变更审计

日期：2026-10-22。本文先说明产品目标和模块边界，再核对实际改动、问题与收敛路线。它是对现有工作的**回溯设计审查**，不把事后整理的决定冒充实施前就已具备的管理记录。

## 1. 先明确要交付什么

目标不是另写一套 UP，也不是把所有 PyTorch 运算再包一遍，而是：

> 保留 UP 的模型和业务能力，将设备接口、专用算子及厂商差异集中在可独立安装的平台包；新增厂商不再逐个修改已有模型。

最终应像 UP、LightX2V 一样，主目录能直接看懂、安装入口明确、扩展点稳定。**当前仓库还不是这个收敛后的形态**：核心隔离已经实现，但产品代码、兼容代码、数值实验和大量阶段证据需要进一步分层。

本次只增加审查说明和可再生变更台账，不改运行时、不搬迁或删除历史实验材料，也不恢复暂停的多随机种子复测。

### 审计基准

| 项目 | 固定版本 |
| --- | --- |
| 原版 UP | `52aaeb9ed27a490820696d7754d19f98cd6ecd11` |
| 本次审查的派生版本 | `fcccc05634536c83cd850728e29fdbe811d92cf6` |
| LightX2V 参考版本 | `19c19614b067f1746ac24350dfd097e1c0b48591` |

已核对上游默认分支仍指向上述 UP 提交。统计比较的是两个**已提交 Git 文件树**，不包含本地密钥、虚拟环境、忽略的 dist、未提交修改，也不是 `.git` 历史或网络克隆体积。

## 2. 自上而下的总体设计

```text
用户 / CLI / YAML
       |
UP 业务层：模型、损失、数据、训练配方、Runner、推理流程
       |
UP 接入层：up.accelerator
  旧 API 兼容、配置转换、UP 坐标等业务约定显式传入
       |
up_platform：固定厂商 + 固定 purpose 的 PlatformContext
       |
       +-- training：AMP / GradScaler / DDP / 训练操作与层
       |      训练中的 eval、train --evaluate 仍在这里
       |
       +-- inference：独立推理操作与层 / 现有 ONNX 导出
       |
       +-- common / shared：设备、RNG、内存、通信及明确共用的实现
                     |
              backends/<vendor>
              设备与 SDK / 局部覆盖 / 可选编译扩展
```

只有一个平台实现中心，不再让 `up.accelerator` 和 `up_platform` 各维护一套硬件逻辑。普通 `torch` 设备无关计算、`new_zeros()` 等从输入继承设备的写法继续直接使用。

### 模块责任与禁止事项

| 模块 | 负责 | 不负责 |
| --- | --- | --- |
| `up/` | 模型、损失、数据增强、优化配方、评估指标和业务流程 | 直接绑定厂商 SDK 或 NVIDIA 二进制 |
| `up/accelerator/` | UP 配置与旧接口转换、全局业务约定显式传参 | 复制平台算法、决定厂商内核实现 |
| `up_platform/common/` | 设备、迁移、RNG、显存、集合通信、环境检查 | 学习率、训练步骤、模型结构 |
| `up_platform/training/` | 训练精度、scaler、DDP、梯度同步、训练内验证保护 | 为 eval 自动切到部署后端 |
| `up_platform/inference/` | 独立推理与已有导出适配 | 自动量化、换模型、新增推理引擎 |
| `up_platform/shared/` | 明确适用范围的公共操作、层和稀疏表示 | 用 CPU 静默替代不支持的设备计算 |
| `up_platform/backends/<vendor>/` | 厂商设备、SDK、用途工厂、专用实现和构建加载 | 导入 UP 或改模型训练配方 |
| `experimental/numerics/` | 默认关闭、单独验收的数值实验 | 成为基本厂商接入的强制要求 |

### 为什么训练和推理分开

分开的是**执行契约和扩展入口**，不是复制两份模型或基础设施。训练需要完整反向、优化器状态、scaler、DDP 和可恢复状态；独立部署可能需要不同的操作实现及导出约束。设备、RNG、通信原语和兼容操作则复用同一份实现。

训练中调用 `.eval()` / `no_grad()` 只改变计算行为，不改变平台用途。验证复用原模型并恢复每个子模块进入验证前的状态，不能更新权重、优化器或 scaler。用途由入口固定；当前全局上下文也不支持在一个活跃进程里随意切换 training/inference。

LightX2V 的 [`lightx2v_platform`](https://github.com/ModelTC/LightX2V/tree/19c19614b067f1746ac24350dfd097e1c0b48591/lightx2v_platform) 与 [`lightx2v_train/runtime/backend`](https://github.com/ModelTC/LightX2V/tree/19c19614b067f1746ac24350dfd097e1c0b48591/lightx2v_train/lightx2v_train/runtime/backend)提供了清晰的设备/运行时边界。其 train 是完整训练子系统，并不只是一个硬件适配目录。我们借鉴职责和懒加载方式，不复制其视频模型、训练器或全部目录，也不要求 UP 增加两套平台包。

## 3. 接口和新增厂商的实际工作

稳定入口是 `configure_platform(..., purpose=...)`、`get_platform()`、`register_backend()`、`register_op()`、`capabilities()` 和 `require()`。设备运行时、训练执行和独立推理分别由 `DeviceBackend`、`TrainingBackend`、`InferenceBackend` 负责。

路由顺序为：当前厂商当前用途/common → 该厂商明确允许的共享当前用途/common → 明确报错。训练不会自动借用推理专用实现；设备同叫 `cuda` 也不意味着 MetaX 可以加载 NVIDIA 二进制。

完整契约以[机器可读定义](../../packages/up-platform/src/up_platform/contracts.py)及生成的[接口索引](../platform/API_INDEX.md)为准；[接口说明](../platform/INTERFACES.md)解释设备、dtype、布局、空输入、梯度、状态与导出要求。31 个旧运行时成员、73 个路由操作是迁移覆盖基线，**不是要求每家重新实现 104 个函数**。兼容部分直接继承/注册共享实现，不兼容部分才覆盖。

新增厂商的最小交付：

1. 准备厂商匹配的驱动、SDK、PyTorch/TorchVision 和通信库，记录版本，不换成 NVIDIA wheel。
2. 增加 `device.py`，声明厂商身份、设备名、SDK 懒加载与可用性检查；兼容方法直接继承。
3. 在 `training.py` / `inference.py` 分别声明用途并复用公共实现，可以只先交付一种用途。
4. 按接口补齐缺失操作；应可微的操作同时交付反向与状态，不可微操作不伪造梯度。
5. 注册后端或插件，增加契约及真机测试，分别发布训练/推理支持状态。无需改已有模型与配方。

插件配置接收可导入模块名；工厂/entry point 则按接口规定注册。完整可安装示例见[厂商指南](../platform/ADDING_VENDOR.md)与[示例插件](../../packages/up-platform/examples/)，不要再从实验脚本推测接入规范。

## 4. 与原版 UP 相比，究竟增删改了什么

### 全量统计

| 统计 | 结果 |
| --- | --- |
| 原版 / 当前跟踪文件 | 721 / 2,437 |
| 新增 | 1,716 个文件 |
| 修改 | 97 个文件，其中 `up/` 内 94 个 |
| 原字节迁移 | 43 个 NVIDIA C++/CUDA/header 文件，Git `R100` |
| 识别迁移后实际删除 | 0 个文件 |
| 原有配置 | 168 个文件全部未改 |
| 原始 `requirements.txt`、Apache-2.0 LICENSE | 未改 |
| 跟踪文件字节总量 | 3.84 MB → 94.65 MB，按十进制计 |

**文件未删除，不等于旧行为完全没删。** 例如自动分析埋点、硬编码 gpu-burn 调用确实有删除或替代，见下表。行数比较需忽略 CRLF 差异，不能把换行变动全部算成重写。

逐文件清单见 [`changes.csv`](upstream-20261022/changes.csv)，每行含新增/修改/迁移、原路径/现路径、目的、兼容注意点、前后字节数及首次/最近修改提交。这里的提交追踪是**该路径在派生阶段的修改记录**，不是逐行原始作者归属。摘要见 [`summary.json`](upstream-20261022/summary.json)。同类冻结证据使用同类目的描述，不伪造逐文件设计意图。

### 按职责解释每一类改动

| 类别 / 路径 | 实际增删改 | 目的与需要注意的地方 |
| --- | --- | --- |
| `packages/up-platform/` | 新增 85 文件，接收 43 份原生源码 | 独立安装与测试，阻断对 UP 的反向依赖；厂商专属实现集中管理 |
| `up/accelerator/` | 新增后再收敛为兼容/配置门面 | 保留首轮 API 和插件入口，不要求上层一次改完；仍需明确兼容生命周期 |
| `up/extensions/` | 修改旧导入门面，源码移至 NVIDIA 后端 | 保留算子名字、`.apply()`、完整反向和 ONNX symbolic；不是删除算子 |
| `up/models/`、`up/tasks/**/models/` | 34 个原有模型目录文件有实际内容变化 | 设备调用、SDK/专用操作转平台，少量 NumPy 别名修复；不是“模型目录完全没改” |
| `up/runner/`、任务 Runner | 设备/AMP/DDP 改走平台，eval 状态恢复 | 训练用途固定，避免验证后无条件 `train()` 破坏原冻结状态 |
| `up/apis/`、`up/commands/` | 入口显式选择用途，迁移模型/数据 | 独立推理与训练内验证区分；没有新增部署引擎 |
| `up/utils/env/` | 设备绑定、随机状态、通信与健康检查重接 | 另修原 MPI 环境变量读取错误；不是所有分布式启动器都已实测 |
| `up/utils/model/`、稀疏/量化/剪枝相关任务 | 直接 linklink、spconv、MQBench、MSBench 改为延迟入口 | 未用功能不强制依赖 SDK；不代表这些 SDK 已全部适配/真机验收 |
| `up/__init__.py`、`up/__main__.py`、`up/flags.py` | 显式/延迟注册，业务标志轻量化 | 解决可选依赖与导入环；原自动 `spring_analytics` 分析埋点块被移除，需单列兼容决定 |
| `ModelHelper.to/cuda` | 经平台迁移，并同步 device/dtype 元数据 | 避免迁移后元数据陈旧；旧 `.cuda()` 现在可表示迁移至所选后端，语义扩展需说明 |
| 数据/评估辅助代码 | NumPy 旧类型别名修复，Open3D 在实际读取时导入 | 解决现代环境和二维任务不必要依赖；不是新增模型算法 |
| 根 `setup.py`、`MANIFEST.in` | 去掉无条件 CUDA 探测/编译，依赖平台包 | 普通安装无 CUDA 可构建；Python 最低元数据从 3.6 提到 3.8，不能当作完整业务版本保证 |
| 平台构建 | `UP_BUILD_BACKEND=nvidia` 显式编译对应扩展 | 构建归厂商，安装顺序先平台后 UP；不自动替换厂商 PyTorch |
| `requirements/` | 新增指定验证环境依赖记录 | 与原历史依赖区分，不宣称跨厂商通用 lockfile |
| `tests/`、平台 tests、CI | 新增契约、兼容、用途、导出与证据检查 | CPU 通过只证明对应契约；当前 CI 根测试硬编码列表漏掉近期三组测试 |
| `tools/` | 新增 46 个维护/实验/诊断脚本 | 支撑执行和复查，但平铺且含阶段专用脚本，需要区分可复用基准与一次性实验 |
| `docs/`、README | 新增接口、指南、报告和环境说明 | 有可追踪证据，但当前历史叙述过长、入口重复，需收敛 |
| `.gitignore`、`.gitattributes` | 密钥/产物排除、证据换行规则 | 安全与文件校验，不是数值修复 |

### 必须纠正的比较口径

此前“202 个模型文件没改”指**独立平台拆包前后**，不指相对原版 UP。相对原版有上述 34 个模型目录文件变化。当前目标是模型公式、配方、参数层级保持，而不是承诺模型文件一个字符都没动；也不能由此推导全仓模型都已经数值验收。

原版私有集群的 60 秒 gpu-burn 压测已被所选设备上的轻量 `health_check()` 替代。新方法可验证基本可计算和同步，**不等价于原压测的硬件可靠性覆盖**。

## 5. 遇到了什么问题，如何改进

| 已观察问题 | 采取的改进 | 结果边界与证据 |
| --- | --- | --- |
| 上层散落 `.cuda()`、CUDA API、厂商 SDK 和直接扩展访问 | 设备/操作/层/通信路由与静态边界检查 | 入口集中；专用能力缺失仍须实现，不能只注册名字 |
| 无 CUDA 或未装可选 SDK 时普通导入/构建失败 | 独立包、元数据注册、SDK 懒加载、可选编译 | 见[结构验收](../platform/VALIDATION.md)；无 Torch 普通导入不代表无 Torch 运行模型 |
| 同叫 cuda 可能错载 NVIDIA 二进制 | 厂商身份与设备名分开 | `nvidia/cuda`、`metax/cuda` 分别选择实现 |
| 平台最初依赖 UP 全局标志，无法独立；两套入口难维护 | 拆 `up_platform`，UP 门面显式传入约定 | 平台禁止导入 UP；旧路径保持薄转发 |
| 训练 eval 被误当部署推理、无条件恢复 train 破坏模块状态 | 显式 purpose、训练内验证上下文 | 用途/供应者不切换，恢复调用前状态及异常路径 |
| NVIDIA Faster R-CNN 训练逐步有差异 | 固定输入、权重和真实 RoIAlign 反向重放 | 定位原生非确定性；不能用放宽逐步容差掩盖，见[历史定位](../hardware/TRAINING_ACCURACY.md) |
| 跨厂商训练精度有差异 | 对齐版本、输入、随机性并交叉推理和操作重放 | 并非“路由一致所以结果必然一致”；历史跨厂商精度只有指定模型通过等价检验 |
| C500 ResNet/YOLO/DETR 拆包比较超差 | 固定真实操作数重放原生卷积反向，拒绝只归因种子 | 复现权重梯度非确定性；关闭 DNN 不能完整解决 |
| C500 局部非确定性 | 显式 `metax_deterministic_v1`，仅平台层 FP32 重算卷积权重梯度 | 三模型同种子短程对照通过；native 默认不变，不宣称跨厂商收敛已等价，见[专项报告](../platform/METAX_DETERMINISM.md) |
| 更早的跨厂商数值实验需要固定 RNG/归约顺序 | 增加显式 `portable_v1` | 含 GPU FP64 和性能开销；NVIDIA 完整复测仍暂停，不是基本厂商接入路径 |
| 证据会因换行、源码阶段或失败重跑产生混淆 | 冻结源码/计划/数据哈希、原始失败、独立指标复算 | 各报告只对自己的版本和配置成立；不能混合种子或覆盖失败 |

## 6. 当前结论，不能扩大成什么

以下为已存在报告的回溯核对，本次没有重跑 GPU 测试：

| 范围 | 已有证据 | 仍不代表 |
| --- | --- | --- |
| 当前 NVIDIA 新包 | A800 四模型短程训练、训练内 eval、恢复、独立推理和 32 项操作对照通过 | 当前包全量收敛、所有模型、AMP/双卡全部通过 |
| 当前 MetaX 新包 native | 四模型能运行；同权重推理一致；严格训练数值仅 ViT 通过 | ResNet/YOLO/DETR 的 native 严格训练已通过 |
| 当前 MetaX 局部确定性选项 | 三模型各三次同种子三步训练，共 27 更新、18 eval、9 独立推理；固定 12 卷积案例通过 | 三个随机种子、全量训练、所有形状、跨厂商精度等价 |
| CPU 及接口 | 上一阶段本地 277 项通过，公共构建/普通导入通过 | 云端 CI 已覆盖 277 或所有硬件通过 |
| 历史模型精度与多卡/AMP | 冻结旧版本报告已有特定验证 | 自动升级为当前包的同等级验收 |
| 昇腾、寒武纪 | 后端入口与共享实现 | 真机训练/推理或专用算子完整支持 |

### 尚存问题与处理优先级

| 编号 | 问题 | 建议动作 / 验收方式 |
| --- | --- | --- |
| G01 | 产品、一次性实验和原始证据混合，README 变成流水账 | 主入口只保留当前产品；历史索引指向固定版本与归档校验 |
| G02 | 两套文档树、接口索引与支持结论容易重复维护 | `docs/platform` 为当前唯一手册，历史报告只读；契约索引自动生成检查 |
| G03 | CI 根测试名单未含 `test_metax_determinism.py`、`test_platform_equivalence.py`、`test_platform_evidence.py` | 改为明确分组发现测试，CPU 组不能漏项；真机组单独报告 |
| G04 | `common/runtime.py` 和 context 仍知道 `portable_v1`，实验未完全隔离 | 抽成可选数值策略生命周期，native 不受影响；另立新阶段验证，不改旧冻结源 |
| G05 | 兼容门面、`.cuda()` 语义扩展、分析埋点移除缺少统一生命周期说明 | 保留/弃用/可选恢复逐项决策，测试旧入口；不为目录好看强删兼容 |
| G06 | 普通共享实现名 portable 与数值实验 portable_v1 易混淆 | 文档明确二者不同，后续必要时加兼容别名再改命名 |
| G07 | 环境记录多，但没有覆盖各厂商的受支持组合矩阵 | 按厂商/用途固定实测组合；区分历史依赖、安装约束和验证快照 |
| G08 | 国产专用算子、spconv/linklink/量化剪枝、当前 AMP/多卡等仍缺验证 | 能力逐项记未支持/未验证，不能注册即记通过 |
| G09 | 确定性选项仅覆盖部分一阶 FP32 eager 单卡场景且增加显存/计算开销 | 明列 dtype/变体限制，补性能测量；不承诺 compile、AMP、高阶梯度或全版本 |
| G10 | 全局上下文固定用途 | 作为当前限制写明；训练与独立部署分进程，不无需求扩成复杂多上下文系统 |

## 7. 收敛成简洁项目库的目录目标

下列是**整理目标，不是当前目录已全部搬迁**：

```text
README.md                          当前能力、安装、运行、新增厂商、限制
up/                                UP 业务与薄兼容门面
packages/up-platform/              独立平台产品及厂商插件示例
configs/                           原有模型配方
benchmark/hardware/                可复用的训练/eval/推理对照入口
tools/platform/                    边界、契约、发布与上游差异维护
tests/                             契约 / 兼容 / 集成分组
docs/platform/                     当前用户与厂商开发手册
docs/architecture/                 总设计、决定、上游变更台账
docs/reports/                      简明结果索引与归档清单
```

保留 UP 现有 `benchmark/`（单数），不再造一个平行 benchmarks 目录。平台内部继续按 common/shared/training/inference/backends 分模块，不新增第三层“硬件管理框架”。示例只保留能安装、能测试的最小完整插件。

### 原始材料如何处理

当前 2,437 个跟踪文件中，`docs/hardware/validation/` 与 `docs/platform/validation/` 合计 **1,505 个 / 89.49 MB，占总跟踪字节约 94.5%**。平台包自身仅 128 文件 / 0.46 MB。冗长主要来自证据归档，不应据文件总数判断平台实现本身有 90 MB。

这些证据有价值，不应直接删除。整理时先打固定标签、生成文件清单与 SHA256，将日志/冻结完整源码放版本化 Release assets 或其他可长期访问的制品存储；验证可下载及哈希后，主分支仅留摘要、环境、复现配置、归档链接和校验清单。保留旧提交和原始失败，不改写 Git 历史。这样能减小默认工作树并清晰入口，**不会自动减小已有 Git 历史的克隆体积**。

## 8. 后续开发如何做到有迹可循

管理链路固定为：

```text
需求 / 问题编号 → 边界与契约 → 最小改动 → 自动测试
             → 固定版本真机证据 → 能力矩阵 → 发布
```

每个改动记录：解决什么、不做什么、改哪些模块、默认行为是否改变、兼容风险、测试命令、证据版本和尚未验证内容。普通重构、厂商修复、数值实验分别立项，不能通过不断扩大“适配层”的定义把它们混在一个提交里。

| 阶段 | 交付 | 完成门槛 |
| --- | --- | --- |
| A：现状与设计基线 | 本文、固定 SHA、逐文件台账与 G01–G10 | 可从 Git 再生全部统计，每项变化能定位目的与提交 |
| B：产品入口与材料整理 | 简短 README、唯一手册、可复用 benchmark、归档索引 | 原证据可恢复且哈希一致，旧命令有迁移说明；无运行时语义变化 |
| C：平台边界收紧 | 实验策略独立、兼容清单、环境矩阵、CI 测试分组 | native 默认不变，独立包不依赖 UP，CPU/构建/契约全过 |
| D：候选发布 | 固定版本双平台训练/训练 eval/独立推理回归及简洁支持表 | 证据与源码对应、保留失败、明确未验证能力；不自动恢复旧多种子任务 |

本轮完成 A；B–D 是后续收敛工作，不能因写了设计就标记完成。以后每一阶段单独提交和验收，不边做实验边无限扩张产品入口。

### 已有关键提交的追踪

| 阶段 | 提交 | 主要内容 |
| --- | --- | --- |
| 第一轮隔离 | [`128384d`](https://github.com/jingfengou/United-Perception-Hardware/commit/128384d) | 硬件路由、兼容改动、初始文档与 NVIDIA 检查 |
| NVIDIA 训练对照 | [`5b5f531`](https://github.com/jingfengou/United-Perception-Hardware/commit/5b5f531) 至 [`06d88b4`](https://github.com/jingfengou/United-Perception-Hardware/commit/06d88b4) | 接口契约、非确定性定位与五模型固定预算/多种子证据 |
| MetaX 与跨厂商实验 | [`516aa21`](https://github.com/jingfengou/United-Perception-Hardware/commit/516aa21) 至 [`0c9a43c`](https://github.com/jingfengou/United-Perception-Hardware/commit/0c9a43c) | 功能、真实数据、环境对齐与 portable 数值实验；后者 NVIDIA 全预算仍暂停 |
| 独立包拆分 | [`b13614a`](https://github.com/jingfengou/United-Perception-Hardware/commit/b13614a) | 同仓独立 up-platform，训练/推理分用途 |
| 拆分真机对照 | [`c6e8d3a`](https://github.com/jingfengou/United-Perception-Hardware/commit/c6e8d3a) | NVIDIA/MetaX 短程对照，包含明确失败 |
| C500 局部修复 | [`fcccc05`](https://github.com/jingfengou/United-Perception-Hardware/commit/fcccc05) | 卷积反向定位与显式厂商确定性选项，不改模型/容差 |

## 9. 台账复现

以下命令只读取 Git 并生成报告，不导入 UP、Torch 或 SDK，不启动设备：

```bash
python tools/platform/audit_upstream.py \
  --base 52aaeb9ed27a490820696d7754d19f98cd6ecd11 \
  --target fcccc05634536c83cd850728e29fdbe811d92cf6 \
  --output docs/architecture/upstream-20261022 --check
python -m unittest discover -s tools/platform -p 'test_*.py'
```

去掉 `--check` 可再生这份固定版本的 `summary.json` 和 `changes.csv`。新增未归类路径会明确报错，要求人工说明目的，而不是自动当作设备调用替换。以后审计新版本应新建输出目录，不覆盖本次固定版本台账。
