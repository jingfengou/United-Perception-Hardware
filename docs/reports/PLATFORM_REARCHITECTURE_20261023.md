# LightX2V 风格架构重构

基线 `ad36d7728b19ae6229e22923d2918feeb6b9ad0d`，参考 LightX2V `19c19614b067f1746ac24350dfd097e1c0b48591`。按用户要求调整整个平台注册和执行架构，不只修改目录展示，不重写 UP 模型算法。

## 改动及目的

| 改动 | 目的与兼容边界 |
| --- | --- |
| `base/base.py`、厂商设备及 factory.py | 设备基类、厂商 SDK 和设备选择集中，不再把算子放在设备后端目录 |
| `registry_factory.py` | 设备与完整操作的注册集中；旧 registry 模块与新模块共享同一注册表 |
| `set_ai_device.py`、context.py、薄公共入口 | 显式初始化与用途固定、操作能力检查分开；普通导入无 Torch/SDK 依赖 |
| `training/runtime/backend` 与 inference/factory.py | 独立训练/推理工厂；公共实现复用，有差异才增加覆盖 |
| `ops/<类别>/<厂商>/` 与类别内 common.py | NMS、RoI、IoU、点云、DeformConv、Focal、归一化、稀疏、卷积、GEMM 按类别查找 |
| 厂商 csrc 随算子迁移 | 数学源码保留，构建收集新目录；聚合绑定移到 base，扩展 ABI 名保持不变 |
| 旧模块别名 | 原算子导入、类身份、可选插件地址和 `.apply()` 继续使用同一实现，不复制两套类或缓存 |
| 指南、示例、CI | 新厂商按设备、用途差异和缺失算子接入；CI 自动覆盖新测试、迁移源码保护与独立安装 |

训练内 eval 继续使用当前 training 模型与供应者，恢复原模块状态；独立推理和导出显式选择 inference。native 默认不变，既有 v1/v2/portable_v1 数学定义与失败结论不变。

## 问题与处理

- LightX2V 推理初始化有导入副作用；本框架保留公共包无 Torch/SDK 普通导入和显式初始化，不照搬该副作用。
- 移动 Python 包与普通模块的方式不同：旧包保留包身份，旧普通模块使用别名，确保历史子模块仍可导入。
- NVIDIA 内核原本共用 include 根目录；现在从类别目录收集源码并显式提供各 include 根，仍构建同名 `_C`，不修改内核或 ONNX symbolic。
- 迁移工具给五份原无末尾换行的源码增加了换行；审计发现后已恢复原 EOF，未改旧哈希清单。
- 最终提交对迁移后的 csrc 显式保留原始字节，避免 Git 在新路径二次转换换行；43 份本地原生源码的新旧 Git blob ID 完全相同。初次分发包保留于 platform/，最终分发包另存 platform-final/，不覆盖前一阶段产物。
- 测试对旧私有实现位置的 monkeypatch 和模块名断言迁移到新主路径，原功能断言及容差不变。机器契约只再生路径/归属，不改操作签名、梯度或返回契约。
- 计算定义哈希采用跨 Python 版本的稳定 AST 序列化，忽略版本新增的空字段，避免 Python 3.10/3.12/3.13 的 AST 展示差异造成误报。

## 源码与证据保护

[迁移清单](../architecture/platform-layout.json)记录 77 个一对一路径迁移及 80 个实现检查项，额外覆盖拆分的公共 NMS/IoU 与上下文。Python 计算定义按 AST 核对，只忽略导入语句；C++/CUDA 只进行 CRLF→LF 归一化后校验，不忽略其他内容。原[665 项保护清单](../architecture/consolidation-protected.json)不替换、不重置，按显式迁移目标核对。

UP 模型、损失、配置与业务编排未修改。原 1,505 项证据索引和六个协议夹具保留；历史 GPU 报告不重写，不恢复旧多随机种子任务。

## 验证

本地 Windows / Python 3.13.11 / Torch 2.10.0+cpu 实际结果如下。结构迁移、CPU 契约和打包不等于新架构 GPU 精度验收；本轮没有重新运行完整 GPU 训练，不提升昇腾/寒武纪或任何数值策略的支持等级。

| 检查 | 结果 |
| --- | --- |
| 全仓自动发现 | 409 tests + 3 subtests 通过；JUnit 412 case，零失败、错误与跳过 |
| wheel 安装后、仓库外、无 UP | 81 tests 通过；零失败、错误与跳过 |
| 新增布局/兼容/构建回归 | 29 项，包含用途工厂隔离、模块别名身份、类别内共享、构建源码集合与头文件路径 |
| 源码与协议保护 | 665 项原保护、80 份迁移实现、六个原字节协议夹具通过；43 份迁移原生源码 Git blob ID 相同；UP/config 无改动 |
| 边界与接口 | 零硬件边界违规；31 成员 / 73 操作无漂移 |
| 公共分发 | 未安装 Torch 的独立环境构建平台与 UP 的 wheel/sdist；平台两个分发包内 180 个源码/数据文件均与当前源码逐字节相同，无旧 ops.py 或失效 csrc 残留 |
| 无 Torch 普通导入 | 安装平台 wheel 后在仓库外 `-I` 导入平台和 ops，通过；不加载 UP、Torch、厂商 SDK 或另一用途 |
| 厂商插件 | 安装按类别组织的单 NMS 覆盖示例，训练/推理均路由到插件，overlap 继续复用公共实现 |
| README | Skill 审计通过；图示 900px/360px 渲染检查通过；242 项维护文档本地链接通过 |

两条测试 warning 为既有 legacy ONNX 弃用提示。公共打包保留既有 setuptools 许可元数据/classifiers 警告，未将不相关元数据整理混入平台重构。[机器可读记录](platform-rearchitecture-20261023.json)保留实际环境、范围和分发校验值；本地 JUnit、构建日志及分发包位于忽略目录 `dist/lightx2v-rearchitecture-20261023/`。

```bash
python tools/platform/check_hardware_boundaries.py
python tools/platform/export_platform_contract.py --check
python tools/platform/export_accelerator_contract.py --check
python tools/platform/check_project.py
python -m pytest -q
```

安装后测试还需在仓库外分别使用平台 wheel 和最小厂商插件；无 Torch/UP 环境测试普通导入。NVIDIA 构建配方的源码集合、所有本地头文件与 include 路径通过 CPU 构建契约核查；实际 nvcc 编译与 GPU 运行未在本轮执行。

现行开发入口见[总设计](../architecture/PLATFORM.md)、[接口定义](../platform/INTERFACES.md)、[新增厂商](../platform/ADDING_VENDOR.md)。历史结果仍按各报告的冻结提交、配置和设备解读。
