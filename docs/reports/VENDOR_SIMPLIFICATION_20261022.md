# 厂商入口简化记录

## 范围与设计

基线为 `9bc45d21345190079aa90339e48bc1d4aeafca33`。本轮参考 LightX2V 的[设备后端](https://github.com/ModelTC/LightX2V/tree/main/lightx2v_platform/base)和[训练后端](https://github.com/ModelTC/LightX2V/tree/main/lightx2v_train/lightx2v_train/runtime/backend)的直接继承风格，减少接入必填项，不复制其视频模型或引擎接口。

新厂商只实现设备后端、实际训练/推理差异和缺失算子。公共能力只保留一份；训练中的 eval 仍使用训练后端，独立推理由入口选择。不改 UP 模型算法、配置、原生算子数学实现或实验数值算法，不恢复多随机种子任务。

| 改动 | 目的 / 兼容边界 |
| --- | --- |
| `register_backend(VendorDevice)` / 装饰器 | 默认复用公共执行类，不必手填两份工厂；差异才通过 training / inference 覆盖 |
| 内置厂商直接选择公共训练 / 推理类 | 移除 10 个空执行子类与 CPU 空设备子类，旧模块地址保留一行别名，不增加动态导入机制 |
| 旧 BackendSpec 保持原语义 | None 仍为未支持；类简写的省略参数才表示公共默认，显式 None 可关闭任一用途 |
| 插件示例删除两份空模块与无效构造函数 | 三个 Python 文件演示设备、注册和单项覆盖；不是模拟厂商真机通过 |
| 厂商指南、接口定义和机器契约同步 | 给出最小路径、可选差异与明确验收，不让新厂商照写兼容文件 |

保留 Context 的用途冻结、完整操作路由、显式错误和 UP 兼容门面：这些承担真实行为，不是为目录对称而存在的空壳。31 个旧运行时成员和 73 项语义操作不增减。SDK 和所选用途按需加载；同叫 cuda 不代表同一厂商。

## 验证

本地环境为 Windows / Python 3.13.11 / Torch 2.10.0+cpu；公共构建另用未安装 Torch / UP 的虚拟环境。实际结果：

| 检查 | 结果 |
| --- | --- |
| 新增设备类注册回归 | 26 项通过：公共默认、单侧覆盖、None / 旧规格兼容、延迟加载、不同设备名、单项覆盖和旧地址 |
| 全仓自动发现 | 320 passed，0 failed/skipped，另 3 个子测试通过；2 条既有 legacy ONNX 弃用提示 |
| wheel 安装后，仓库外、无 UP | 56 passed，0 failed/skipped；同样 2 条 ONNX 提示 |
| 公共分发 | 平台与 UP 各自 wheel / sdist 构建成功；无 Torch / UP 环境安装平台 wheel 后 `-I` 导入通过 |
| 可安装插件示例 | 实际安装后训练 / 推理均通过，NMS 来自插件，overlap 来自公共实现 |
| 边界与契约 | 0 违规；31 成员 / 73 操作索引无漂移 |
| 算法、配置和证据保护 | 665 项受保护文件、6 项原字节协议夹具通过；归档清单和历史报告无修改 |
| 文档 | 190 项本地链接通过；README Skill 审计通过，未新增图片或视觉包装 |
| 云端 CI | 提交后由三个 CPU 版本组合及无 Torch 公共构建验证；最终运行链接另行补记 |

CI 自动发现新增测试，并在安装平台 wheel 后实际安装最小插件、分别执行训练和推理路由。未引入新一套测试调度。原始 JUnit 与构建日志保留在本地忽略目录 `dist/vendor-simplification-20261022/`，云端相应 JUnit / 分发包由 workflow artifact 保存。

```bash
python tools/platform/check_hardware_boundaries.py
python tools/platform/export_platform_contract.py --check
python tools/platform/export_accelerator_contract.py --check
python tools/platform/check_project.py
python -m pytest -q
```

独立分发检查：无 Torch 环境执行 `python -m build --no-isolation --sdist --wheel packages/up-platform`，UP 同理构建；另在只有 CPU Torch、未安装 UP 的环境安装平台 wheel，切到仓库外以 `python -I -m pytest` 执行平台测试目录。插件安装和两侧运行示例见[接入指南](../platform/ADDING_VENDOR.md#3-运行示例与配置)。

没有新增 GPU 真机验收。历史 NVIDIA、沐曦通过项与未通过项继续只对[原冻结版本](README.md)成立；不能把结构简化、CPU 测试或 SDK 注册升级为硬件验证。

## 历史保护

沿用前次固定的[665 项保护清单](../architecture/consolidation-protected.json)和[1,505 项归档清单](evidence.manifest.json)，不改写归档、历史报告或原始协议。新增测试和本报告独立记录，不替换旧阶段结论。复现与最小厂商模板见[新增厂商指南](../platform/ADDING_VENDOR.md)。
