# 平台结构迁移验收

日期：2026-10-22。迁移前代码基线为 `0c9a43c7493563f837ad9dd27761cb00bf5fa1ce`。这是 CPU/接口/构建验收，**不是四家硬件、完整模型训练、精度或性能的新验收**。此前暂停的 NVIDIA/沐曦精度复测没有恢复。

## 本轮实际完成

| 检查 | 结果 | 范围 |
| --- | --- | --- |
| 平台与 UP CPU 回归 | 248 passed，0 skipped/failed | 运行时、老接口、数值协议、证据工具、新作用域/插件/用途保护；[JUnit](validation/cpu-junit.xml) |
| 安装后 wheel，仓库外运行 | 19 passed，0 skipped/failed | 不安装 UP，`-I` + 单独 wheel 路径；[JUnit](validation/wheel-only-junit.xml) |
| 无 UP、无 PyTorch 的普通导入 | 通过 | 独立 wheel 的元数据注册可用；不是无 Torch 运行张量 |
| 无 PyTorch 的公共构建 | 两包 wheel/sdist 均通过 | Python 3.13.11；默认不探测/编译 CUDA；产物在忽略目录 dist/platform-refactor-build |
| 独立厂商示例 | 实际执行通过 | 安装示例插件；只覆盖 NMS，overlap 仍来自 portable；CPU 示例不是国产硬件模拟 |
| 31 旧运行时成员 / 73 操作 | 全部有迁移归属与契约，无漂移 | 两个 exporter --check 通过；完整语义见 [API 索引](API_INDEX.md) |
| 静态边界 | 0 违规 | 平台禁止 UP 反向依赖/用途交叉导入，上层禁止设备/SDK/原生扩展绕过；旧门面有明确白名单 |
| 旧 `.apply()` 和参数兼容 | CPU 契约通过 | 测试后端转发前向/梯度；稀疏与 SyncBN 权重名/身份、恢复等原测试保留 |
| ONNX | 有限 CPU 检查通过 | Linear 真正导出并检查 Gemm/参数名；3 类 RoI symbolic 的节点/属性逐项比较；不是全部原生算子导出验收 |
| 训练验证状态 | 通过 | 实际 BaseRunner.evaluate 方法及状态保护；train→eval→train 保持用途/供应者，混合模块状态恢复，异常恢复；不更新权重/optimizer/scaler |

机器记录、归一化源码清单和构建哈希见 [local.json](validation/local.json) 与 [source_manifest.json](validation/source_manifest.json)。源码对照同时确认上层模型目录、历史 validation 证据未改；43 个 NVIDIA C++/CUDA/header 文件仅迁移路径，未编译执行。

本地测试环境：Windows 11、Python 3.13.11、PyTorch 2.10.0+cpu、pytest 9.1.1、NumPy 2.5.4、SciPy 1.18.1、ONNX 1.23.2、pycocotools 2.0.10。该环境不是建议的 GPU 训练环境。导出测试的两条 warning 来自保留 legacy ONNX 路径的弃用提示，不是失败。

测试 UP 集成时只执行与本次迁移有关的方法/契约，不假装完整 Runner、CLI 数据链路和所有业务依赖在此 Windows 环境中通过。

## 未验证与保留结论

- 新路径 NVIDIA 原生扩展尚未在 CUDA 工具链重编译、加载和运行；新包 A800 算子/梯度/AMP/双卡/模型训练仍需真机复验。
- 昇腾、寒武纪仍无真机验收；MetaX 旧快照有限功能与精度结果保留，不能转记为新包源码已通过。
- 原生 RoIAlign 反向非确定性不要求逐位一致；本次没有修改或“修好”其 CUDA 数学实现。
- portable_v1 仍为实验配置，默认关闭，含 GPU FP64 与性能代价；旧失败阶段、修复阶段和冻结 manifest 不变，不宣称跨厂商精度等价。
- spconv、linklink、MQBench、MSBench 的当前结构执行、国产 AMP/多卡、全量 ImageNet/COCO 收敛未验证。
- CI 已更新为先安装独立平台包，再跑平台与兼容测试。云端 CI 只有观察到真实成功才算通过，本地通过不代替其结果。

## 复现

从仓库源码安装，先准备 CPU 或厂商 PyTorch，不覆盖厂商分发：

```bash
python -m pip install setuptools wheel packaging pytest PyYAML numpy scipy onnx pycocotools
python -m pip install --no-deps --no-build-isolation -e ./packages/up-platform
python tools/check_hardware_boundaries.py
python tools/export_accelerator_contract.py --check
python tools/export_platform_contract.py --check
python -m pytest -q packages/up-platform/tests tests
```

无 Torch 构建测试应在另一个干净环境执行，而不是卸载已有厂商环境：

```bash
python -m venv /tmp/up-platform-build
/tmp/up-platform-build/bin/python -m pip install setuptools wheel packaging
UP_BUILD_BACKEND=none /tmp/up-platform-build/bin/python -m pip wheel --no-deps --no-build-isolation ./packages/up-platform
UP_BUILD_BACKEND=none /tmp/up-platform-build/bin/python -m pip wheel --no-deps --no-build-isolation .
```

独立 wheel 测试在只有 CPU Torch/测试依赖、没有 UP 的虚拟环境中安装平台 wheel 后，从仓库外执行：

```bash
cd /tmp
/path/to/test-env/bin/python -m pytest -q /path/to/repo/packages/up-platform/tests \
  -c /path/to/repo/packages/up-platform/pyproject.toml
```

厂商接入最终还需按[新增厂商验收](ADDING_VENDOR.md#8-交付与验收)在目标硬件分别完成训练与推理，不以这里的结构测试替代。
