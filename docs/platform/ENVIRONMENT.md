# 环境与构建

环境须同时满足 UP、厂商 Torch/SDK 及模型的可选工具链，不是只看 `torch.cuda.is_available()`。

## 本轮环境

| 项目 | NVIDIA | 沐曦 |
| --- | --- | --- |
| 设备 | RTX 4090，单卡/双卡 | C500，单卡/双卡 |
| Python | 3.12.3 | 3.12.11 |
| PyTorch | 2.10.0+cu126 | 2.10.0+metax3.8.1.0 |
| TorchVision | 0.25.0+cu126 | 0.25.0+metax3.8.1.0 |
| 稀疏卷积 | spconv-cu126 2.3.8，cumm-cu126 0.7.11 | 厂商 spconv 2.1.0+metax3.8.1.0torch2.10 |
| 工具链 | CUDA 编译器、cu126 环境 | MACA 3.8.1.3，驱动 3.6.11 |
| MQBench | 00f477d58ef67c1c1bbe71d14312731fd3a4960d | 同一源码 |
| MSBench | b00632036c8a03cacff6ae94814195e98f1361f8 | 同一源码 |
| spring.linklink | 缺失 | 缺失 |

精确依赖以 [证据索引](VALIDATION.md) 为准。本轮是**同平台原版/重构版**对照，不声称厂商间逐位相同。
包元数据 `Python>=3.10` 不等于所有第三方版本都已验证，真机仅覆盖上表。

## 安装顺序

1. 进入与驱动和设备映射匹配的厂商容器/虚拟环境。Docker 可固定环境，但不是平台接口的硬性要求。
2. 保留厂商 Torch/TorchVision，核对 `python -c "import torch; print(torch.__version__)"`。
3. 安装所用 UP 模型的 Python 依赖，再安装本分支。原版 requirements 的老 SciPy、Numba、ONNX 等不能原样用于 Python 3.12。
4. 按模型需要安装匹配的 spconv/MQBench/MSBench，沐曦不能使用 NVIDIA spconv wheel。
5. 显式编译当前厂商扩展，公共安装不探测 CUDA。

[requirements-common.txt](requirements-common.txt) 是两套 Python 3.12 环境实际共同使用的普通 Python 依赖子集，
不是完整锁文件，也不包含厂商 Torch/SDK、pycocotools、pandas 和所有工具链依赖。应在独立环境中使用，
保留厂商提供的包；其余版本以各平台证据索引为准。

```bash
python -m pip install -r docs/platform/requirements-common.txt
python -m pip install --no-deps -e .
UP_BUILD_BACKEND=nvidia python setup.py build_ext --inplace
# 或：UP_BUILD_BACKEND=metax python setup.py build_ext --inplace
```

沐曦经厂商 Torch 的 CUDAExtension 兼容入口使用 MACA 工具链，产物为独立 `_metax_ops`。
这不是 NVIDIA 二进制。编译器、库路径仍需按厂商配置。

## 可选工具链

验证使用任务目录中的固定依赖源码，不修改共享虚拟环境的厂商包：

```bash
# 在独立依赖目录执行，避免嵌套仓库进入项目提交
git clone https://github.com/ModelTC/MQBench.git MQBench
git -C MQBench checkout 00f477d58ef67c1c1bbe71d14312731fd3a4960d
git clone https://github.com/ModelTC/msbench.git msbench
git -C msbench checkout b00632036c8a03cacff6ae94814195e98f1361f8
python /path/to/UP/tools/prepare_platform_dependencies.py --root "$PWD"
export PYTHONPATH="/path/to/UP:$PWD/MQBench:$PWD/msbench:${PYTHONPATH}"
```

兼容脚本记录修改前后 SHA256，只处理 MQBench 的 Torch 2.10 ONNX 内部导入、legacy exporter、MSBench 已有类缺失的 re-export、TorchVision 已移除 Conv2d 别名。
不改量化/剪枝策略。原始调用和平台调用使用同样补丁；不同依赖版本要重新核查。

Amba 诊断另需 `--msbench-dist torch`，仅给原源码未定义的 `USE_LINKLINK` 设置 `False`，选择其已有的
`torch.distributed` 分支，修改前后哈希另存。它既不提供 linklink，也不改剪枝算法。
即使补齐该开关，公开 MSBench 仍缺少原版 UP 配方的 `FakeSparse` 类，Amba 验收保持阻塞；
没有把 `DefaultFakeSparse` 改名来冒充同一实现。

## 限制

`spring.linklink` 不是 NVIDIA 驱动或 Torch 自带组件，原版安装脚本也未提供。缺口见 [LINKLINK.md](LINKLINK.md)。
NART、Open3D PCD 读取及所有部署引擎未在本轮通过验收；ONNX 导出不等于部署执行。
第三方 SDK 自带许可须独立核查。本仓库不重新分发 MQBench/MSBench 的源码、私有库或厂商 wheel。
