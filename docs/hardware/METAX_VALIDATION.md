# 沐曦 C500 四模型验证

后续真实数据训练与 NVIDIA 的多种子对照单独记录在[跨厂商训练报告](CROSS_VENDOR_ACCURACY.md)。本页保留早期两步合成检查，不能将两轮证据混为完整训练。

## 结论与范围

2026-10-21 在服务器已有的沐曦 PyTorch Docker 镜像中完成了单张 C500 的验证。**显式关闭 TF32、使用严格 FP32 时，UP 的 ResNet18、YOLOv5-S、ViT-B/16，以及已有外部官方 DETR 适配器，均通过两步合成训练、有限梯度、推理和新建模型/优化器恢复检查。** 没有修改 `up/` 模型或硬件后端实现，没有加载 UP 的 NVIDIA `_C`，没有新增静默 CPU 回退。

这不是 ImageNet/COCO 真实数据精度、完整 Runner 训练、收敛、性能或多卡验收，也不表示全部 UP 模型和专用算子均支持 C500。

| 模型 | 本次输入与类别 | 严格 FP32 前后向/推理/恢复 | 两步 loss |
| --- | --- | --- | --- |
| UP ResNet18 | batch2，64x64，1000 类，随机初始化 | 通过；另完成一次 AMP/GradScaler 路径检查 | 7.768791 / 7.765743 |
| UP YOLOv5-S SiLU | batch2，640x640，80 个前景类，随机初始化 | 通过；分类、目标性、框回归损失均执行 | 0.485236 / 0.487079 |
| UP ViT-B/16 | batch2，224x224，1000 类，完整 Base 结构，随机初始化 | 通过 | 7.514354 / 7.420273 |
| 官方 DETR R50 适配器 | batch2，128x128，person/no-object，复用此前验证的官方权重 | 通过；不是上游 UP 自有模型 | 15.280449 / 31.653660 |

原始结果：[汇总](validation/metax_20261021/results/summary.json)、[严格 FP32 四份报告](validation/metax_20261021/results/strict-fp32/)、[执行记录](validation/metax_20261021/results/execution-cohorts.json)。只检查 loss 有限，不把两步 loss 的变化解释为收敛。

## 默认 TF32 的失败不能忽略

当前镜像默认 `float32_matmul_precision=high`，matmul 和 cuDNN 的 TF32 均开启。本轮默认设置复测中，YOLO、ViT 通过，但 ResNet 和 DETR 的恢复一致性检查失败。早期 ResNet 默认设置曾通过，不能只选该次结果宣称稳定；这些报告全部保留在 [默认设置对照](validation/metax_20261021/results/default-current-driver/)及 results 根目录。

DETR 的[诊断](validation/metax_20261021/results/detr-restore-diagnostic.json)显示：保存恢复后的参数完全相同，两边所有模块都处于 eval 模式，但原模型自身重复推理也有漂移。

| 同一组模型/输入的诊断 | 原模型重复最大差 | 恢复模型重复最大差 | 两模型输出最大差 | 原恢复容差检查 |
| --- | --- | --- | --- | --- |
| 默认 TF32 | 0.394058 | 0.465549 | 0.309875 | 失败 |
| 显式关闭 matmul/cuDNN TF32 | 0 | 0 | 0 | 通过 |
| 关闭 TF32 后另禁用 MHA fastpath | 0 | 0 | 0 | 通过，不是通过所必需 |

这是本次镜像/驱动组合下的观测，尚未定位到具体内核，也不证明其他版本都存在同样问题。没有放宽 `atol=1e-5, rtol=1e-4`，没有舍入或回放梯度。四模型使用相同的显式严格 FP32 验证策略重新运行，不能将其通过结果写成“默认设置无条件通过”。

`--strict-fp32` 是验证工具的显式选项，不会自动改变普通模型或 Runner 的默认配置。当前镜像还实测支持以下启动环境变量，读回为 highest/false/false，见[环境变量读回](validation/metax_20261021/deployment-precision.json)：

```bash
export PYTORCH_DISABLE_CUDA_CUDNN_TF32=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
```

## 已验证环境

| 项目 | 版本/状态 |
| --- | --- |
| 宿主机 | Ubuntu 22.04.5；Kernel driver 3.6.11，宿主机 MACA 3.5.3.18 |
| GPU | 8 张 C500，每张 65536 MiB；本轮只使用物理 GPU 0 |
| 已有镜像 | `cr.metax-tech.com/public-library/maca-pytorch:3.8.1.2-torch2.10-py312-ubuntu24.04-amd64` |
| 镜像 ID | `sha256:1a10a6cc56741f5e84da5dd6f0aa9dc4cfd420d66188e43983c9a3efd9d85ab0` |
| 容器 | `up-metax-validate-20261021-01`，独立新建，UID/GID 1009，附加设备组 44 |
| Python / PyTorch / TorchVision | 3.12.11 / `2.10.0+metax3.8.1.0` / `0.25.0+metax3.8.1.0` |
| NumPy / SciPy | 1.26.4 / 1.16.0，保留镜像已有版本 |
| 应用环境 | 网络盘上的 `--system-site-packages` 虚拟环境；`pip check` 通过 |
| 公共包 | `UP_BUILD_BACKEND=none` 的 editable 安装通过，没有编译 NVIDIA 扩展 |

完整版本与实际导入路径见 [environment.json](validation/metax_20261021/results/environment.json)，应用依赖见 [metax-c500-validation.txt](../../requirements/metax-c500-validation.txt)。它只针对这个已有厂商环境，不是跨版本/跨厂商通用安装锁。Torch 报告的 CUDA 11.6 是厂商兼容接口信息，实际设备为 MetaX C500。

额外在自建容器内安装了 `libgl1` 及其依赖以支持 OpenCV；没有更改宿主机驱动、SDK 或其他用户的容器。虚拟环境、数据和缓存放在 `/mnt/afs`，Docker 可写层仍使用系统盘；系统盘接近满载，不应盲目下载新镜像。

## 如何进入当前容器

Docker 提供厂商环境，框架仍按 `UP 模型 -> 统一接口 -> metax 后端 -> 厂商 PyTorch/SDK -> C500` 调用。用户提供的 [C500 示例](https://github.com/xumy1995/model-verify/blob/120d9af31fb022ccec71f1060eee892491341ca1/mx-c500/README.md) 是部署参考，其 Transformers/Ultralytics 评测不是本次 UP 验收。

免密 sudo 已生效，可进入已保留的任务容器：

```bash
sudo docker exec -it \
  -e USER=oujingfeng -e LOGNAME=oujingfeng \
  -e DEFAULT_TASKS=cls:det \
  -e PYTORCH_DISABLE_CUDA_CUDNN_TF32=1 \
  -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
  up-metax-validate-20261021-01 bash

source /workspace/up-run/venv/bin/activate
cd /workspace/up-run/source
```

映射关系为 `/mnt/afs/oujingfeng/up-hardware-metax-20261021:/workspace/up-run`。容器以非 root 用户运行，显式 USER/LOGNAME 避免镜像没有 UID 1009 的 passwd 条目时 Torch 编译缓存查询用户名失败。Docker 启动时已映射 `/dev/dri`、`/dev/mxcd`，附加组 44，设置 `CUDA_VISIBLE_DEVICES=0`，未使用 privileged 或 host network。

宿主账号通过 sudo 启动有设备权限的容器时，不必同时加入宿主机 video 组。直接裸机运行才需要另行授予显卡节点权限。早期[预检阻塞记录](validation/metax_20261021/preflight.json)保留为历史，不能继续解读成当前 Docker 不可用。

## 复现检查

在上述容器和虚拟环境中执行。DETR 清单指向已部署的官方源码和 tensor-only 权重，见 [detr-manifest.json](validation/metax_20261021/results/detr-manifest.json)；使用新环境时需要重新准备这些文件并校验哈希。

```bash
python tools/validate_models.py --backend metax --strict-fp32 --model resnet18 --output /workspace/up-run/results/recheck-resnet18.json
python tools/validate_models.py --backend metax --strict-fp32 --model yolov5s --output /workspace/up-run/results/recheck-yolov5s.json
python tools/validate_models.py --backend metax --strict-fp32 --model vit_b16 --output /workspace/up-run/results/recheck-vit_b16.json
python tools/validate_models.py --backend metax --strict-fp32 --model detr --detr-manifest /workspace/up-run/results/detr-manifest.json --output /workspace/up-run/results/recheck-detr.json
python tools/validate_vendor_runtime.py --backend metax --output /workspace/up-run/results/recheck-runtime.json
```

模型入口显式选择 `metax`，原有 `validate_models.py` 的默认后端仍为 nvidia，未改变默认精度行为。后续 `validate_training_accuracy.py` 已增加显式 `--backend metax`，跨厂商汇总使用新的 `run_cross_vendor_accuracy.py`；旧 `run_multiseed_accuracy.py` 的原版/路由版环境相同检查不适用于跨厂商比较。

验证入口恢复了 `collections.Iterable`、`np.float`、`np.int` 三项历史别名，与此前 NVIDIA 检查一致。这不代表未修改的原始 CLI 对所有现代依赖组合都已通过。

## 其他证据和限制

- [运行时和算子检查](validation/metax_20261021/results/runtime-ops.json)：设备迁移、随机状态恢复、旧式张量构造、同步/显存与缓存操作通过；NMS/IoU 的 FP16/FP32、两种 NMS offset、空 IoU，以及一个 FP32 TorchVision NMS 用例通过。输入和计算均在 C500，实际供应者为 portable，不是 CPU 数值回退。固定小用例不覆盖任意形状/变体。
- [CPU 契约测试](validation/metax_20261021/results/cpu-contracts.final.xml)：隐藏显卡后 126 项通过。初次部署漏了测试清单，另有 4 项 fixture 缺失失败，原始 XML 保留，不将其计为模型执行结果。
- [源码核对](validation/metax_20261021/source_integrity.json)：436 个 `up/` Python 文件与本地一致，比较时只归一化 CRLF/LF；应用与后端代码没有修改。DETR 的 28 个 Python 文件和 checkpoint 复用此前 A800 的已校验字节，Windows Git archive 的行尾差异单独核对后解决，没有改算法。
- 四模型主验收共执行 8 个 FP32 优化器步骤，另检查一次 ResNet 的 AMP/GradScaler 路径，未单独断言该次 scaler 是否跳步；默认精度对照及早期独立进程尝试分开保留，不将它们混为更多训练种子。
- 未验收真实数据集精度、全程训练、性能、长期稳定性、双卡通信、所有 AMP 精度、稀疏卷积、RoIAlign、3D/点云、linklink 或量化/剪枝。
- 本次只新增验证工具、测试和依赖/文档，没有改变任何模型、训练算法或 NVIDIA 内核，也没有将厂商 PyTorch 替换为 NVIDIA wheel。测试完成后未留 GPU 计算进程，独立容器保留供复用。
