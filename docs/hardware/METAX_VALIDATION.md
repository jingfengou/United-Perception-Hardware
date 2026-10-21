# 沐曦 C500 四模型测试记录

## 当前状态

2026-10-21 已登录目标服务器，完成环境预检，并上传代码快照。**尚未执行 YOLO、ResNet、DETR、ViT 的显卡计算或模型测试，不能标记为通过。** 本次阻塞发生在框架执行之前，不能据此判断模型或后端实现是否兼容。

| 项目 | 实际检查结果 |
| --- | --- |
| 显卡 | 8 张 MetaX C500，每张 65536 MiB；采样时利用率为 0%，无计算进程 |
| 驱动和 SDK | Kernel driver 3.6.11，MACA 3.5.3.18，mx-smi 2.2.12 |
| 当前 Python | 系统 Python 3.10.12，没有 torch、torchvision、numpy 或 pip |
| 显卡访问 | `/dev/mxcd` 和全部 8 个 render 节点属于 video 组，权限 0660；当前账号不在该组，无读写权限 |
| Docker | 客户端存在，但当前账号无权访问 Docker socket；非交互 sudo 需要密码 |
| 磁盘 | 系统盘接近满载；代码已放在可写、空间充足的 `/mnt/afs`，未在系统盘安装环境 |

结构化证据见 [preflight.json](validation/metax_20261021/preflight.json)。`mx-smi` 能显示显卡，不等于当前账号能执行张量计算。

## 已准备的代码

- 快照提交：`06d88b48cd9e181c162b5e3c97115123fde86efb`。
- 工作目录：`/mnt/afs/oujingfeng/up-hardware-metax-20261021`；源码在 `source/`。
- 压缩包 SHA-256：`aab49782e1c806df05aaf8f5623cf8768b3154f880c29b40eced8b8ccf9abd83`，服务器与本地一致。
- 459 个 Python 文件在服务器 Python 3.10 下通过 AST 解析；**这只是语法检查，不是导入、算子或模型测试**。
- 快照来自 Git 已跟踪文件，未上传 SSH 密钥。仓库忽略 `*.pem`，服务器 IP 和密钥内容不写入验证产物。

## 继续测试所需条件

1. 管理员允许该账号访问显卡，例如执行以下命令，然后重新建立 SSH 会话：

   ```bash
   sudo usermod -aG video oujingfeng
   ```

2. 提供可用的沐曦 PyTorch/TorchVision 环境路径、可访问的容器，或与部署版本匹配的厂商安装包。无需向聊天发送 sudo 密码。现有容器的使用方式由管理员决定，不自动修改 Docker 权限或全局驱动。
3. 如自行安装，将虚拟环境、缓存和数据放在独立网络盘目录，不使用当前接近满载的系统盘。保留厂商 PyTorch/TorchVision，不替换为官方 NVIDIA wheel。

厂商提供 [mcPyTorch 安装说明](https://developer.metax-tech.com/api/client/document/preview/AI%20%E5%BA%94%E7%94%A8%E5%BC%80%E5%8F%82%E8%80%83/mcPyTorch%E7%94%A8%E6%88%B7%E6%8C%87%E5%8D%97/%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97/latest/split_files/%E5%BF%AB%E9%80%9F%E5%AE%89%E8%A3%85.html)，包含 wheel 和容器方式；具体安装包仍需核对当前服务器驱动、SDK 和 Python 的版本兼容性，不能把最新版文档当成本机组合已通过验证的证明。

## 四模型验收范围

| 模型 | 框架来源 | 本次状态 |
| --- | --- | --- |
| YOLOv5-S SiLU | 原 UP 模型 | 未运行 |
| ResNet18 | 原 UP 模型 | 未运行 |
| DETR R50 | 已有外部官方 DETR 适配器，经 UP Runner 执行；不是上游原生模型 | 未运行 |
| ViT-B/16 | 原 UP 模型 | 未运行 |

环境可用后，先验证 `backend: metax`、`device: cuda` 下的张量计算和所需算子，再检查四模型前向、反向、优化器更新、推理及新建模型恢复 checkpoint；真实数据训练指标单独记录。不可将 CPU 计算、注册成功或本次环境预检视为上述项目通过。

现有 `validate_training_accuracy.py`、`run_multiseed_accuracy.py` 是 [NVIDIA 成对训练协议](MULTISEED_ACCURACY.md) 的驱动，仍固定使用 NVIDIA 后端和设备预检查；不能原样用它们宣称完成沐曦测试。正式执行时需使用明确选择 MetaX 的验证入口，并保存有效配置、软件版本、数据/权重哈希及完整日志。跨厂商结果也不预设逐位一致。
