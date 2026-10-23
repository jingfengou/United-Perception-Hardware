# linklink 依赖缺口

状态：**阻塞，未通过验收**。核查日期：2026-10-23。

本分支仍以原版 UP 的 `spring.linklink` API 为目标，不将 PyTorch DDP、
普通 BatchNorm 或 SGD/AdamW 冒充 linklink 实现。

## 已核查

- 原版 UP 的 `requirements.txt` 和 `easy_setup.sh` 不安装 `spring.linklink`。
- 当前 NVIDIA 和沐曦验证解释器均没有这个 SDK。
- [DeCLIP 的 linklink](https://github.com/Sense-GVT/DeCLIP/blob/main/linklink/__init__.py)
  是基于 `torch.distributed` 的接口改写，不包含 UP 所需的完整 FP16 和融合优化器。
- [TFMQ-DM 的 linklink](https://github.com/ModelTC/TFMQ-DM/blob/main/linklink/__init__.py)
  也转发到 `torch.distributed`；其 `SyncBatchNorm2d` 明确使用普通 BatchNorm2d。
- [PyPI 同名包](https://pypi.org/project/linklink/) 用于 URL 拼接，与 GPU SDK 无关。

上述公开代码均不能作为本计划的等价依赖，因此没有安装或复制进项目。
本次检索未找到可验证的完整 SDK，不等于证明任何渠道都不存在。

## 待交付的真实依赖

需要对应 Python/PyTorch/设备版本的 `spring.linklink` 安装包或源码，并执行：

1. initialize/finalize、rank/group、同步与异步集合通信。
2. SyncBatchNorm2d、SyncBNFunc、var mode、分组状态和反向。
3. FP16 模块注册、转换及状态恢复。
4. FusedFP16SGD、FusedFP16AdamW、FusedZeroFP16AdamW 更新与 master-weight 恢复。

PyTorch `dist` 路径单独实现和验证，不能用于把本表的状态改为通过。
