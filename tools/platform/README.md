# 平台维护工具

本目录只放开发、发布和静态检查；模型硬件验证在 [`benchmark/hardware`](../../benchmark/hardware/README.md)。

```bash
python tools/platform/check_hardware_boundaries.py
python tools/platform/export_platform_contract.py --check
python tools/platform/export_accelerator_contract.py --check
python tools/platform/check_project.py
python -m pytest -q
```

- `check_hardware_boundaries.py`：阻止上层厂商调用、平台反向依赖、用途交叉导入及公共核心依赖实验实现。
- 两个 exporter：维护新平台与旧门面的契约/索引；审查语义后用 `--write` 更新，CI 用 `--check` 防漂移。
- `check_project.py`：核对整理前 UP/配置/数学实现的受保护清单、六个归档夹具及当前文档的本地文件链接。只有首次建立固定基线时使用 `--write-protected`，不能为掩盖算法变更重建基线。
- `archive_evidence.py`：从固定 Git 提交按原 blob 字节归档/校验，不读取密钥、工作区临时产物或执行厂商代码，拒绝覆盖。
- `audit_upstream.py`：生成指定两个 Git 提交之间的变更台账；[已有台账](../../docs/architecture/upstream-20261022/summary.json)固定属于其原版本，后续审计使用新输出目录。
- `build_hardware_inventory.py`：上游硬件依赖清单维护，需本地有上游基线提交。
- `snapshot_hardware_source.py`：为新验证阶段保存源码与哈希，不覆写旧快照。

`pytest.ini` 自动收集本目录维护测试、根兼容/协议测试和独立平台测试，CI 不再手写会漏项的测试文件名单。
