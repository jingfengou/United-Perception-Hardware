"""Generate a Git-based change ledger without importing UP, Torch or vendor SDKs."""
import argparse
import collections
import csv
import io
import json
from pathlib import Path
import subprocess


BASE = "52aaeb9ed27a490820696d7754d19f98cd6ecd11"
ROOT = Path(__file__).resolve().parents[2]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def changes(raw):
    fields = raw.split(b"\0")
    position = 0
    while position < len(fields) and fields[position]:
        status = fields[position].decode()
        old = new = fields[position + 1].decode()
        position += 2
        if status[0] in "RC":
            new = fields[position].decode()
            position += 1
        yield {"status": status, "old_path": "" if status == "A" else old,
               "path": "" if status == "D" else new}


def tree(ref):
    result = {}
    for record in git("ls-tree", "-r", "-l", "-z", ref).split(b"\0"):
        if record:
            metadata, path = record.split(b"\t", 1)
            mode, kind, oid, size = metadata.split()
            result[path.decode()] = {"blob": oid.decode(), "bytes": int(size) if size != b"-" else 0}
    return result


def area(path):
    if path.startswith(("docs/hardware/validation/", "docs/platform/validation/")):
        return "frozen_evidence"
    if path.startswith("docs/") or path.startswith("README"):
        return "documentation"
    for prefix, category in (("packages/up-platform/", "platform_package"), ("up/", "up"),
                             ("tools/", "tools"), ("tests/", "tests"), ("configs/", "configs")):
        if path.startswith(prefix):
            return category
    return "other"


PURPOSES = {
    "up/__init__.py": ("bootstrap", "改为显式/延迟注册，解除普通导入对完整业务与可选 SDK 的依赖", "启动时自动分析埋点被移除；并非纯路径迁移，需单列兼容决定"),
    "up/__main__.py": ("bootstrap", "CLI 解析前显式注册业务组件", "配合延迟导入，不能漏掉历史子命令"),
    "up/flags.py": ("bootstrap", "将 UP 业务全局标志移到轻量模块以打断导入环", "标志仍属于 UP，不允许平台反向读取"),
    "up/utils/general/global_flag.py": ("compatibility", "保留旧标志导入路径，转发到同一份 up.flags 对象", "不能生成两套不同全局状态"),
    "up/models/model_helper.py": ("device", "显式初始化注册；使用统一设备迁移并同步模型 device/dtype 元数据", "旧 ModelHelper.cuda 作为选中后端的兼容入口，不等同原 PyTorch CUDA 专属语义"),
    "up/runner/base_runner.py": ("training", "设备/AMP/DDP 走平台；训练验证保护现有模型状态与 training 用途", "eval 恢复是实质行为改进，不只是机械替换 cuda"),
    "up/apis/inference.py": ("inference", "独立推理显式固定 inference 用途并迁移模型与输入到所选设备", "未增加 TensorRT 等部署引擎"),
    "up/utils/env/dist_helper.py": ("distributed", "设备绑定、集合通信、对象传输、linklink 训练包装移入平台", "另修 MPI 环境变量读取；私有集群 gpu-burn 被轻量 health_check 替代，检查强度不等价"),
    "up/utils/env/launch.py": ("distributed", "启动子进程时使用所选设备与平台通信入口", "保留原 dist/linklink 业务参数含义；各启动器并非全部真机验收"),
    "up/utils/env/gene_env.py": ("device", "随机种子与数据迁移使用所选设备，不硬编码 cuda", "不改变 Python/NumPy 的原种子配方"),
    "up/utils/model/optimizer_helper.py": ("optional_sdk", "融合优化器构造与 master 参数状态恢复改走平台操作", "学习率、参数分组和优化步骤仍由 UP 决定；真实 linklink 未验收"),
    "up/extensions/__init__.py": ("compatibility", "保留原扩展公开名字，延迟加载旧算子门面", "没有移除原公开算子名字"),
    "up/extensions/ext.py": ("compatibility", "旧扩展加载入口转发到平台，而非上层直接加载 _C", "不能因设备名都是 cuda 就加载 NVIDIA 二进制"),
    "up/tasks/det_3d/data/metrics/kitti_object_eval_python/rotate_iou.py": ("nvidia_ops", "Numba CUDA 旋转 IoU 实现移到 NVIDIA 后端，原路径保留转发", "保留显式 CPU/评估语义；不代表其他厂商已支持"),
    "up/tasks/det_3d/data/data_reader.py": ("optional_sdk", "将 Open3D 导入延迟到实际读取点云时，二维任务不强制安装", "点云读取算法未替换；使用该读取器仍须安装 Open3D"),
    "setup.py": ("packaging", "UP 本身不探测 nvcc/编译 GPU 扩展；依赖独立 up-platform 包", "最低 Python 元数据从 3.6 改为 3.8；完整业务环境兼容范围仍需单独认证"),
    "MANIFEST.in": ("packaging", "UP 源码分发规则不再绑定旧 csrc，避免打包独立平台源码副本", "NVIDIA 源码改由平台包分发"),
    "requirements.txt": ("environment", "保留上游依赖文件", "历史版本约束不是已验证现代环境锁"),
    "README.md": ("documentation", "改写为派生项目的安装、硬件选择、适配指南与验证入口", "当前首屏混有较长历史报告，需收敛而不是继续堆结果"),
    ".github/workflows/hardware-boundaries.yml": ("ci", "增加 CPU 契约、边界检查、契约漂移与公共打包检查", "目前硬编码根测试文件列表，未包含最近新增的部分回归；本地 277 通过不等于 CI 覆盖 277"),
    ".gitattributes": ("evidence", "控制文本换行，保留冻结原始证据的逐字节哈希", "不是模型精度修复"),
    ".gitignore": ("maintenance", "排除密钥、构建产物、权重与本地环境，仅显式保留公开审计日志", "忽略的本地密钥/环境不计入公开仓库文件统计"),
}


def purpose(path, status, diff=""):
    if path in PURPOSES:
        return PURPOSES[path]
    if area(path) == "frozen_evidence":
        return "evidence", "冻结该阶段配置、脚本、日志、结果或文件哈希，使结论可复查", "历史证据，不是产品运行依赖；归档时保留校验和与原提交"
    if "/csrc/" in path and path.startswith("packages/up-platform/"):
        return "nvidia_ops", "将上游 NVIDIA C++/CUDA 内核归到对应厂商构建/加载边界", "R100 是原字节迁移，不是删除算子，也不是新写了内核"
    if path.startswith("up/accelerator/"):
        return "compatibility", "保留第一轮适配 API、插件或旧实现路径，转发至独立平台", "薄兼容门面不是第二份硬件实现；后续需明确支持与弃用范围"
    if path.startswith("up/extensions/python/"):
        return "compatibility", "原导入与 .apply 入口保留，完整前后向路由到平台实现", "保持原签名、checkpoint 和 ONNX 语义；不能只转发前向"
    if path.startswith("packages/up-platform/"):
        if "/tests/" in path:
            return "tests", "独立平台契约、作用域或导出测试，不安装 UP 也可运行", "CPU 契约不等同 GPU 真机支持"
        if "/examples/" in path:
            return "examples", "可安装的外部厂商插件示例，演示共享复用与单项覆盖", "CPU 示例不冒充国产硬件模拟"
        if path.endswith(("pyproject.toml", "setup.py", "MANIFEST.in", "LICENSE", "README.md")):
            return "packaging", "独立平台包的构建、安装、许可或入口说明", "普通安装不自动编译 GPU 扩展，不自动替换厂商 PyTorch"
        if "/experimental/" in path:
            return "experimental", "显式 portable_v1 的固定顺序/RNG/FP64 数值实验实现", "默认关闭；不是新厂商基本接入义务，未完成跨厂商全预算验收"
        if path.endswith("metax/determinism.py"):
            return "metax_fix", "显式 GPU FP32 卷积权重梯度替换，以处理已复现的 C500 非确定性", "仅所测变体与短程通过；native 默认不变，无全局确定性承诺"
        if "/backends/" in path:
            return "vendor_backend", "按已选厂商提供设备、用途工厂、专用操作或可选 SDK 加载", "空的 training/inference 子类表示复用共享实现，不是该厂商全算子已通过"
        if "/training/" in path:
            return "training", "训练专属 AMP/scaler/DDP/旧梯度同步与训练内验证保护", "不决定学习率、损失或训练配方，不切换部署推理实现"
        if "/inference/" in path:
            return "inference", "独立推理接口与现有 PyTorch/ONNX 导出入口", "没有新增推理引擎、自动量化或权重打包"
        if "/common/" in path or "/shared/" in path:
            return "shared", "共用设备/RNG/通信/精度与同设备操作、层、稀疏表示", "portable 共享实现不表示静默 CPU 回退；当前数值实验仍耦合部分 common 代码"
        return "platform_core", "平台配置、后端注册、显式用途/操作路由、能力查询及完整契约", "一个操作的声明、依赖可用、执行、验证是不同状态"
    if path.startswith(("tools/", "benchmark/hardware/")):
        return "validation_tool", "准备/执行/诊断/审计或打包该模型与硬件回归，不参与 UP 训练运行时", "当前工具平铺且有专用阶段，后续应归入统一 benchmark/hardware 入口"
    if path.startswith("tests/") or path == "pytest.ini":
        return "tests", "保护平台契约、旧 API、协议、数据与证据工具的回归", "不是全体模型或所有 GPU 真机通过"
    if path.startswith("requirements/"):
        return "environment", "记录已观察到的指定厂商/版本依赖组合", "只对对应环境有证据；不是跨厂商通用 lockfile"
    if path.startswith("docs/"):
        return "documentation", "架构、接口、接入、环境、验证结论或历史报告入口", "当前/历史结论应分开；生成索引不应多份手工维护"
    if path.startswith("up/commands/") or path == "up/tasks/nas/metax/commands/search.py":
        return "entry_point", "UP 入口解析配置并显式激活 training/inference 后端", "业务 CLI 仍留在 UP；nas/metax 是原有 NAS 任务名，不是本次沐曦后端"
    if path.startswith("up/"):
        details = []
        if any(token in diff for token in ("get_accelerator", "up.accelerator")):
            details.append("将设备/AMP/显存/层/专用操作入口转发到平台")
        if any(token in diff for token in ("OptionalModule", "optional_symbol")):
            details.append("可选 SDK 改为选择对应功能时加载")
        if "preserve_evaluation_state" in diff or ("/runner/" in path and "-                self.model.train()" in diff):
            details.append("训练验证恢复调用前状态，不再无条件重置为 train")
        if any(token in diff for token in ("np.int", "np.float", "np.bool")):
            details.append("修复已移除的 NumPy 类型别名")
        if path.endswith("cls_transforms.py"):
            details.append("随机擦除临时张量继承调用设备，取消默认 cuda")
        if not details:
            raise ValueError("Manual purpose review required: " + path)
        return "up_integration", "；".join(details), "意图是保留算法/配方；不能据此宣称全体模型已数值验收"
    raise ValueError("Manual purpose review required: " + path)


def audit(base, target):
    base, target = [git("rev-parse", ref + "^{commit}").decode().strip() for ref in (base, target)]
    old, new = tree(base), tree(target)
    events = collections.defaultdict(list)
    commits = git("rev-list", "--reverse", base + ".." + target).decode().splitlines()
    for commit in commits:
        for row in changes(git("diff-tree", "--no-commit-id", "-r", "-M", "--name-status", "-z", commit)):
            for path in {row["path"], row["old_path"]} - {""}:
                events[path].append(commit)
    rows = []
    for row in changes(git("diff", "--find-renames", "--name-status", "-z", base, target)):
        path = row["path"] or row["old_path"]
        diff = git("diff", "--ignore-cr-at-eol", base, target, "--", path).decode(errors="replace") if path.startswith("up/") and row["status"] == "M" else ""
        category, reason, note = purpose(path, row["status"], diff)
        before, after = old.get(row["old_path"], {}), new.get(row["path"], {})
        normalized_equal = False
        if before and after:
            normalized_equal = git("cat-file", "blob", before["blob"]).replace(b"\r\n", b"\n") == git(
                "cat-file", "blob", after["blob"]).replace(b"\r\n", b"\n")
        row.update(area=area(path), category=category, purpose=reason, compatibility_note=note,
                   bytes_before=before.get("bytes", 0), bytes_after=after.get("bytes", 0),
                   normalized_content_equal=normalized_equal, first_change=events[path][0], last_change=events[path][-1])
        rows.append(row)
    footprint = collections.defaultdict(lambda: {"files": 0, "bytes": 0})
    for path, value in new.items():
        footprint[area(path)]["files"] += 1
        footprint[area(path)]["bytes"] += value["bytes"]
    model_paths = {path for path in old if path.startswith("up/models/") or (path.startswith("up/tasks/") and "/models/" in path)}
    summary = {"base": base, "target": target, "comparison": "committed Git trees, not dirty files or .git pack size",
               "files_before": len(old), "files_after": len(new), "bytes_before": sum(v["bytes"] for v in old.values()),
               "bytes_after": sum(v["bytes"] for v in new.values()), "changes": dict(collections.Counter(r["status"] for r in rows)),
               "areas": dict(footprint), "categories": dict(collections.Counter(r["category"] for r in rows)),
               "model_directory_modified": [r["path"] for r in rows if r["path"] in model_paths and not r["normalized_content_equal"]],
               "configs_unchanged": not any(r["area"] == "configs" for r in rows),
               "root_requirements_unchanged": old["requirements.txt"]["blob"] == new["requirements.txt"]["blob"],
               "license_unchanged": old["LICENSE"]["blob"] == new["LICENSE"]["blob"]}
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary, rows = audit(args.base, args.target)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    outputs = {"summary.json": json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
               "changes.csv": buffer.getvalue()}
    if args.check:
        for name, value in outputs.items():
            with (args.output / name).open("r", encoding="utf-8", newline="") as stream:
                saved = stream.read()
            if saved != value:
                raise SystemExit("Stale audit: " + name)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, value in outputs.items():
            with (args.output / name).open("w", encoding="utf-8", newline="") as stream:
                stream.write(value)
    print(json.dumps({key: summary[key] for key in ("base", "target", "changes", "files_before", "files_after")}, indent=2))


if __name__ == "__main__":
    main()
