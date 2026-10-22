"""Compare legacy ONNX symbols and attributes in separate upstream/routed runs."""
import argparse
import io
import json
import sys
import traceback
from pathlib import Path

import onnx
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from validate_native_ops import baseline_modules


def run(args):
    from up.accelerator import configure_accelerator, get_accelerator
    configure_accelerator(backend="nvidia", reset=True)
    get_accelerator().set_device(args.device)
    modules = baseline_modules(Path(args.baseline)) if args.baseline else {}
    if not modules:
        from up.extensions.python import roi_align, psroi_align, psroi_pool, deformable_conv
        modules = dict(roi_align=roi_align, psroi_align=psroi_align,
                       psroi_pool=psroi_pool, deformable_conv=deformable_conv)

    class ExportOp(torch.nn.Module):
        def __init__(self, function, extra):
            super().__init__()
            self.function, self.extra = function, extra

        def forward(self, *values):
            return self.function(*values, *self.extra)

    report = {"mode": "upstream" if args.baseline else "routed", "cases": {}, "errors": {},
              "torch": torch.__version__, "onnx": onnx.__version__,
              "scope": "legacy exporter symbols/attributes; no ONNX runtime execution"}
    for device in ("cuda", "cpu"):
        rois = torch.tensor([[0., 1., 1., 5., 5.]], device=device)
        feature = torch.randn(1, 8, 8, 8, device=device)
        cases = {
            "roi_align": ("RoIAlignFunction", (feature, rois), (2, 2, 1., 2, "AVG")),
            "psroi_align": ("PSRoIAlignFunction", (feature, rois), (2, 1., 2, 2)),
            "psroi_pool": ("PSRoIPoolFunction", (feature, rois), (2, 1., 2)),
            "deformable_conv": (
                "DeformableConvFunction",
                (feature, torch.zeros(1, 18, 8, 8, device=device),
                 torch.randn(2, 8, 3, 3, device=device)),
                (2, (3, 3), (1, 1), (1, 1), (1, 1), 1, 1)),
        }
        for name, (function, values, extra) in cases.items():
            key = device + "." + name
            try:
                model = ExportOp(getattr(modules[name], function).apply, extra).eval()
                output = io.BytesIO()
                torch.onnx.export(model, values, output, opset_version=11, dynamo=False)
                graph = onnx.load_model_from_string(output.getvalue())
                nodes = [dict(domain=node.domain, op_type=node.op_type,
                              attributes={attribute.name: str(onnx.helper.get_attribute_value(attribute))
                                          for attribute in node.attribute})
                         for node in graph.graph.node]
                assert nodes, "Export produced no nodes"
                report["cases"][key] = nodes
                print(key, "PASS", flush=True)
            except Exception:
                report["errors"][key] = traceback.format_exc()
                print(key, report["errors"][key], flush=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(bool(report["errors"]))


def compare(args):
    baseline, routed = [json.loads(Path(path).read_text()) for path in args.compare]
    regressions = {
        name: dict(expected=value, actual=routed["cases"].get(name), error=routed["errors"].get(name))
        for name, value in baseline["cases"].items() if routed["cases"].get(name) != value}
    regressions.update({name: error for name, error in routed["errors"].items()
                        if name not in baseline["errors"]})
    report = dict(passed=not regressions,
                  compared_cases=len(baseline["cases"]), regressions=regressions,
                  upstream_gaps=baseline["errors"], routed_errors=routed["errors"],
                  scope=routed["scope"], torch=routed["torch"], onnx=routed["onnx"])
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(not report["passed"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline")
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compare", nargs=2)
    args = parser.parse_args()
    compare(args) if args.compare else run(args)
