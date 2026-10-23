"""Compare upstream and routed legacy ONNX symbols, retaining original errors."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import onnx
import torch

from up_platform import configure_platform
from up_platform.ops._native.loader import load_native
from compare_platform_native import cases, original_module


def graph_signature(data):
    model = onnx.load_model_from_string(data)
    # Value names contain tracing scope names; topology uses stable value indices.
    names = {value.name: 'input:'+str(i) for i, value in enumerate(model.graph.input)}
    result = []
    for i, node in enumerate(model.graph.node):
        result.append({'domain': node.domain, 'op': node.op_type,
                       'inputs': [names.get(value, value) for value in node.input],
                       'attributes': {attr.name: attr.SerializeToString().hex() for attr in node.attribute}})
        for j, value in enumerate(node.output):
            names[value] = f'node:{i}:{j}'
    return {'nodes': result, 'outputs': [names[v.name] for v in model.graph.output],
            'initializers': [hashlib.sha256(v.SerializeToString()).hexdigest() for v in model.graph.initializer]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vendor', required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    p = configure_platform({'backend': args.vendor}, purpose='inference')
    torch.manual_seed(131)
    flags = SimpleNamespace(aligned=False, offset=1)
    extension = load_native(args.vendor)
    records = []
    for index, (category, method, module, entry, inputs, gradients, kwargs) in enumerate(cases(p.device.device())):
        if method not in ('roi_align', 'psroi_align', 'psroi_pool', 'deform_conv'):
            continue
        reference = original_module(args.baseline, module, extension, flags, args.vendor == 'metax')[entry].apply
        flags.aligned = kwargs.get('aligned', False)
        graphs, errors = {}, {}
        tensor_inputs = tuple(value for value in inputs if isinstance(value, torch.Tensor))
        for route in ('original', 'routed'):
            class Wrapper(torch.nn.Module):
                def forward(self, *values):
                    iterator = iter(values)
                    values = [next(iterator) if isinstance(value, torch.Tensor) else value for value in inputs]
                    if route == 'original':
                        return reference(*values)
                    return getattr(p.category(category), method)(*values, **kwargs)
            buffer = io.BytesIO()
            try:
                p.execution.export_onnx(Wrapper(), tensor_inputs, buffer, opset_version=11,
                                        do_constant_folding=False)
                data = buffer.getvalue()
                (args.output/f'{index}-{method}-{route}.onnx').write_bytes(data)
                graphs[route] = graph_signature(data)
            except Exception as error:
                errors[route] = {'type': type(error).__name__, 'message': str(error)}
        records.append({'case': f'{index}-{method}', 'graphs': graphs, 'errors': errors,
                        'passed': len(graphs) == 2 and graphs['original'] == graphs['routed']})
    (args.output/'summary.json').write_text(json.dumps({'records': records,
        'scope': 'Legacy graph topology and symbolic attributes; not execution by a deployment engine',
        'reference_adaptations': ['PSRoIPool scratch .contiguous()'] if args.vendor == 'metax' else []}, indent=2))
    print(json.dumps([{'case': r['case'], 'passed': r['passed'], 'errors': r['errors']} for r in records]))
    raise SystemExit(0 if all(r['passed'] for r in records) else 1)


if __name__ == '__main__':
    main()
