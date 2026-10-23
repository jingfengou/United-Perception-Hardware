"""Check owned boundaries; explicitly ledger the still-blocked linklink branch.

This is an AST guard, not a Python tensor-type inference engine. It does not claim
all upstream candidate calls are accelerator operations or all models are tested.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from platform_inventory import dotted
from platform_contracts import collect


class EagerPlatformCalls(ast.NodeVisitor):
    """Definition-time defaults must not choose a vendor before CLI parsing."""
    def __init__(self):
        self.depth, self.calls = 0, []

    def visit_FunctionDef(self, node):
        for value in [*node.args.defaults, *filter(None, node.args.kw_defaults), *node.decorator_list]:
            self.visit(value)
        self.depth += 1
        for value in node.body:
            self.visit(value)
        self.depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        for value in [*node.args.defaults, *filter(None, node.args.kw_defaults)]:
            self.visit(value)
        self.depth += 1
        self.visit(node.body)
        self.depth -= 1

    def visit_Call(self, node):
        name = ast.unparse(node.func)
        if self.depth == 0 and (name == 'get_platform' or name.startswith('platform_layers.')):
            if name != 'platform_layers.defer_default':
                self.calls.append((node.lineno, 'import-time platform call: '+name))
        self.generic_visit(node)


def inspect_boundaries():
    routed = {x.get('contract', {}).get('source') for x in collect()}
    forbidden_imports = ('spconv', 'cumm', 'mqbench', 'msbench', 'numba.cuda', 'torch.cuda', 'torch.distributed')
    violations, sdk_gaps = [], []
    for path in (ROOT/'up').rglob('*.py'):
        source = path.read_text(encoding='utf8')
        tree = ast.parse(source)
        eager = EagerPlatformCalls()
        eager.visit(tree)
        violations.extend((str(path), line, name) for line, name in eager.calls)
        aliases = {}
        for node in ast.walk(tree):
            entries = []
            if isinstance(node, ast.Import):
                entries = [(a.asname or a.name.split('.')[0], a.name) for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                entries = [(a.asname or a.name, node.module+'.'+a.name) for a in node.names]
            for alias, name in entries:
                aliases[alias] = name
                if name.startswith('spring.linklink') or name == 'spring.linklink':
                    sdk_gaps.append({'path': path.relative_to(ROOT).as_posix(), 'import': name,
                                     'statement': ast.unparse(node), 'reason': 'Matching SDK unavailable; not validated or replaced'})
                elif any(name == prefix or name.startswith(prefix+'.') for prefix in forbidden_imports):
                    violations.append((str(path), node.lineno, name))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = dotted(node.func)
            if name:
                head, *rest = name.split('.')
                canonical = '.'.join([aliases.get(head, head), *rest])
                if canonical in routed or canonical.startswith(('torch.cuda.', 'torch.distributed.', 'torch.backends.')) or canonical == 'torch.onnx.export':
                    violations.append((str(path), node.lineno, canonical))
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'cuda':
                violations.append((str(path), node.lineno, '.cuda'))
            if (isinstance(node.func, ast.Attribute) and node.func.attr == 'matmul' and node.args
                    and isinstance(node.args[0], ast.Name) and node.args[0].id in ('np', 'numpy')):
                violations.append((str(path), node.lineno, 'NumPy call incorrectly routed as Tensor'))
    return violations, sorted(sdk_gaps, key=lambda item: (item['path'], item['import']))


if __name__ == '__main__':
    violations, gaps = inspect_boundaries()
    path = ROOT/'docs/platform/linklink_call_sites.json'
    expected = json.dumps(gaps, indent=2)+'\n'
    if '--write-gaps' in sys.argv:
        path.write_text(expected, encoding='utf8', newline='\n')
    elif not path.exists() or path.read_text(encoding='utf8') != expected:
        violations.append(('linklink ledger', 0, 'Changed SDK imports must be reviewed, not silently whitelisted'))
    for row in violations:
        print(':'.join(map(str, row)))
    print(json.dumps({'violations': len(violations), 'blocked_sdk_imports': len(gaps)}))
    raise SystemExit(bool(violations))
