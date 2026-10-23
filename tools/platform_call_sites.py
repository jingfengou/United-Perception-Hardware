"""Index actual UP-to-platform calls and check their names against the contract."""
import argparse
import ast
import json
from pathlib import Path

from platform_contracts import collect, ROOT
from platform_inventory import dotted


def collect_calls():
    contracts = collect()
    names = {item['name'] for item in contracts}
    layers = {item['name'].split('.')[1]: item['name'] for item in contracts
              if item.get('contract', {}).get('source', '').startswith('torch.nn.')
              and item['name'].split('.')[1][:1].isupper()}
    records, errors = [], []
    for path in sorted((ROOT/'up').rglob('*.py')):
        source = path.read_text(encoding='utf8')
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            parent = node.func.value
            target = None
            if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Attribute) and parent.func.attr == 'category':
                if isinstance(parent.func.value, ast.Call) and dotted(parent.func.value.func) == 'get_platform' and len(parent.args) == 1:
                    target = parent.args[0].value+'.'+node.func.attr
            elif isinstance(parent, ast.Attribute) and isinstance(parent.value, ast.Call) and dotted(parent.value.func) == 'get_platform':
                if parent.attr in ('device', 'execution'):
                    if parent.attr == 'device':
                        target = 'device.'+node.func.attr
                    else:
                        possible = [n for n in names if n in ('training.'+node.func.attr, 'inference.'+node.func.attr)]
                        target = '|'.join(sorted(possible))
            elif dotted(parent) == 'platform_layers':
                target = layers.get(node.func.attr)
            if target:
                if any(name not in names for name in target.split('|')):
                    errors.append((path.as_posix(), node.lineno, target))
                records.append({'path': path.relative_to(ROOT).as_posix(), 'line': node.lineno, 'interface': target})
    return sorted(records, key=lambda r: (r['path'], r['line'], r['interface'])), errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    records, errors = collect_calls()
    if errors:
        raise SystemExit(str(errors))
    path = ROOT/'docs/platform/call_sites.json'
    expected = json.dumps({'scope': 'Statically resolved current calls; see upstream_inventory.json for original candidates and limitations',
                           'calls': records}, indent=2)+'\n'
    if args.write:
        path.write_text(expected, encoding='utf8', newline='\n')
    elif not path.exists() or path.read_text(encoding='utf8') != expected:
        raise SystemExit('Call-site index drift; run --write and review')
    print(json.dumps({'calls': len(records), 'files': len({r['path'] for r in records})}))
