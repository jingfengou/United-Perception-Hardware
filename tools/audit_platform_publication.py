"""Verify published evidence and runtime source against a hardware trial index.

The only permitted post-trial runtime edit is removal of an unused platform
import. Its before bytes must match the real trial index, not a guessed source.
Test/tools changes are separate: full trial manifests and archives retain them.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def without_unused_platform_imports(raw):
    source = raw.decode('utf8').replace('\r\n', '\n')
    tree = ast.parse(source)
    loaded = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    removable = set()
    for node in tree.body:
        if (isinstance(node, ast.ImportFrom) and node.module == 'up_platform' and len(node.names) == 1
                and (node.names[0].asname or node.names[0].name) not in loaded
                and node.names[0].name in ('get_platform', 'layers')):
            removable.update(range(node.lineno, node.end_lineno+1))
    return ''.join(line for i, line in enumerate(source.splitlines(keepends=True), 1) if i not in removable).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True, help='Published vendor directory containing index.json')
    parser.add_argument('--reference-archive', type=Path, help='Original source archive, required for an unused-import-only difference')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--startup-evidence', type=Path, help='Successful follow-up for the two import-time-default changes')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    index = json.loads((args.evidence/'index.json').read_text())
    evidence = []
    for item in index['artifacts']:
        path = args.evidence/item['path']
        if path.is_file():
            assert digest(path.read_bytes()) == item['sha256'], str(path)
            evidence.append(item['path'])
    files = index['source']['files']
    runtime = {name: value for name, value in files.items() if name.startswith(('up/', 'up_platform/', 'configs/'))}
    current = {p.relative_to(ROOT).as_posix() for folder in ('up', 'up_platform', 'configs')
               for p in (ROOT/folder).rglob('*')
               if p.suffix in ('.py', '.cu', '.cpp', '.h', '.yaml', '.json') and '__pycache__' not in p.parts}
    assert current == runtime.keys(), 'Runtime file set changed after the trial'
    followup = None
    if args.startup_evidence:
        followup = json.loads((args.startup_evidence/'results/startup-stage15/summary.json').read_text())
        assert followup['passed'], 'Startup/default compatibility did not pass'
    followup_paths = {'up_platform/layers.py', 'up/tasks/seg/models/encoder/segformer_encoder.py'}
    identical, cleanup, additional = [], [], []
    archive = tarfile.open(args.reference_archive) if args.reference_archive else None
    try:
        for name, expected in sorted(runtime.items()):
            raw = (ROOT/name).read_bytes().replace(b'\r\n', b'\n')
            if digest(raw) == expected:
                identical.append(name)
                continue
            if followup and name in followup_paths:
                assert followup['source']['files'][name] == digest(raw), 'Follow-up source mismatch: '+name
                additional.append({'path': name, 'before': expected, 'after': digest(raw),
                                   'validation': 'startup-stage15: default identity, RNG, forward/gradients and CLI'})
                continue
            assert archive and name.endswith('.py'), 'Changed runtime requires new validation: '+name
            old = archive.extractfile(name).read().replace(b'\r\n', b'\n')
            assert digest(old) == expected, 'Archive does not match the trial: '+name
            assert without_unused_platform_imports(old) == raw, 'Change exceeds unused imports: '+name
            cleanup.append({'path': name, 'before': expected, 'after': digest(raw)})
    finally:
        if archive:
            archive.close()
    report = {'trial_source_sha256': index['source']['sha256'], 'normalization': 'CRLF to LF only',
              'scope': 'up/, up_platform/, configs/; tools/tests retain their separate trial manifests',
              'published_evidence_files_verified': len(evidence), 'runtime_files': len(runtime),
              'identical_files': len(identical), 'unused_import_only': cleanup,
              'additional_validated_changes': additional,
              'followup_source_sha256': followup['source']['sha256'] if followup else None,
              'reference_archive': args.reference_archive.name if args.reference_archive else None,
              'reference_archive_sha256': digest(args.reference_archive.read_bytes()) if args.reference_archive else None,
              'other_runtime_changes': 0, 'passed': True}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'unused_import_only'}))


if __name__ == '__main__':
    main()
