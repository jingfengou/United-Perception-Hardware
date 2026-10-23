"""Index all trial artifacts and package small JSON reports; never rewrite trials."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import tarfile

from run_platform_validation import source_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--results-subdir', help='Optional single result directory for an additional validation stage, not a full-history index')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    scope = (args.root/'results'/args.results_subdir).resolve() if args.results_subdir else (args.root/'results').resolve()
    assert scope.is_relative_to((args.root/'results').resolve()) and scope.is_dir(), 'Invalid result scope'
    paths = set(scope.rglob('*'))
    for pattern in ('build-*.json', 'build-*.log', 'stage-*.tar.gz', 'baseline.tar.gz', 'dependency-*.json'):
        paths.update(args.root.glob(pattern))
    paths.update((args.root/'up_platform/backends').glob('*.so'))
    items = []
    for path in sorted(paths):
        if not path.is_file():
            continue
        with path.open('rb') as file:
            digest = hashlib.file_digest(file, 'sha256').hexdigest()
        items.append({'path': path.relative_to(args.root).as_posix(), 'bytes': path.stat().st_size, 'sha256': digest})
    packages = {dist.metadata['Name']: dist.version for dist in metadata.distributions()}
    import torch
    devices = [{'name': torch.cuda.get_device_properties(i).name,
                'total_memory': torch.cuda.get_device_properties(i).total_memory}
               for i in range(torch.cuda.device_count())]
    index = {'python': sys.version, 'packages': packages, 'devices': devices,
             'evidence_scope': str(scope.relative_to(args.root.resolve())),
             'source': source_manifest(args.root), 'artifacts': items,
             'large_artifacts': 'Retained in the task evidence directory; this archive contains small reports only.'}
    index_path = args.output.with_suffix('.index.json')
    if index_path.exists():
        raise FileExistsError(index_path)
    index_path.write_text(json.dumps(index, indent=2))
    with tarfile.open(args.output, 'w:gz') as archive:
        archive.add(index_path, arcname='index.json')
        for item in items:
            path = args.root/item['path']
            if path.suffix == '.json' or path.name.endswith('.xml'):
                archive.add(path, arcname=item['path'])
    print(json.dumps({'artifacts': len(items), 'bytes': sum(i['bytes'] for i in items), 'archive': str(args.output)}))


if __name__ == '__main__':
    main()
