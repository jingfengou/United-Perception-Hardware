"""Audit the fixed portable-profile cohort and publish compact evidence views."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile
from pathlib import PurePosixPath
import tempfile
from types import SimpleNamespace


def sibling(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')


def read_source(root, backend):
    archive = root / (backend + '-source.tar.gz')
    index = json.loads(archive.with_name(archive.name + '.index.json').read_text())
    sha = lambda value: hashlib.sha256(value).hexdigest()
    assert sha(archive.read_bytes()) == index['archive_sha256']
    files, normalized, upper = {}, {}, {}
    with tarfile.open(archive) as stream:
        for member in stream:
            path = PurePosixPath(member.name)
            assert member.isfile() and path.parts[0] in ('up', 'configs', 'tools', 'tests')
            assert '..' not in path.parts and not path.is_absolute() and '\\' not in member.name
            assert member.name not in files and member.size < 16 * 1024 * 1024
            content = stream.extractfile(member).read()
            files[member.name] = sha(content)
            if member.name.startswith('up/') and member.name.endswith('.py'):
                normalized[member.name] = sha(content.replace(b'\r\n', b'\n'))
            if (path.parts[0] in ('up', 'configs') and not member.name.startswith('up/accelerator/')
                    and path.suffix in ('.py', '.yaml', '.json')):
                upper[member.name] = sha(content.replace(b'\r\n', b'\n'))
    assert files == index['files'] and normalized == index['normalized_up_files']
    source = sha(json.dumps(normalized, sort_keys=True).encode())
    assert source == index['normalized_up_manifest_sha256']
    return source, {p: value for p, value in files.items() if p.startswith('up/') and p.endswith('.py')}, index, upper


def write_statistics(root, suite):
    if (root / 'summary.json').exists():
        raise ValueError('Preserve the existing summary; do not replace statistics')
    with tempfile.TemporaryDirectory(prefix='up-cohort-reports-') as temporary:
        directory = Path(temporary)
        for backend in ('nvidia', 'metax'):
            archive = root / (backend + '-evidence.tar.gz')
            index = json.loads(archive.with_name(archive.name + '.index.json').read_text())
            assert hashlib.sha256(archive.read_bytes()).hexdigest() == index['archive_sha256']
            with tarfile.open(archive) as stream:
                for member in stream:
                    path = PurePosixPath(member.name)
                    if path.name != 'report.json' or len(path.parts) != 4 or path.parts[:2] != (backend, 'results'):
                        continue
                    assert member.isfile() and '..' not in path.parts and '\\' not in member.name
                    payload = stream.extractfile(member).read()
                    assert hashlib.sha256(payload).hexdigest() == index['files'][path.relative_to(backend).as_posix()]
                    target = directory / backend / path.parts[2] / 'report.json'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(payload)
        suite.summarize(SimpleNamespace(plan=root/'plan.json', output=root/'summary.json',
                                        nvidia=directory/'nvidia', metax=directory/'metax'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--repository',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--write-statistics',action='store_true')
    args=parser.parse_args()
    reader=sibling('summarize_diagnosis_evidence')
    suite=sibling('run_cross_vendor_accuracy')
    if args.write_statistics:
        write_statistics(args.root, suite)
    plan=suite.load_plan(args.root/'plan.json')
    summary=json.loads((args.root/'summary.json').read_text())
    assert summary['plan_sha256']==suite.sha256(args.root/'plan.json')
    current_source=reader.source_manifest(args.repository)
    sources={}
    frozen_upper={}
    cohorts,environments,archives={},{},{}
    for backend in plan['backends']:
        source, source_files, snapshot, frozen_upper[backend] = read_source(args.root, backend)
        sources[backend] = {'manifest': source, 'snapshot_sha256': snapshot['archive_sha256'],
                            'matches_current_working_up': source == current_source}
        index=json.loads((args.root/(backend+'-evidence.tar.gz.index.json')).read_text())
        assert index['plan_sha256']==suite.sha256(args.root/'plan.json')
        assert index['data_sha256']==index['files']['data/data.json']
        cohorts[backend],environments[backend],archives[backend]=reader.read_cohort(args.root,backend)
        for name,report in cohorts[backend].items():
            suite.verify_run(report,plan,report['model'],report['seed'],backend,
                             index['data_sha256'],suite.sha256(args.repository/'tools/validate_training_accuracy.py'))
            assert report['normalized_source_manifest_sha256']==source,'Executed application differs: '+name
            assert report['source_files']==source_files,'Source bytes differ from training report: '+name
            assert report['driver_sha256']==snapshot['files']['tools/validate_training_accuracy.py']
            if report['model']=='detr':
                assert report['adapter_sha256']==snapshot['files']['tools/accuracy_detr_adapter.py']
            else:
                assert report['adapter_sha256'] is None
            assert report['numerics']['profile']==plan['numerical_profile']
            assert report['numerics']['operations'].get('conv2d',0)>0
            target=args.root/backend/'results'/name/'run-summary.json'
            view=json.loads(target.read_text())
            view['numerics']=report['numerics']
            save(target,view)
    rows=[]
    for model,specification in plan['models'].items():
        for pair in summary['models'][model]['pairs']:
            name=model+'-seed'+str(pair['seed'])
            a,b=[cohorts[backend][name] for backend in plan['backends']]
            suite.compare_pair(a,b,plan)
            reports=[a,b]
            for backend,report,expected in zip(plan['backends'],reports,pair['report_sha256'],strict=True):
                index=json.loads((args.root/(backend+'-evidence.tar.gz.index.json')).read_text())
                assert expected==index['files']['results/'+name+'/report.json']
            rng_gaps=[i for i,(left,right) in enumerate(zip(a['batch_audit'],b['batch_audit'],strict=True))
                      if left['rng_sha256']!=right['rng_sha256']]
            values=[r['final_metrics'][specification['primary_metric']]*specification['metric_to_percentage_points'] for r in reports]
            assert values==[pair['nvidia'],pair['metax']]
            rows.append({'model':model,'seed':pair['seed'],'nvidia':values[0],'metax':values[1],
                         'delta_pp':values[1]-values[0],'training_steps_per_vendor':a['training_steps'],
                         'all_training_inputs_equal':a['all_training_batches_sha256']==b['all_training_batches_sha256'],
                         'all_pre_forward_rng_hashes_equal':not rng_gaps,'rng_mismatch_steps':rng_gaps,
                         'losses_equal':a['losses']==b['losses'],
                         'final_state_bitwise_equal':a['final_state_sha256']==b['final_state_sha256']})
    changed=[]
    reference_upper={}
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only','95a8e145b5d1fb96d7ec3da96158f289cad5680c','up','configs'],cwd=args.repository).decode().splitlines()
    for relative in paths:
        if relative.startswith('up/accelerator/') or not relative.endswith(('.py','.yaml','.json')):
            continue
        old=subprocess.check_output(['git','show','95a8e145b5d1fb96d7ec3da96158f289cad5680c:'+relative],cwd=args.repository)
        reference_upper[relative]=hashlib.sha256(old.replace(b'\r\n',b'\n')).hexdigest()
        new=(args.repository/relative).read_bytes()
        if old.replace(b'\r\n',b'\n')!=new.replace(b'\r\n',b'\n'):
            changed.append(relative)
    assert not changed,'Upper application changed: '+repr(changed)
    assert all(values==reference_upper for values in frozen_upper.values()),'Executed upper application/configs differ from reference'
    completed=sum(len(values) for values in cohorts.values())
    steps=sum(r['training_steps'] for c in cohorts.values() for r in c.values())
    save(args.root/'coverage.json',{'status':'passed','scope':'Integrity and declared-budget checks, not accuracy equivalence',
         'complete_runs':completed,'optimizer_steps':steps,'pairs':rows,'archives':archives,
         'application_source_snapshots':sources,'current_working_up_manifest':current_source,
         'upper_application_and_configs_changed':changed,
         'executed_upper_application_and_configs_match_reference':True,
         'upper_reference_commit':'95a8e145b5d1fb96d7ec3da96158f289cad5680c'})
    print(json.dumps({'complete_runs':completed,'steps':steps,'fair_pairs':len(rows),
                      'all_rng_pairs_equal':all(r['all_pre_forward_rng_hashes_equal'] for r in rows),
                      'bitwise_final_state_pairs':sum(r['final_state_bitwise_equal'] for r in rows)}))


if __name__=='__main__':
    main()
