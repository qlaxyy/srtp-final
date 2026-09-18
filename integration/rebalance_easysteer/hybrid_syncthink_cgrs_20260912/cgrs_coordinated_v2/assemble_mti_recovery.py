"""Assemble fixed 500-identity coverage, explicitly NOT a continuous run.

The failed process did not serialize aggregate auxiliary/timing counters.
Those fields remain null; never impute them from the recovered subset.
"""
import argparse,hashlib,json
from pathlib import Path


def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,d):
    with p.open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True)
    p.add_argument('--recovery',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plan=read(a.recovery/'plan.json');release=read(a.recovery/'release.json')
    complete=read(a.recovery/'complete.json');assert complete['passed'] and complete['phase']=='full'
    assert 'preemption' in read(a.original/'failure.json')['error']
    source=a.original/'MTI_L27/partial.jsonl'
    assert sha(source)==plan['recovery_provenance']['partial_sha256']
    original=[json.loads(x) for x in source.read_text(encoding='utf8').splitlines()]
    recovered=read(a.recovery/'MTI_L27/result.json')
    assert recovered['status']=='complete'
    assert len(original)==458 and len(recovered['records'])==42
    assert sorted(r['dataset_index'] for r in original)==plan['recovery_provenance']['completed_indices']
    assert sorted(r['dataset_index'] for r in recovered['records'])==plan['recovery_indices']
    merged=sorted(original+recovered['records'],key=lambda r:r['dataset_index'])
    assert [r['dataset_index'] for r in merged]==list(range(500))
    for q,r in zip(plan['rows'],merged):
        assert q['problem_sha256']==r['problem_sha256']
        assert r['tokens']==len(r['token_ids'])<=16000
    a.output.mkdir(parents=True,exist_ok=False);folder=a.output/'MTI_L27';folder.mkdir()
    save(a.output/'plan.json',plan);save(a.output/'release.json',release)
    provenance=dict(coverage_complete=True,continuous_execution=False,uniform_schedule=False,
        original_path=str(a.original),recovery_path=str(a.recovery),original_partial_sha256=sha(source),
        recovery_result_sha256=sha(a.recovery/'MTI_L27/result.json'),
        original_process_wall_seconds=read(a.original/'failure.json')['wall_seconds'],
        recovery_process_wall_seconds=complete['wall_seconds'],recovery_generation_seconds=recovered['generation_seconds'],
        source_by_dataset_index=['recovery32' if i in plan['recovery_indices'] else 'original256' for i in range(500)],
        aggregate_pure_generation_seconds=None,aggregate_auxiliary_forward_count=None,
        reason_missing='Original process terminated on preemption before aggregate telemetry serialization',
        limitation='Mixed-schedule descriptive full coverage; not uniform-batch confirmation or speed comparison')
    save(a.output/'recovery_provenance.json',provenance)
    save(folder/'result.json',dict(status='complete',records=merged,generation_seconds=None,setup_seconds=None,
        control_gpu_seconds=None,extra_model_forward_count=None,probe_count=None,
        mti=dict(aggregate_unavailable=True,recovery_only=recovered['mti']),
        recovery_provenance=provenance,
        timing_note='Not a single run. Total pure generation and auxiliary count unavailable; recovery portion recorded separately.'))
    save(a.output/'complete.json',dict(phase='full',passed=True,coverage_complete=True,
        assembled_recovery=True,continuous_execution=False,arms=['MTI_L27'],
        release_sha256=sha(a.output/'release.json'),
        wall_seconds=provenance['original_process_wall_seconds']+provenance['recovery_process_wall_seconds']))
    print(json.dumps(provenance,ensure_ascii=False))


if __name__=='__main__':main()
