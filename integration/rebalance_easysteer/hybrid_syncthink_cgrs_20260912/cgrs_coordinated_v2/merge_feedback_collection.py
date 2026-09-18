"""Explicitly assemble training-only outputs from failed batch and bounded recovery."""
import argparse, json
from pathlib import Path
from grade_self_feedback import read, save, sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--first',type=Path,required=True)
    p.add_argument('--recovery',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--recovery-artifact',type=Path,required=True)
    a=p.parse_args()
    partial=a.first/'L27_L27/partial.jsonl'
    assert sha(partial)=='b755ff41fdff8d56f87644bd11d094e80926f52752125abcb640f82106a4d5f9'
    gate=read(a.recovery/'complete.json')
    assert gate['phase']=='full' and gate['passed'] and gate['release_sha256']==sha(a.recovery_artifact/'release.json')
    # Runner writes JSON again on Linux; verify content as well as original input hash.
    assert read(a.recovery/'release.json')==read(a.recovery_artifact/'release.json')
    assert read(a.first/'release.json')==read(a.artifact/'release.json')
    plan=read(a.first/'plan.json'); recovery_plan=read(a.recovery/'plan.json')
    assert plan['rows']==recovery_plan['rows']
    first=[json.loads(s) for s in partial.read_text().splitlines()]
    second=read(a.recovery/'L27_L27/result.json')
    assert second['status']=='complete'
    assert [r['dataset_index'] for r in second['records']]==recovery_plan['recovery_indices']
    assert len(first)==404 and len(second['records'])==96
    rows=sorted(first+second['records'],key=lambda r:r['dataset_index'])
    assert [r['dataset_index'] for r in rows]==list(range(500))
    for r,g in zip(rows,plan['rows']): assert r['problem_sha256']==g['problem_sha256']
    a.out.mkdir(exist_ok=False)
    (a.out/'L27_L27').mkdir()
    for n in ['plan.json','release.json']:
        assert read(a.first/n)==read(a.artifact/n)
        (a.out/n).write_bytes((a.artifact/n).read_bytes())
    sources=[partial,a.first/'failure.json',a.artifact/'release.json',a.recovery/'L27_L27/result.json',a.recovery_artifact/'release.json',a.recovery/'complete.json']
    save(a.out/'merge_provenance.json',dict(sources=[dict(path=str(s),sha256=sha(s)) for s in sources],
        batch_counts=[404,96],max_num_seqs=[256,32],interpretation='One training collection across two schedules, not a single-run evaluation',
        failed_process_wall_seconds=read(a.first/'failure.json')['wall_seconds'],recovery_generation_seconds=second['generation_seconds'],
        merge_script_sha256=sha(Path(__file__))))
    save(a.out/'L27_L27/result.json',dict(status='complete',records=rows,generation_seconds=None,extra_model_forward_count=0,
        timing_note='Partial attempt did not finish its generation timer. See merge_provenance: failed process wall time and recovery generation are separate; do not call their sum pure generation.'))
    save(a.out/'complete.json',dict(phase='collection_merged',passed=True,arms=['L27_L27'],release_sha256=sha(a.out/'release.json')))


if __name__=='__main__':main()
