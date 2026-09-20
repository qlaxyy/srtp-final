"""Export immutable saved trajectories; never regenerate calibration answers."""
import hashlib,json,tarfile,unicodedata
from pathlib import Path
from prepare_length_vector import HERE,ROOT

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x): Path(p).write_bytes((json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode())
def main():
    out=HERE/'motivation_replay_20260918_run1';out.mkdir(exist_ok=True)
    assert not (out/'release.json').exists()
    old=json.loads((HERE/'norm_preserving_math_20260918_run2/release.json').read_text())
    registry=HERE/'motivation_replication_20260918_cpu/math500_pair_registry.json'
    pairs=json.loads(registry.read_text());selected={r['dataset_index']:k for k,rows in pairs.items() for r in rows}
    archive=ROOT/'.codex_work/auto_code_v2_500_20260908/math500_artifacts.tar.gz'
    with tarfile.open(archive) as t: ev=json.load(t.extractfile('math500_eval.json'))
    rows=[]
    for group,key in [('U','baseline'),('R','rebalance_dynamic')]:
        for r in ev[key]['records']:
            i=r['dataset_index']
            if i in selected:
                rows.append(dict(group=group,dataset_index=i,pair_kind=selected[i],problem=r['problem'],
                    token_ids=r['token_ids'],problem_sha256=hashlib.sha256(''.join(unicodedata.normalize('NFKC',r['problem']).split()).encode()).hexdigest()))
    assert len(rows)==572 and sum(len(r['token_ids']) for r in rows)==1586422
    with tarfile.open(ROOT/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz') as t:
        train=[json.loads(line) for line in t.extractfile('generations.jsonl').read().splitlines()]
    # Fixed order of already frozen calibration rows; no confidence selection.
    indices=[r['train_index'] for r in train[:8]]
    engineering=[]
    for i in indices:
        r=next(x for x in train if x['train_index']==i)
        engineering.append(dict(train_index=i,problem=r['problem'],prompt_token_ids=r['prompt_token_ids'],
            token_ids=r['token_ids'][:512],original_logmax=r['logprobs'][:512]))
    save(out/'rows.json',rows);save(out/'engineering.json',engineering)
    source=['motivation_replay_sampler.py','run_motivation_replay.py']
    release=dict(assets=old['assets'],runtime_source_sha256=old['runtime_source_sha256'],
        source_sha256={n:sha(HERE/n) for n in source},rows_sha256=sha(out/'rows.json'),
        engineering_sha256=sha(out/'engineering.json'),registry_sha256=sha(registry),
        engineering_checks='8 saved training prefixes x U/R, each512 tokens; exact output, raw confidence equality before native update, request reset; direct U prefix alignment independently scored',
        full_gate='Requires successful engineering complete.json with identical release hash',
        full_seconds_ceiling=1800)
    save(out/'release.json',release)
    with tarfile.open(out/'package.tar.gz','w:gz') as t:
        for n in source:t.add(HERE/n,arcname=n)
        for n in ['rows.json','engineering.json','release.json']:t.add(out/n,arcname=n)
    print(json.dumps(dict(rows=len(rows),tokens=1586422,package_sha256=sha(out/'package.tar.gz'))))

if __name__=='__main__':main()
