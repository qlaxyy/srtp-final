import csv,hashlib,json,sys
from pathlib import Path
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();root=args.root
here=Path(__file__).resolve().parents[2]
for name,digest in json.loads((root/'artifact_manifest.json').read_text()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
sys.path.insert(0,str(here))
from full_grade import compare
read=lambda p:json.loads(p.read_text(encoding='utf8'))
screen=root/'results/cgrs_narrow8_7b_screen100_20260914'
analysis=read(screen/'analysis.json');plan=read(screen/'resolved_plan.json');records={};compact={}
for name in plan['arms']:
    result=read(screen/name/'result.json');rows=result['records'];g=analysis['groups'][name]
    assert len(rows)==g['n']==100
    for i,(r,label) in enumerate(zip(rows,g['labels'])):
        assert r['dataset_index']==label['dataset_index']==i
        assert r['problem_sha256']==label['problem_sha256']==plan['rows'][i]['problem_sha256']
        assert hashlib.sha256(r['text'].encode()).hexdigest()==label['text_sha256']
        assert len(r['token_ids'])==r['tokens']<=16000
        assert r['thinking_tokens']==(r['token_ids'].index(151649) if 151649 in r['token_ids'] else r['tokens'])
    assert sum(x['correct'] for x in g['labels'])==g['correct']
    assert sum(x['tokens'] for x in rows)/100==g['mean_total_tokens']
    assert sum(x['thinking_tokens'] for x in rows)/100==g['mean_thinking_tokens']
    records[name]=[dict(r,correct=x['correct']) for r,x in zip(rows,g['labels'])]
    compact[name]={k:g[k] for k in ('n','correct','mean_total_tokens','mean_thinking_tokens','capped',
                                   'generation_seconds','gpu_mean_percent','output_tokens_per_second','replayed_input_tokens')}
for a,b in [('R','RC14'),('R','RC8'),('RC14','RC8')]:
    assert compare(records[a],records[b],analysis['groups'][b]['labels'])==analysis['comparisons'][b+'_vs_'+a]
speed={};speed_records={}
for profile in ('current32','candidate48'):
    folder=root/('results/cgrs_narrow8_7b_speed64_20260914_'+profile)
    d=read(folder/'R/result.json');rs=d['records'];speed_records[profile]=rs
    assert len(rs)==64 and all(r['tokens']<=1024 for r in rs)
    util=[float(r[1]) for r in csv.reader((folder/'R/gpu.csv').read_text().splitlines()) if len(r)==4]
    speed[profile]=dict(n=64,tokens=sum(r['tokens'] for r in rs),seconds=d['generation_seconds'],
        tokens_per_second=sum(r['tokens'] for r in rs)/d['generation_seconds'],gpu_mean=sum(util)/len(util),
        preemptions=len(d['scheduler_preemptions']),replayed_input_tokens=sum(x.get('replay_prefill_tokens',0) for x in d['replay_events']))
assert [r['problem_sha256'] for r in speed_records['current32']]==[r['problem_sha256'] for r in speed_records['candidate48']]
speed['comparison']=dict(exact_token_sequences=sum(a['token_ids']==b['token_ids'] for a,b in zip(speed_records['current32'],speed_records['candidate48'])),
    throughput_change_percent=100*(speed['candidate48']['tokens_per_second']/speed['current32']['tokens_per_second']-1),
    generation_time_change_percent=100*(speed['candidate48']['seconds']/speed['current32']['seconds']-1),
    limitation='Single short run per config with cap1024, fixed order. Not a sustained long-context or accuracy test; no automatic promotion.')
e=root/'results/cgrs_narrow8_7b_engineering_20260914'
engineering={n:read(e/n/'result.json') for n in ('R','Roff8','RC14default','RC14explicit','RC8shadow','RC8')}
key=lambda d:[(r['token_ids'],r['R_history_sha256']) for r in d['records']]
assert key(engineering['R'])==key(engineering['Roff8'])==key(engineering['RC8shadow'])
assert key(engineering['RC14default'])==key(engineering['RC14explicit'])
assert all(d['forced_preemption'] and d['replay_counts']['restored']>=1 for d in engineering.values())
summary=dict(status='completed_training_screen_and_short_speed_check',execution_commit=read(root/'closure.json')['execution_commit'],
    groups=compact,comparisons=analysis['comparisons'],decision=analysis['decision'],speed=speed,
    engineering=dict(passed=True,answers=48,pure_generation_seconds=sum(d['generation_seconds'] for d in engineering.values())),
    stages={p.stem:read(p) for p in (root/'launch').glob('*_exit.json')},
    model_and_source='Pinned plan hashes and collection manifest verified',extra_probes=0,auxiliary_forwards=0,
    control_gpu_seconds=None,screen_records_verified=300,paired_bootstrap_reproduced=True,
    confirmation_generated=0,limitations=analysis['limitations'])
with args.output.open('x',encoding='utf8') as f:json.dump(summary,f,ensure_ascii=False,indent=2)
print(json.dumps(summary,ensure_ascii=False,indent=2))
