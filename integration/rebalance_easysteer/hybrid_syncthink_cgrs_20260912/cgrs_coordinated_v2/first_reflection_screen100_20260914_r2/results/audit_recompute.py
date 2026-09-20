import hashlib,json,sys
from pathlib import Path
root=Path('E:/srtp/B-history-screen-r2_0914')
repo=Path('E:/srtp/hybrid-syncthink-cgrs-20260912')
v=repo/'integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912/cgrs_coordinated_v2'
sys.path.insert(0,str(v))
from full_grade import compare
from narrow_grade import decision
def read(p):return json.loads(p.read_text(encoding='utf8'))
manifest=read(root/'artifact_manifest.json')
for name,digest in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
analysis=read(root/'results/analysis.json');plan=read(root/'results/resolved_plan.json')
raw={};groups=analysis['groups'];compact=[]
for arm in plan['arms']:
    data=read(root/'results'/arm/'result.json');records=data['records'];g=groups[arm]
    assert len(records)==g['n']==100
    for i,(row,r,label) in enumerate(zip(plan['rows'],records,g['labels'])):
        assert row['problem_sha256']==r['problem_sha256']==label['problem_sha256']
        assert r['dataset_index']==label['dataset_index']==i
        assert len(r['token_ids'])==r['tokens']<=16000
        assert r['thinking_tokens']==(r['token_ids'].index(151649) if 151649 in r['token_ids'] else r['tokens'])
        assert hashlib.sha256(r['text'].encode()).hexdigest()==label['text_sha256']
        assert type(label['correct']) is bool
    assert sum(x['correct'] for x in g['labels'])==g['correct']
    assert sum(r['tokens'] for r in records)/100==g['mean_total_tokens']
    assert sum(r['thinking_tokens'] for r in records)/100==g['mean_thinking_tokens']
    assert sum(r['finish_reason']=='length' for r in records)==g['capped']
    assert g['generation_seconds']==data['generation_seconds']
    raw[arm]=[dict(r,correct=l['correct']) for r,l in zip(records,g['labels'])]
for base,candidate in [('R','RC14'),('R','RChistory'),('RC14','RChistory')]:
    assert compare(raw[base],raw[candidate],groups[candidate]['labels'])==analysis['comparisons'][candidate+'_vs_'+base]
assert decision(groups,'RChistory')==analysis['decision']
for i,row in enumerate(plan['rows']):
    entry={k:row[k] for k in ['train_index','problem_sha256','dataset_index']}
    entry['arms']={arm:{k:raw[arm][i][k] for k in ['tokens','thinking_tokens','correct','finish_reason']} for arm in plan['arms']}
    for base in ['R','RC14']:
        a=raw[base][i]['token_ids'];b=raw['RChistory'][i]['token_ids']
        entry['first_difference_from_'+base]=next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)) if len(a)!=len(b) else None)
    compact.append(entry)
summary={k:v for k,v in analysis.items() if k!='groups'}
summary['groups']={arm:{k:v for k,v in group.items() if k!='labels'} for arm,group in groups.items()}
summary['local_audit']=dict(manifest_files=len(manifest),all_hashes=True,all_metrics_recomputed=True,
    bootstrap_recomputed_exact=True,author_labels_reused_not_regraded=True,
    identity_vs_R=sum(r['first_difference_from_R'] is None for r in compact),
    identity_vs_RC14=sum(r['first_difference_from_RC14'] is None for r in compact))
summary['per_question']=compact
summary['closure']=read(root/'closure.json')
summary['process']=read(root/'launch/history_complete.json')
summary['raw_root']=str(root)
out=v/'first_reflection_screen100_20260914_r2/results';out.mkdir(exist_ok=False)
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in summary.items() if k not in ['per_question','groups']},ensure_ascii=False))
print(json.dumps(summary['groups'],ensure_ascii=False))
