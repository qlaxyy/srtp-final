"""CPU contribution audit; no model import, fitted vector, or runtime change."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import tarfile
import time

import numpy as np
from audit_calibration_labels import ROOT, decoder, lexical_rule, read, save, sha

CONFIG=ROOT/'integration/rebalance_easysteer/configs'
PLAN=CONFIG/'calibration_question_weights_20260911.json'
OUT=ROOT/'.codex_work/calibration_question_weights_20260911'

def source_rows(model, folder):
    protocol=read(folder/'protocol.json')
    if model=='1.5B':
        with tarfile.open(ROOT/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz') as archive:
            manifest=json.load(archive.extractfile('manifest.json'))
            raw=archive.extractfile('generations.jsonl').read()
    else:
        manifest=read(folder/'calibration/manifest.json')
        raw=(folder/'calibration/generations.jsonl').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==protocol['source_sha256']
    rows=[json.loads(line) for line in raw.splitlines() if line]
    assert len(rows)==manifest['count']==500 and manifest['temperature']==0
    assert [r['train_index'] for r in rows]==manifest['train_indices']
    assert all(len(r['token_ids'])<=16000 for r in rows)
    return rows

def complete_steps(rows, tokenizer, params):
    decode,vocab=decoder(tokenizer)
    boundaries={i for text,i in vocab.items() if 'ĊĊ' in text}
    assert boundaries==set(params['boundary_token_ids'])
    lexical,_=lexical_rule()
    steps=[]
    for q,row in enumerate(rows):
        ids,logs=row['token_ids'],row['logprobs']
        assert len(ids)==len(logs) and all(math.isfinite(p) and p<=1e-5 for p in logs)
        closed=params['think_end_token_id'] in ids
        end=ids.index(params['think_end_token_id']) if closed else len(ids)
        first,previous=0,None
        for stop in range(end+1):
            if not (stop<end and ids[stop] in boundaries) and not (stop==end and closed):continue
            if stop>first:
                c=math.fsum(math.exp(p) for p in logs[first:stop])/(stop-first)
                v=0. if previous is None else (c-previous)**2/4
                steps.append(dict(question=q,confidence=c,variance=v,lexical_hit=bool(lexical(decode(ids[first:stop])))))
                previous=c
            first=stop+1
    return steps

def inverse_cdf(values,weights):
    order=np.argsort(values,kind='stable')
    cumulative=np.cumsum(weights[order])
    total=cumulative[-1]
    answers=[]
    for p in [.25,.75]:
        i=int(np.searchsorted(cumulative,p*total,side='left'))
        value=float(values[order[i]])
        # Independently check the weighted CDF including tied values.
        below=float(math.fsum(weights[values<value]))/float(math.fsum(weights))
        through=float(math.fsum(weights[values<=value]))/float(math.fsum(weights))
        assert below < p+1e-10 and through>=p-1e-10
        answers.append(value)
    return answers

def contributions(counts,capped):
    total=int(counts.sum())
    active=counts>0
    probabilities=counts[active]/total
    equal_class_cap=float(np.sum(active&capped)/np.sum(active))
    return dict(steps=total,questions_with_class=int(active.sum()),questions_without_class=int((~active).sum()),
        capped_step_share=float(counts[capped].sum()/total),capped_question_share_if_class_balanced=equal_class_cap,
        top50_step_share=float(np.sort(counts)[-50:].sum()/total),max_question_step_share=float(counts.max()/total),
        inverse_squared_question_mass=float(1/np.sum(probabilities**2)),
        concentration_note='Contribution concentration only; not a statistical independent sample count')

def main():
    assert not OUT.exists(),'Existing audit; inspect saved results rather than rerunning'
    plan=read(PLAN)
    assert plan['status']=='planned_cpu_audit'
    frozen_path=CONFIG/'final_results_20260909.json'
    protected={str(p.relative_to(ROOT)):sha(p) for p in [frozen_path,ROOT/'integration/rebalance_easysteer/scripts/calibrate_auto.py',ROOT/'sources/ReBalance/hidden_analysis_auto.py']}
    OUT.mkdir()
    save(OUT/'plan.json',plan)
    frozen=read(frozen_path)
    report=dict(status='completed_cpu_diagnostic',started_unix=time.time(),plan_sha256=sha(PLAN),models={},protected_before=protected)
    for model,relative in [('1.5B','.codex_work/auto_code_v2_500_20260908'),('7B','.codex_work/qwen7b_validation/auto_code_v2_qwen7b_20260908')]:
        folder=ROOT/relative
        protocol,fit=read(folder/'protocol.json'),read(folder/'fit.json')
        rows=source_rows(model,folder)
        if model=='1.5B':
            tokenizer=ROOT/'.codex_work/label_audit_30_20260910/tokenizer.json'
            assert sha(tokenizer)==frozen['assets']['models'][model]['tokenizer.json']['sha256']
            params=next(b for b in frozen['benchmarks'] if b['model']==model)['protocol']['dynamic_params']
            steps=complete_steps(rows,read(tokenizer),params)
        else:
            steps=read(folder/'steps.json')
        q=np.array([s['question'] for s in steps],dtype=int)
        c=np.array([s['confidence'] for s in steps],dtype=np.float64)
        v=np.array([s['variance'] for s in steps],dtype=np.float64)
        lex=np.array([s['lexical_hit'] for s in steps],dtype=bool)
        assert len(steps)==protocol['steps'] and len(q)==len(c)==len(v)
        for data,expected in [(c,protocol['confidence_quantiles']),(v,protocol['variance_quantiles'])]:
            np.testing.assert_allclose(np.quantile(data,[.25,.75]),expected,rtol=0,atol=1e-12)
        over=lex|(c<protocol['confidence_quantiles'][0])
        under=~lex&(c>protocol['confidence_quantiles'][1])
        assert int(over.sum())==fit['positives'] and int(under.sum())==fit['negatives']
        masks={'all':np.ones(len(q),dtype=bool),'over':over,'under':under,'unselected':~(over|under)}
        counts={name:np.bincount(q[mask],minlength=500) for name,mask in masks.items()}
        capped=np.array([r['finish_reason']=='length' for r in rows])
        assert all(len(r['token_ids'])==16000 for r in rows if r['finish_reason']=='length')
        weights=1/counts['all'][q]
        weight_sum=np.bincount(q,weights=weights,minlength=500)
        np.testing.assert_allclose(weight_sum[counts['all']>0],1,rtol=0,atol=1e-12)
        result=dict(questions=500,capped_questions=int(capped.sum()),complete_steps=len(steps),source_sha256=protocol['source_sha256'],
            fit_sha256=sha(folder/'fit.json'),scope='Frozen labels used only for counting prototype contributions',
            contributions={name:contributions(count,capped) for name,count in counts.items()},
            sensitivity={name:dict(frozen_type7=expected,step_uniform_inverse_cdf=inverse_cdf(values,np.ones(len(values))),question_uniform_inverse_cdf=inverse_cdf(values,weights))
                         for name,values,expected in [('confidence',c,protocol['confidence_quantiles']),('variance',v,protocol['variance_quantiles'])]},
            weighted_quantile_definition='Infimum x with normalized weighted CDF at least p; same inverse-CDF estimator in both arms; diagnostic only, no fitted parameters exported',
            per_question=[dict(calibration_index=i,train_index=r['train_index'],tokens=len(r['token_ids']),capped=bool(capped[i]),**{name:int(count[i]) for name,count in counts.items()}) for i,r in enumerate(rows)],
            selected_hidden_file_available_locally=(folder/f"layer_{fit['hidden_state_index']}.npy").exists())
        save(OUT/(model.replace('.','p')+'.json'),result)
        report['models'][model]=result
        print(json.dumps({model:{k:value for k,value in result.items() if k!='per_question'}},ensure_ascii=False),flush=True)
    report.update(completed_unix=time.time(),new_generations=0,model_forwards=0,server_connections=0,
                  protected_after={name:sha(ROOT/name) for name in protected})
    assert report['protected_before']==report['protected_after']
    save(OUT/'summary.json',report)
    print('CPU seconds: '+str(report['completed_unix']-report['started_unix']))

if __name__=='__main__':main()
