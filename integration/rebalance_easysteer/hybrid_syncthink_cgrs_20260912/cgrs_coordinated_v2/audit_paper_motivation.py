"""Frozen-output grouping and training-side confidence audit; no circular labels."""
import json,tarfile,hashlib,unicodedata
import numpy as np
from scipy.stats import spearmanr
from prepare_length_vector import HERE,ROOT,save,read,sha
from review_wsc_native_cpu import Decoder

def main():
    out=HERE/'motivation_replication_20260918_cpu';out.mkdir(exist_ok=True);assert not (out/'result.json').exists()
    archive=ROOT/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz'
    sp=ROOT/'.codex_work/question_balanced_20260911/original_selected_layer/steps.json'
    ev=ROOT/'.codex_work/auto_code_v2_500_20260908/math500_artifacts.tar.gz'
    tokenizer=ROOT/'.codex_work/label_audit_30_20260910/tokenizer.json'
    protocol=dict(paper='https://arxiv.org/html/2603.12372v3',locations=['2.1 Eq1-2','2.2 Figure2b','A.2 Table6','C.6 Table19'],
        goal='First audit existing evidence; no inference that length alone defines thinking mode.',
        figure2_pair_proxy='Same question U vs current R adaptation: O pair iff both correct and R thinking tokens<U; U pair iff U correct,R wrong and R thinking tokens<U. No confidence-based selection.',
        figure2_missing_details=['Exact mitigation method(s) used for Figure2b','Pair IDs and count','Aggregation weighting','Figure2 length unit','Whether Figure2 variance is whole-trajectory or averaged sliding variance','Probability distribution before/after sampling warpers'],
        supplement='500 training trajectories, not MATH500; recompute geometric and arithmetic step confidence on frozen greedy raw max logprobs and old step boundaries. Parent equal weighting.',
        stats='Spearman length vs minimum/mean confidence, whole-step variance and mean W2 variance. 2000 paired-parent percentile bootstrap, seed20260918. Report all and completed-only, no statistical efficacy claims.',
        no_new_gpu=True,no_new_generation=True,
        inputs={str(p):sha(p) for p in (archive,sp,ev,tokenizer)})
    with tarfile.open(archive) as t: generation_bytes=t.extractfile('generations.jsonl').read()
    assert hashlib.sha256(generation_bytes).hexdigest()=='4d0194b30bb97d1ef0779b13d24ba47ba0aa7ccf38803e6a68f5c233ceb756a3'
    protocol['generation_member_sha256']=hashlib.sha256(generation_bytes).hexdigest()
    assert protocol['inputs'][str(sp)]=='fc1e677fa50b834eec62a4caebbc39d341f93716624d4deca9c427a5fa0c8f93'
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==protocol
    else:save(out/'protocol.json',protocol)
    with tarfile.open(archive) as t:rows=[json.loads(x) for x in t.extractfile('generations.jsonl').read().splitlines()]
    steps=read(sp);by=[[] for _ in rows]
    for s in steps:by[s['question']].append(s)
    decoder=Decoder(tokenizer);parents=[]
    for q,row in enumerate(rows):
        lp=np.asarray(row['logprobs'],dtype=float);ids=row['token_ids']
        assert len(lp)==len(ids) and np.isfinite(lp).all() and (lp<=1e-5).all()
        closed=151649 in ids;stop=ids.index(151649) if closed else len(ids)
        item=dict(train_index=row['train_index'],problem_sha256=hashlib.sha256(''.join(unicodedata.normalize('NFKC',row['problem']).split()).encode()).hexdigest(),
                  steps=len(by[q]),thinking_tokens=stop,thinking_words=len(decoder.decode(ids[:stop]).split()),closed=closed,finish_reason=row.get('finish_reason'))
        for mode in ('arithmetic','geometric'):
            vals=np.array([np.exp(lp[s['start']:s['stop']]).mean() if mode=='arithmetic' else np.exp(lp[s['start']:s['stop']].mean()) for s in by[q]])
            assert len(vals)>0
            if mode=='arithmetic':assert np.allclose(vals,[s['confidence'] for s in by[q]],atol=1e-12,rtol=1e-12)
            local=np.r_[0,np.diff(vals)**2/4]
            item[mode]=dict(mean=float(vals.mean()),minimum=float(vals.min()),whole_step_variance=float(vals.var()),mean_local_w2_variance=float(local.mean()))
        parents.append(item)
    if (out/'calibration_parent_statistics.json').exists():assert read(out/'calibration_parent_statistics.json')==parents
    else:save(out/'calibration_parent_statistics.json',parents)
    results={}
    for subset in ('all','closed_only'):
        selected=[p for p in parents if subset=='all' or p['closed']]
        results[subset]=dict(n=len(selected),modes={})
        for mode in ('arithmetic','geometric'):
            metrics={}
            for metric in ('mean','minimum','whole_step_variance','mean_local_w2_variance'):
                x=np.array([p['thinking_words'] for p in selected]);y=np.array([p[mode][metric] for p in selected])
                rng=np.random.default_rng(20260918);boots=[]
                for _ in range(2000):
                    ix=rng.integers(0,len(x),len(x));boots.append(spearmanr(x[ix],y[ix])[0])
                metrics[metric]=dict(rho=float(spearmanr(x,y)[0]),ci95=np.quantile(boots,[.025,.975]).tolist())
            results[subset]['modes'][mode]=metrics
    with tarfile.open(ev) as t:
        evaluation=json.load(t.extractfile('math500_eval.json'))
        grading=json.load(t.extractfile('math500_author_grading.json'))
    # Use frozen AUTHOR labels, not the older math_verify labels in raw evaluation.
    u=evaluation['baseline']['records'];r=evaluation['rebalance_dynamic']['records']
    assert len(u)==len(r)==500
    for name,records in [('baseline',u),('rebalance_dynamic',r)]:
        labels=grading['groups'][name]['records'];assert len(labels)==len(records)
        for record,label in zip(records,labels):
            assert record['dataset_index']==label['index']
            record['correct']=label['author_correct']
    assert sum(x['correct'] for x in u)==422 and sum(x['correct'] for x in r)==411
    pairs={'over_proxy':[],'under_proxy':[]}
    for a,b in zip(u,r):
        assert a['problem']==b['problem'] and a['dataset_index']==b['dataset_index']
        if b['thinking_tokens']>=a['thinking_tokens']:continue
        kind='over_proxy' if a['correct'] and b['correct'] else 'under_proxy' if a['correct'] and not b['correct'] else None
        if kind:
            pairs[kind].append(dict(dataset_index=a['dataset_index'],problem_sha256=hashlib.sha256(''.join(unicodedata.normalize('NFKC',a['problem']).split()).encode()).hexdigest(),
                U_thinking_tokens=a['thinking_tokens'],R_thinking_tokens=b['thinking_tokens'],U_total_tokens=a['tokens'],R_total_tokens=b['tokens'],
                U_cap=a['tokens']==16000,R_cap=b['tokens']==16000,U_closed=a['thinking_ended'],R_closed=b['thinking_ended']))
    save(out/'math500_pair_registry.json',pairs)
    pair_summary={k:dict(n=len(v),U_mean_thinking_tokens=float(np.mean([p['U_thinking_tokens'] for p in v])),R_mean_thinking_tokens=float(np.mean([p['R_thinking_tokens'] for p in v])),
       both_closed=sum(p['U_closed'] and p['R_closed'] for p in v),any_capped=sum(p['U_cap'] or p['R_cap'] for p in v)) for k,v in pairs.items()}
    save(out/'result.json',dict(training_supplement=results,figure2_pair_summary=pair_summary,
        missing='Frozen MATH500 records contain tokens/text/labels, not step confidence or token logprobs. Figure2 confidence/variance differences remain unmeasured; do not populate from training supplement.',
        interpretation='Geometric/arithmetic agreement tests A2-style association on different data only. Pair labels are outcome-based proxies, not verified semantic modes; R intervention uses confidence itself.',
        script_sha256=sha(__import__('pathlib').Path(__file__))))
    print(json.dumps(dict(training_supplement=results,pairs=pair_summary),indent=2))

if __name__=='__main__':main()
