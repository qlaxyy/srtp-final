"""CPU-only exposed-training audit. No inference, new labels or parameter search."""
import argparse,hashlib,json,re
from pathlib import Path
from engineering import save,sha

def first_difference(a,b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None if len(a)==len(b) else min(len(a),len(b)))

def boxes(text):
    # Exact complete TeX boxed expressions; normalization never uses the answer key.
    found=[]
    for m in re.finditer(r'\\boxed\s*\{',text):
        depth=1;i=m.end()
        while i<len(text) and depth:
            escaped=i>0 and text[i-1]=='\\'
            if not escaped:
                if text[i]=='{':depth+=1
                elif text[i]=='}':depth-=1
            i+=1
        if not depth:
            value=re.sub(r'\s+','',text[m.end():i-1])
            if value:found.append(dict(start=m.start(),end=i,value=value))
    return found

def repeated_box_prefix(text):
    thinking=text.split('</think>',1)[0];bs=boxes(thinking)
    for a,b in zip(bs,bs[1:]):
        if a['value']==b['value']:
            return dict(offset=b['end'],value=b['value'],remaining_thinking_chars=len(thinking)-b['end'])
    return None

def summarize(rows):
    groups={}
    for category in ('both_correct','correct_to_wrong','wrong_to_correct','both_wrong'):
        subset=[r for r in rows if r['category']==category]
        groups[category]=dict(n=len(subset),thinking_delta_sum=sum(r['thinking_delta'] for r in subset),total_delta_sum=sum(r['total_delta'] for r in subset),
            shorter_thinking=sum(r['thinking_delta']<0 for r in subset),longer_thinking=sum(r['thinking_delta']>0 for r in subset))
    stable=[r for r in rows if not r['either_capped']]
    return dict(n=len(rows),unique_questions=len({r['problem_sha256'] for r in rows}),categories=groups,
        thinking_delta_sum=sum(r['thinking_delta'] for r in rows),total_delta_sum=sum(r['total_delta'] for r in rows),
        cap_involved_pairs=sum(r['either_capped'] for r in rows),cap_pair_thinking_delta_sum=sum(r['thinking_delta'] for r in rows if r['either_capped']),
        uncapped_pairs=len(stable),uncapped_thinking_delta_sum=sum(r['thinking_delta'] for r in stable),
        prefix_groups={k:sum(r['prefix_relation']==k for r in rows) for k in sorted({r['prefix_relation'] for r in rows})})

def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.raw/'manifest.json').read_text());inputs={};data={};signals={}
    for seed in (142,242):
        for arm in ('R','RC14','RChistory'):
            folder=args.raw/'results'/f's{seed}_{arm}';result=folder/'result.json';label_path=folder/'stopped_batch_author_labels.jsonl'
            for f in (result,label_path):
                assert sha(f)==manifest[f.relative_to(args.raw).as_posix()]
                inputs[str(f)]=sha(f)
            d=json.loads(result.read_text());labels={x['dataset_index']:x for x in map(json.loads,label_path.read_text().splitlines())};assert d['status']=='complete' and len(d['records'])==100
            ordered=sorted(d['records'],key=lambda x:x['dataset_index']);assert [r['dataset_index'] for r in ordered]==list(range(100))
            seen=[]
            for r in ordered:
                label=labels[r['dataset_index']];assert label['problem_sha256']==r['problem_sha256'] and label['text_sha256']==hashlib.sha256(r['text'].encode()).hexdigest()
                r['correct']=label['correct'];r['event']=d['events'].get(r['request_id'],{})
                assert len(r['token_ids'])==r['tokens']<=16000
                assert r['thinking_tokens']==(r['token_ids'].index(151649) if 151649 in r['token_ids'] else r['tokens'])
                assert ('</think>' in r['text'])==(151649 in r['token_ids'])
                signal=repeated_box_prefix(r['text'])
                if signal:seen.append(dict(train_index=r['train_index'],problem_sha256=r['problem_sha256'],final_correct=r['correct'],capped=r['finish_reason']=='length',**signal))
            data[seed,arm]=ordered;signals[f's{seed}_{arm}']=dict(n=100,covered=len(seen),finally_wrong=sum(not x['final_correct'] for x in seen),rows=seen)
    allrows=[];grouped={}
    for seed in (142,242):
        for base,candidate in [('R','RC14'),('RC14','RChistory')]:
            rows=[]
            for a,b in zip(data[seed,base],data[seed,candidate]):
                assert a['problem_sha256']==b['problem_sha256']
                diff=first_difference(a['token_ids'],b['token_ids'])
                masks=[r['event'].get('first_change',-1) for r in (a,b)];masks=[k for k in masks if k>=0]
                first=min(masks) if masks else None
                relation='identical' if diff is None else ('different_without_either_mask' if first is None else ('different_before_either_mask' if diff<first else 'different_at_or_after_mask_not_causal_proof'))
                category='both_correct' if a['correct'] and b['correct'] else 'correct_to_wrong' if a['correct'] else 'wrong_to_correct' if b['correct'] else 'both_wrong'
                rows.append(dict(seed=seed,base=base,candidate=candidate,train_index=a['train_index'],problem_sha256=a['problem_sha256'],category=category,base_thinking=a['thinking_tokens'],candidate_thinking=b['thinking_tokens'],thinking_delta=b['thinking_tokens']-a['thinking_tokens'],total_delta=b['tokens']-a['tokens'],either_capped=a['finish_reason']=='length' or b['finish_reason']=='length',first_difference=diff,earliest_mask=first,prefix_relation=relation))
            key=f's{seed}_{candidate}_vs_{base}';grouped[key]=summarize(rows);allrows+=rows
    save(out/'pairs.json',allrows)
    # Read every correctness flip; this selection is diagnostic, never a validation split.
    cases=[]
    for r in allrows:
        if r['category'] not in ('correct_to_wrong','wrong_to_correct'):continue
        a=next(x for x in data[r['seed'],r['base']] if x['train_index']==r['train_index']);b=next(x for x in data[r['seed'],r['candidate']] if x['train_index']==r['train_index'])
        cases.append(dict(**r,problem=a['problem'],gold=a['answer'],base_text=a['text'],candidate_text=b['text']))
    save(out/'flip_cases.json',cases)
    save(out/'audit.json',dict(status='cpu_exposed_training_descriptive',source_sha256=sha(Path(__file__),True),input_sha256=inputs,groups=grouped,repeated_box_hypothesis=dict(rule='First two consecutive complete identical boxed expressions inside thinking; whitespace-only normalization; fixed before this audit; no threshold sweep, answer key or model forward used by detector.',results=signals),limitations=['600 existing outputs,100 unique exposed training questions; repeated seeds are not independent questions.','Comparisons do not isolate mask causality: batch numerical effects and post-divergence paths remain.','Correctness of final answer is not correctness at detector time. Remaining chars are not saved tokens.','No stored per-step max-probability series or logits here; cannot reconstruct confidence or fit a new confidence gate.','Cap-stratified and correctness-stratified analyses are descriptive post-treatment partitions, not cherry-picked main metrics.'],gpu_ready=False))
    print(json.dumps(dict(groups=grouped,box_coverage={k:{x:v[x] for x in ('covered','finally_wrong')} for k,v in signals.items()},flip_cases=len(cases)),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
