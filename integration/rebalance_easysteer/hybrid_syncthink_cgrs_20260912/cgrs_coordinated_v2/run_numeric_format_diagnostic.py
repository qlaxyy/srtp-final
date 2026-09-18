import hashlib,json,tarfile,time
import numpy as np
from prepare_length_vector import HERE,ROOT,read,save,sha
from numeric_format_diagnostic import extract,value,checks
from prepare_and_error_audit import phash

def main():
    start=time.monotonic();checks();out=HERE/'numeric_format_diagnostic_20260918_cpu';out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text\n',encoding='utf8')
    protocol=dict(scope='Four complete saved GSM1319 arms U/R/L27/AND; no model generation or official score replacement.',
        rules=['Require stopped generation and explicit think closure; ignore all reasoning-region answers.',
            'Prefer last balanced boxed expression in final region; accept one rational decimal/fraction with plain unit/LaTeX spacing only.',
            'No box: last nonempty final paragraph must contain one distinct numeric value and no question/negation/alternative wording.',
            'Unsupported expressions, malformed boxes, conflicting numbers and incomplete outputs remain unresolved; never choose extraction by gold.',
            'Compare extracted rational to numeric GSM gold exactly. Report all disagreements and unresolved outputs for every arm.'],
        limits='Posthoc diagnostic designed after8-pair audit, frozen before full scoring. Not official GSM evaluator or new independent evaluation. Literal answer equivalence is not reasoning validity.',
        unit_checks=checks(),extractor_sha256=sha(HERE/'numeric_format_diagnostic.py'))
    save(out/'protocol.json',protocol)
    refs=read(HERE/'strict_and_gsm_20260918_run1/historical_compact.json');sources={}
    for name in ('gsm8k_eval.json','gsm8k_author_grading.json'):
        p=ROOT/'.codex_work/auto_code_v2_500_20260908'/name
        expected=next(v for k,v in refs['source_sha256'].items() if k.endswith(name));assert sha(p)==expected;sources[str(p)]=expected
    original=read(ROOT/'.codex_work/auto_code_v2_500_20260908/gsm8k_eval.json')
    groups={'U':original['baseline']['records'],'R':original['rebalance_dynamic']['records']}
    prior=HERE.parents[3]/'.codex_work/label_alignment_20260917/label_alignment_gsm1319_20260917_run1/L27_L27/result.json'
    assert sha(prior)==next(v for k,v in refs['source_sha256'].items() if k.endswith('L27_L27\\result.json'))
    sources[str(prior)]=sha(prior);groups['L27']=read(prior)['records']
    archive=HERE.parents[3]/'.codex_work/strict_and_gsm_20260918_run1.completed.tar.gz';assert sha(archive)=='faf1913255c3fba9d9fa32509d04a48c2e6a55daf2f77089a49ef63a53884e62';sources[str(archive)]=sha(archive)
    with tarfile.open(archive) as t:
        base='strict_and_gsm_20260918_run1/full/';groups['AND']=json.load(t.extractfile(base+'AND_NORM_L27/result.json'))['records']
        labels=[json.loads(s) for s in t.extractfile(base+'AND_NORM_L27/author_partial.jsonl').read().splitlines()]
    author={k:[r['correct'] for r in refs['groups'][n]['records']] for k,n in [('U','U'),('R','R'),('L27','L27_L27')]};author['AND']=[r['correct'] for r in labels]
    questions=read(HERE/'strict_and_gsm_20260918_run1/plan.json')['rows'];rows=[];summaries={};disagreements=[];unresolved=[]
    for arm,rs in groups.items():
        assert len(rs)==len(author[arm])==1319;armrows=[]
        for i,(r,q,old) in enumerate(zip(rs,questions,author[arm])):
            assert r['dataset_index']==q['dataset_index']==i and phash(r['problem'])==q['problem_sha256']
            assert len(r['token_ids'])==r['tokens']<=16000
            gold=value(q['answer']);assert gold is not None
            d=extract(r['text'],r['finish_reason']);correct=value(d['value'])==gold if d['status']=='resolved' else None
            row=dict(arm=arm,dataset_index=i,problem_sha256=q['problem_sha256'],text_sha256=hashlib.sha256(r['text'].encode()).hexdigest(),original_correct=old,
                supplemental_correct=correct,extraction=d,gold=str(gold),tokens=r['tokens'],thinking_tokens=r['thinking_tokens'])
            rows.append(row);armrows.append(row)
            if correct is None:unresolved.append(dict(row,final_region=r['text'].rsplit('</think>',1)[-1]))
            elif correct!=old:disagreements.append(dict(row,final_region=r['text'].rsplit('</think>',1)[-1]))
        resolved=[r for r in armrows if r['supplemental_correct'] is not None]
        known=sum(r['supplemental_correct'] is True for r in armrows);unknown=1319-len(resolved)
        summaries[arm]=dict(original_correct=sum(author[arm]),resolved=len(resolved),unresolved=unknown,resolved_correct=known,
            unresolved_0_or_1_arithmetic_range_percent=[100*known/1319,100*(known+unknown)/1319],
            original_false_resolved_true=sum(not r['original_correct'] and r['supplemental_correct'] is True for r in armrows),
            original_true_resolved_false=sum(r['original_correct'] and r['supplemental_correct'] is False for r in armrows),
            original_true_unresolved=sum(r['original_correct'] and r['supplemental_correct'] is None for r in armrows))
    lookup={(r['arm'],r['dataset_index']):r for r in rows};paired={}
    for basearm in ('U','R','L27'):
        lower=[];upper=[]
        for i in range(1319):
            a=lookup[basearm,i]['supplemental_correct'];b=lookup['AND',i]['supplemental_correct']
            lower.append(int(b is True)-int(a is not False));upper.append(int(b is not False)-int(a is True))
        paired[basearm]=dict(and_minus_base_unresolved_0_or_1_range_pp=[100*np.mean(lower),100*np.mean(upper)])
    save(out/'rows.json',rows);save(out/'disagreements.json',disagreements);save(out/'unresolved.json',unresolved)
    result=dict(summaries=summaries,paired=paired,seconds=time.monotonic()-start,source_sha256=sources,
        protocol_sha256=sha(out/'protocol.json'),output_sha256={n:sha(out/n) for n in ('rows.json','disagreements.json','unresolved.json')},
        limits='Arithmetic ranges reflect unresolved assignments only, not statistical CIs or guaranteed semantic accuracy bounds. Original scores unchanged. No model or grader source modifications.',gpu_calls=0,new_answers=0)
    save(out/'result.json',result);print(json.dumps(result,ensure_ascii=True))
if __name__=='__main__':main()
