"""Frozen descriptive flip audit and separate calibration content sample; no fitting."""
import collections,hashlib,json,tarfile,unicodedata
import numpy as np
from prepare_length_vector import HERE,ROOT,read,save,sha
from review_wsc_native_cpu import Decoder

def phash(s):return hashlib.sha256(''.join(unicodedata.normalize('NFKC',s).split()).encode()).hexdigest()
def main():
    out=HERE/'and_error_audit_20260918_cpu';out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text\n',encoding='utf8')
    archive=HERE.parents[3]/'.codex_work/strict_and_gsm_20260918_run1.completed.tar.gz'
    assert sha(archive)=='faf1913255c3fba9d9fa32509d04a48c2e6a55daf2f77089a49ef63a53884e62'
    old=HERE.parents[3]/'.codex_work/label_alignment_20260917/label_alignment_gsm1319_20260917_run1'
    old_analysis=read(old/'analysis.json')['candidates']['L27_L27']
    assert sha(old/'L27_L27/result.json')==old_analysis['source_result_sha256']
    assert sha(old/'L27_L27/author_partial.jsonl')==old_analysis['labels_sha256']
    protocol=dict(selection='Test: four per loss/gain stratum by SHA256(and-error-v1:problem_sha256). Calibration: eight parents by SHA256(and-calibration-v1:problem_sha256), earliest strict-AND step >=16 tokens. Fixed before reading content.',
        test_use='All160 flips quantitative; 8 pairs qualitative. Test content cannot be used to fit vectors, set thresholds, or choose training examples.',
        calibration_use='Original500 saved trajectories only. Local content review, not final-answer-derived step gold. No labels inferred from matching test topics.',
        rubric=['missing_or_wrong_operation','arithmetic_error','interpretation_error','necessary_correction_visible','harmful_revision_visible','answer_extraction_issue','uncertain'],
        caveat='Single assistant qualitative review; no prevalence or causal mechanism inference from eight selected pairs. More than one category can apply.',new_gpu=0,new_answers=0)
    save(out/'protocol.json',protocol)
    with tarfile.open(archive) as t:
        base='strict_and_gsm_20260918_run1/full/'
        data=json.load(t.extractfile(base+'AND_NORM_L27/result.json'))
        labels=[json.loads(x) for x in t.extractfile(base+'AND_NORM_L27/author_partial.jsonl').read().splitlines()]
        analysis=json.load(t.extractfile(base+'analysis.json'))
        assert hashlib.sha256(t.extractfile(base+'AND_NORM_L27/result.json').read()).hexdigest()==analysis['candidates']['AND_NORM_L27']['source_result_sha256']
    baseline=read(old/'L27_L27/result.json')['records'];blabel=[json.loads(x) for x in (old/'L27_L27/author_partial.jsonl').read_bytes().splitlines()]
    rows=[];pairs={}
    for a,b,la,lb in zip(baseline,data['records'],blabel,labels):
        assert a['dataset_index']==b['dataset_index']==la['dataset_index']==lb['dataset_index']
        assert a['problem_sha256']==b['problem_sha256']==la['problem_sha256']==lb['problem_sha256']==phash(a['problem'])
        for rec,label in ((a,la),(b,lb)):assert hashlib.sha256(rec['text'].encode()).hexdigest()==label['text_sha256']
        status='loss' if la['correct'] and not lb['correct'] else 'gain' if lb['correct'] and not la['correct'] else 'both_correct' if la['correct'] else 'both_wrong'
        common=next((i for i,(x,y) in enumerate(zip(a['token_ids'],b['token_ids'])) if x!=y),min(a['tokens'],b['tokens']))
        row=dict(dataset_index=a['dataset_index'],problem_sha256=a['problem_sha256'],status=status,common_prefix_tokens=common,
            old_total=a['tokens'],new_total=b['tokens'],old_thinking=a['thinking_tokens'],new_thinking=b['thinking_tokens'],old_cap=a['tokens']==16000,new_cap=b['tokens']==16000)
        rows.append(row);pairs[a['dataset_index']]=dict(row,problem=a['problem'],old_text=a['text'],new_text=b['text'])
    assert collections.Counter(r['status'] for r in rows)['loss']==97 and collections.Counter(r['status'] for r in rows)['gain']==63
    summary={}
    for group in ('loss','gain','both_correct','both_wrong'):
        rs=[r for r in rows if r['status']==group]
        summary[group]=dict(count=len(rs),total_delta_sum=sum(r['new_total']-r['old_total'] for r in rs),thinking_delta_sum=sum(r['new_thinking']-r['old_thinking'] for r in rs),
            shorter=sum(r['new_total']<r['old_total'] for r in rs),longer=sum(r['new_total']>r['old_total'] for r in rs),
            common_prefix_quantiles=np.quantile([r['common_prefix_tokens'] for r in rs],[0,.5,.9,1]).tolist())
    selected=[]
    for group in ('loss','gain'):
        selected.extend(sorted((r for r in rows if r['status']==group),key=lambda r:hashlib.sha256(('and-error-v1:'+r['problem_sha256']).encode()).hexdigest())[:4])
    save(out/'test_registry.json',dict(rows=rows,summary=summary,source_sha256=sha(archive),selected=[r['dataset_index'] for r in selected]))
    save(out/'test_cases.json',dict(cases=[pairs[r['dataset_index']] for r in selected]))
    with tarfile.open(ROOT/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz') as t:raw=t.extractfile('generations.jsonl').read()
    assert hashlib.sha256(raw).hexdigest()=='4d0194b30bb97d1ef0779b13d24ba47ba0aa7ccf38803e6a68f5c233ceb756a3'
    gens=[json.loads(x) for x in raw.splitlines()];steps=read(ROOT/'.codex_work/question_balanced_20260911/original_selected_layer/steps.json')
    lo=read(ROOT/'.codex_work/auto_code_v2_500_20260908/protocol.json')['confidence_quantiles'][0]
    pool={}
    for i,s in enumerate(steps):
        if s['lexical_hit'] and s['confidence']<lo and s['stop']-s['start']>=16:pool.setdefault(s['question'],i)
    selected=sorted(pool,key=lambda q:hashlib.sha256(('and-calibration-v1:'+phash(gens[q]['problem'])).encode()).hexdigest())[:8]
    dec=Decoder(ROOT/'.codex_work/label_audit_30_20260910/tokenizer.json');cases=[]
    for q in selected:
        g=gens[q];i=pool[q];s=steps[i];ids=g['token_ids']
        cases.append(dict(question=q,train_index=g['train_index'],problem_sha256=phash(g['problem']),problem=g['problem'],step_index=i,
            start=s['start'],stop=s['stop'],confidence=s['confidence'],lexical_hit=s['lexical_hit'],
            before=dec.decode(ids[max(0,s['start']-350):s['start']]),target=dec.decode(ids[s['start']:s['stop']]),after=dec.decode(ids[s['stop']:min(len(ids),s['stop']+350)])))
    save(out/'calibration_cases.json',dict(cases=cases,eligible_questions=len(pool),source_sha256=hashlib.sha256(raw).hexdigest()))
    print(json.dumps(dict(summary=summary,test_indices=[r['dataset_index'] for r in json.loads((out/'test_cases.json').read_text(encoding='utf8'))['cases']],calibration_indices=[r['train_index'] for r in cases])))
if __name__=='__main__':main()
