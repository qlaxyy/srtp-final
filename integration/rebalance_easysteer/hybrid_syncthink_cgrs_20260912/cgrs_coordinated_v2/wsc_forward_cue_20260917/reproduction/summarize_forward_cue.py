from pathlib import Path
import json,hashlib,unicodedata
root=Path('E:/srtp/hybrid-syncthink-cgrs-20260912')
b=root/'integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912/cgrs_coordinated_v2'
raw=root/'.codex_work/wsc_forward_cue_complete_20260917'
report=json.loads((raw/'active_audit.json').read_text(encoding='utf8'))
out=b/'wsc_forward_cue_20260917/results_run1';out.mkdir(exist_ok=True)
dataset=root/'sources/ReBalance/Data/Math_Train/test.jsonl'
data=[json.loads(x) for x in dataset.read_text(encoding='utf8').splitlines()]
def boxed(s):
    start=s.rfind('\\boxed{')
    if start<0:return None
    start+=7;balance=1
    for i in range(start,len(s)):
        if s[i]=='{':balance+=1
        elif s[i]=='}':balance-=1
        if balance==0:return s[start:i]
    return None
def norm(x):return ''.join(x.split()).replace('\\dfrac','\\frac')
rows=[]
for r in report['rows']:
    d=data[r['train_index']]
    h=hashlib.sha256(''.join(unicodedata.normalize('NFKC',d['problem']).split()).encode()).hexdigest()
    assert h==r['problem_sha256']
    answer=boxed(r['answer_text']);old=boxed(r['reference_answer_text'])
    correct=norm(answer)==norm(d['answer']);oldcorrect=norm(old)==norm(d['answer'])
    assert correct==(r['train_index'] not in (4432,229))
    assert oldcorrect==(r['train_index'] not in (647,4432,229))
    rows.append({k:v for k,v in r.items() if k not in ('text','answer_text','reference_answer_text')}|
                dict(final_answer=answer,reference_final_answer=old,gold=d['answer'],correct=correct,reference_correct=oldcorrect))
sums={k:sum(r[k] for r in rows) for k in ('total_tokens','reference_total_tokens','thinking_tokens','reference_thinking_tokens')}
summary=dict(phase='exposed_engineering_only',questions=8,model='DeepSeek-R1-Distill-Qwen-1.5B',dataset='MATH Train',seed=42,max_tokens=16000,
    correct=sum(r['correct'] for r in rows),reference_correct=sum(r['reference_correct'] for r in rows),
    wrong_to_right=[r['train_index'] for r in rows if r['correct'] and not r['reference_correct']],right_to_wrong=[],
    mean_total_tokens=sums['total_tokens']/8,reference_mean_total_tokens=sums['reference_total_tokens']/8,
    mean_thinking_tokens=sums['thinking_tokens']/8,reference_mean_thinking_tokens=sums['reference_thinking_tokens']/8,
    total_token_delta_percent=100*(sums['total_tokens']/sums['reference_total_tokens']-1),
    thinking_token_delta_percent=100*(sums['thinking_tokens']/sums['reference_thinking_tokens']-1),
    caps=sum(r['finish_reason']=='length' for r in rows),reference_caps=0,cue_tokens=sum(r['cue_tokens'] for r in rows),
    generation_seconds=report['generation_seconds'],reference_generation_seconds=report['reference_generation_seconds'],
    wall_seconds=245.3187284618616,reference_wall_seconds=168.6031699925661,
    detector_score_seconds=report['cue_report']['score_seconds'],host_observation_seconds=report['cue_report']['host_observation_seconds'],
    timing_note='Generation window includes instrumentation; detector time is a subset of host observation time, which includes synchronization. Not additive or a pure overhead comparison.',
    added_probe_model_forward_calls=0,all_pre_intervention_token_control_R_prefixes_match=report['all_prefixes_eligible'],
    online_offline_scores_max_difference=max(r['online_offline_max_score_diff'] for r in rows),
    verdict='Implementation gate passed; this fixed cue did not demonstrate compression. Do not promote to full benchmark or retune cue/threshold on these rows. Retain RC14.',
    limitation='Selected exposed eight-question engineering sample. No independent accuracy, noninferiority, confidence interval or synergy claim.',
    grader='Manual final-answer review plus balanced-box comparison (whitespace and dfrac presentation normalized) against hashed dataset rows; not a replacement benchmark grader.',
    dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),rows=rows)
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in summary.items() if k!='rows'},ensure_ascii=False))
