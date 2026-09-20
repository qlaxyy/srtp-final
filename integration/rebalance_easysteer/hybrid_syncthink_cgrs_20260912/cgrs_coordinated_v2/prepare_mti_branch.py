"""Freeze stage-0 inputs from already exposed calibration; CPU only."""
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from test_mti_branch_cpu import run

HERE=Path(__file__).resolve().parent
OUT=HERE/'mti_context_20260918'


def main():
    source=HERE/'wsc_probe_diagnostic_20260917/plan.json'
    old=json.loads(source.read_text(encoding='utf8'))
    lengths=[255,256,257,511,512,513,767,768]
    assert [x['train_index'] for x in old['cases']]==[3241,5353,759,7012,26,64,76,385]
    cases=[]
    for case,n in zip(old['cases'],lengths):
        ids=case['prompt_token_ids']+case['token_ids']
        assert len(case['prompt_token_ids'])<n<len(ids)
        text=re.sub(r'\s+','',unicodedata.normalize('NFKC',case['problem']))
        prefix=ids[:n]
        cases.append(dict(train_index=case['train_index'],
            normalized_problem_sha256=hashlib.sha256(text.encode()).hexdigest(),
            purpose='Existing exposed calibration prefix, engineering only; no independent confirmation',
            prefix_ids=prefix,next_saved_token=ids[n],
            prefix_sha256=hashlib.sha256(json.dumps(prefix,separators=(',',':')).encode()).hexdigest()))
    files=['mti_branch.py','mti_branch_diagnostic.py']
    plan=dict(kind='mti_stage0_unsteered_cache_oracle',run_id='mti_stage0_prefix8_20260918_run1',
        model='DeepSeek-R1-Distill-Qwen-1.5B',model_path=old['model_path'],model_files=old['model_files'],
        cue='OUTPUT ERROR',cases=cases,generated_tokens=0,prefix_input_tokens=sum(lengths),
        forward_calls=40,max_probability_error=.002,hard_stop_seconds=600,
        expected_gpu_wall_minutes=[2,5],cost_estimate_status='Engineering estimate, not measured; no speed guarantee',
        source_plan_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_sha256_lf={f:hashlib.sha256((HERE/f).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for f in files},
        scope='HF unsteered cache oracle only. No answer grading, no vLLM/batched/steered equivalence claim.',
        gpu_executed=False,full_evaluation_allowed=False)
    OUT.mkdir(exist_ok=True)
    for name,data in [('stage0_plan.json',plan),('branch_cpu_checks.json',run())]:
        target=OUT/name
        text=json.dumps(data,ensure_ascii=False,indent=2)+'\n'
        if target.exists():
            assert target.read_text(encoding='utf8')==text,'Refuse to change frozen preparation: '+name
        else:target.write_text(text,encoding='utf8')
    print(json.dumps(dict(prefixes=len(cases),prefix_tokens=sum(lengths),gpu_used=False,
        checks=json.loads((OUT/'branch_cpu_checks.json').read_text())['checks_passed']),indent=2))


if __name__=='__main__':main()
