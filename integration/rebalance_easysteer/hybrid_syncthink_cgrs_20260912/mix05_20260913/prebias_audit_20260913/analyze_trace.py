"""Read-only analysis of the bounded GPU trace; no model imports or grading."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    status=read(a.run/'diagnostic_status.json')
    assert status['status']=='complete'
    result={arm:read(a.run/'gsm8k_mix05'/arm/'result.json') for arm in ('R','shadow_R','RS')}
    assert all(x['status']=='complete' and x['cap']==512 and len(x['records'])==64 and x['preemptions']==0 for x in result.values())
    prefix_identity={}
    per_question=[]
    for arm in ('R','RS'):
        old=read(a.reference/'gsm8k_mix05'/arm/'result.json')['records']
        prefix_identity[arm]=sum(x['token_ids']==y['token_ids'][:512] for x,y in zip(result[arm]['records'],old))
        assert prefix_identity[arm]==64
    for x,y in zip(result['R']['records'],result['shadow_R']['records']):
        assert x['token_ids']==y['token_ids'] and x['R_history']['sha256']==y['R_history']['sha256']
    target={arm:next(x for x in v['records'] if x['train_index']==5424) for arm,v in result.items()}
    assert target['RS']['hybrid']['trigger_count']==target['RS']['hybrid']['bias_count']==0
    for arm,v in result.items():
        for x in v['records']:
            assert len(x['token_ids'])==x['tokens']<=512
            per_question.append(dict(arm=arm,train_index=x['train_index'],tokens=x['tokens'],
                prompt_sha256=hashlib.sha256(json.dumps(x['prompt_token_ids'],separators=(',',':')).encode()).hexdigest(),
                output_sha256=hashlib.sha256(json.dumps(x['token_ids'],separators=(',',':')).encode()).hexdigest(),
                hybrid=x['hybrid'],R_history_sha256=x['R_history']['sha256']))
    snapshots=[]
    for t in range(409,418):
        metadata={arm:read(a.run/f'{arm}_5424_t{t}.json') for arm in result}
        arrays={arm:{kind:np.load(a.run/f'{arm}_5424_t{t}_{kind}.npy',allow_pickle=False) for kind in ('hidden','logits')} for arm in result}
        for kind in ('hidden','logits'):
            assert np.array_equal(arrays['R'][kind],arrays['shadow_R'][kind])
            assert all(np.isfinite(arrays[arm][kind]).all() for arm in result)
        assert all(m['sampling_seed']==42 and m['input_position']==67+t-1 for m in metadata.values())
        diff={}
        for kind in ('hidden','logits'):
            x,y=arrays['R'][kind].astype(np.float64),arrays['RS'][kind].astype(np.float64)
            diff[kind]=dict(max_abs=float(np.max(np.abs(y-x))),
                relative_l2=float(np.linalg.norm(y-x)/np.linalg.norm(x)),
                unequal_elements=int(np.count_nonzero(x!=y)),elements=int(x.size))
        snapshots.append(dict(position=t,metadata=metadata,differences=diff,
            R_state_equal=metadata['R']['R_state']==metadata['RS']['R_state'],
            same_prior_tokens=target['R']['token_ids'][:t]==target['RS']['token_ids'][:t],
            R_sampled_token=target['R']['token_ids'][t],RS_sampled_token=target['RS']['token_ids'][t]))
    first=next(x for x in snapshots if x['differences']['hidden']['max_abs']>0)
    assert first['position']==413 and first['same_prior_tokens'] and first['R_state_equal']
    assert first['metadata']['R']['padded_reqs']==48 and first['metadata']['RS']['padded_reqs']==40
    transitions=[]
    traces={arm:[json.loads(x) for x in (a.run/f'{arm}_trace.jsonl').read_text(encoding='utf-8').splitlines()] for arm in result}
    for step in range(min(len(traces['R']),len(traces['RS']))):
        x,y=traces['R'][step],traces['RS'][step]
        if x['padded_tokens']!=y['padded_tokens']:
            transitions.append(dict(step=step,R_padded=x['padded_tokens'],RS_padded=y['padded_tokens'],R_active=x['num_reqs'],RS_active=y['num_reqs']))
    assert transitions and transitions[0]['step']==413
    analyzed=dict(status='target_GSM_case_localized_to_forward_numerical_path',
        old_run_prefix_identity=prefix_identity,shadow_token_and_R_history_identity=64,
        captured_shadow_hidden_and_logits_bitwise_identity=True,
        formal_answers_generated=0,diagnostic_answers=192,
        published_tokens=sum(x['tokens'] for x in per_question),
        target_train_index=5424,target_never_triggered_or_biased=True,
        whole_runtime=status,instrumentation=read(a.run/'instrumentation_identity.json'),
        closure=read(a.run/'closure.json'),first_actual_padding_difference=transitions[0],
        observed_case_count=1,all46_cases_individually_explained=False,
        early_write_and_slot_guards='passed during all3 groups',
        snapshots=snapshots,
        limitations=['Direct snapshots are for GSM8K5424 only. MATH has not received this GPU trace.',
          'The boundary coincides with changed hidden states before the end hook on matched prefix/R state/seed; a particular GEMM or attention kernel is not isolated.',
          'No counterfactual run with fixed batch shape or batch-invariant kernels was authorized or performed.',
          'Do not use512-token diagnostic outputs to grade or estimate compression/speed.',
          'No independent confirmation of the original mix05 efficacy or2pp noninferiority.'],
        raw_sha256={p.relative_to(a.run).as_posix():sha(p) for p in a.run.rglob('*') if p.is_file()})
    a.output.mkdir(exist_ok=False)
    for n,v in [('analysis.json',analyzed),('per_question.json',per_question)]:
        with (a.output/n).open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
    print('Verified192 diagnostic outputs; target413: ',first['differences'])


if __name__=='__main__':main()
