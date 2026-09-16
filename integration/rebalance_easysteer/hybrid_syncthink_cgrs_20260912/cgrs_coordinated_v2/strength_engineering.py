"""Bounded engineering-only protocol. No formal evaluation authorization."""
from pathlib import Path
from engineering import read,sha,ROOT
from prepare_screen import phash
from policy import token_flags,TRIGGERS
from calibrate_penalty_scale_cpu import vocabulary

CASES=[('R','off','original14'),('Roff_scaled','off','original14'),
       ('Rshadow_scaled','shadow','original14'),('RC14default','negative',None),
       ('RC14explicit','negative','original14'),('RCscaled','negative','original14'),
       ('RCconstant','negative','original14')]


def validate(plan):
    assert plan['candidate_kind']=='strength_engineering' and plan['phase']=='engineering_only'
    assert plan['model_family']=='1p5b' and plan['engineering_cap']==512
    assert len(plan['engineering_rows'])==8
    assert [r['train_index'] for r in plan['engineering_rows']]==[647,1837,4972,1441,4432,3497,1483,229]
    assert len({r['problem_sha256'] for r in plan['engineering_rows']})==8
    for r in plan['engineering_rows']:assert phash(r['problem'])==r['problem_sha256']
    assert plan['runtime']==dict(dtype='bfloat16',max_tokens=16000,max_model_len=32768,max_num_seqs=32,
        max_num_batched_tokens=32768,gpu_memory_utilization=.90,async_scheduling=False,chunked_prefill=False,
        seed=42,temperature=.7,top_p=.95)
    a=plan['assets'];p=a['rebalance_parameters']
    assert plan['penalty']['lower_bound']==min(p['low_val_1'],p['low_val_2'])
    assert 0<plan['penalty']['constant_scale']<1
    audit=ROOT/plan['penalty']['calibration_audit_path']
    assert sha(audit)==plan['penalty']['calibration_audit_sha256']
    frozen=read(audit)['results']['1.5B']
    assert plan['penalty']['constant_scale']==frozen['constant_scale']
    assert a['fit']['sha256']==frozen['fit_sha256']
    assert plan['arm_seconds']=={'engineering':120} and plan['process_seconds']==1200


def adapter_options(plan,name):
    p=plan['penalty']
    if name in ('Roff_scaled','Rshadow_scaled','RCscaled'):
        return dict(penalty_mode='coefficient_scaled',lower_bound=p['lower_bound'])
    if name=='RCconstant':
        return dict(penalty_mode='calibration_constant',lower_bound=p['lower_bound'],constant_scale=p['constant_scale'])
    if name=='RC14explicit':return dict(penalty_mode='fixed')
    return {}


def tokenizer_gate(tok,assets):
    # Runs before loading model; uses installed tokenizer as independent reference.
    path=Path(assets['model_path'])/'tokenizer.json'
    pieces,boundaries=vocabulary(path)
    actual={i for text,i in tok.get_vocab().items() if 'ĊĊ' in text}
    assert actual==boundaries
    for i in pieces:
        assert token_flags(tok.decode([i]),i in actual)==token_flags(pieces[i],i in boundaries),i
    for i,text in TRIGGERS.items():assert tok.decode([i])==text and tok.encode(text,add_special_tokens=False)==[i]
    return dict(passed=True,tokenizer_sha256=sha(path),checked_tokens=len(pieces),boundary_ids=sorted(boundaries))


def check_result(name,results):
    key=lambda r:[(x['token_ids'],x['R_history_sha256']) for x in r['records']]
    if name in ('Roff_scaled','Rshadow_scaled'):assert key(results[name])==key(results['R'])
    if name=='RC14explicit':assert key(results[name])==key(results['RC14default'])
    if name in ('RCscaled','RCconstant'):
        assert sum(x['changed'] for x in results[name]['events'].values())>0
        assert any(x['token_ids']!=y['token_ids'] for x,y in zip(results[name]['records'],results['RC14default']['records'])), 'No trajectory-change coverage; stop, not an efficacy verdict'
