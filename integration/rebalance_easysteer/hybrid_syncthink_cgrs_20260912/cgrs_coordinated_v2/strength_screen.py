"""One-seed, four-arm training screen; no automatic confirmation."""
from engineering import ROOT,read,sha
from prepare_screen import phash

ARMS=['R','RC14','RCconstant','RCscaled']
RUNTIME=dict(dtype='bfloat16',max_tokens=16000,max_model_len=32768,max_num_seqs=128,
    max_num_batched_tokens=32768,gpu_memory_utilization=.90,async_scheduling=False,
    chunked_prefill=False,seed=42,temperature=.7,top_p=.95)


def validate(plan):
    assert plan['candidate_kind']=='strength_screen' and plan['phase']=='screen_only'
    assert plan['model_family']=='1p5b' and plan['arms']==ARMS and plan['primary_candidate']=='RCscaled'
    assert plan['runtime']==RUNTIME and plan['arm_seconds']=={'screen':450} and plan['process_seconds']==2100
    rows=plan['rows'];assert len(rows)==100 and len({r['problem_sha256'] for r in rows})==100
    for i,r in enumerate(rows):
        assert r['dataset']=='math_train' and r['split']=='train' and r['dataset_index']==i
        assert phash(r['problem'])==r['problem_sha256']
    assert rows==read(ROOT/plan['rows_path']) and sha(ROOT/plan['rows_path'])==plan['rows_sha256']
    p=plan['penalty'];audit=ROOT/p['calibration_audit_path'];assert sha(audit)==p['calibration_audit_sha256']
    f=read(audit)['results']['1.5B'];assert p['constant_scale']==f['constant_scale'] and p['lower_bound']==f['lower_bound']
    assert plan['assets']['fit']['sha256']==f['fit_sha256']


def validate_authorization(plan,receipt):
    validate(plan)
    assert receipt['scope']=='strength_1p5b_mathtrain100x4_seed42_only'
    assert receipt['authorized_phases']==['screen'] and receipt['data_reconciled'] is True
    assert receipt['unregistered_claims_checked'] is True
    for side in ('local','remote'):
        r=receipt[side+'_reconciliation'];audit=read(r['path']);assert sha(r['path'])==r['sha256']
        assert audit['passed'] and not audit['hits'] and not audit['errors']
        assert audit['rows_sha256']==plan['rows_sha256']


def decision(groups):
    c=groups['RCscaled']
    assert all(groups[a]['n']==100 for a in ARMS)
    compression=all(c[k]<groups[a][k] for a in ARMS[:-1] for k in ('mean_total_tokens','mean_thinking_tokens'))
    accuracy=all(c['correct']-groups[a]['correct']>=-2 for a in ('R','RC14'))
    caps=all(c['capped']<=groups[a]['capped'] for a in ('R','RC14'))
    return dict(compression_vs_all_three=compression,accuracy_observed_margin_pass=accuracy,
                caps_pass=caps,retain_for_separate_confirmation=compression and accuracy and caps,
                statistically_confirmed=False)
