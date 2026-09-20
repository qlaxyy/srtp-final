"""User-requested fixed MATH-500 evaluation; no training screen or retuning."""
from engineering import ROOT,read,sha
from prepare_screen import phash
from strength_screen import RUNTIME

def validate(plan,receipt):
    assert plan['candidate_kind']=='strength_full' and plan['phase']=='full_test'
    assert plan['model_family']=='1p5b' and plan['arms']==['RCscaled']
    assert plan['primary_candidate']=='RCscaled' and list(plan['datasets'])==['math_test']
    assert receipt['scope']=='strength_1p5b_math500_scaled_only'
    assert receipt['authorized_phases']==['full'] and receipt['exposed_test_reuse_acknowledged'] is True
    assert plan['runtime']==RUNTIME
    rows=plan['datasets']['math_test']['rows']
    assert len(rows)==500 and len({r['problem_sha256'] for r in rows})==500
    for i,r in enumerate(rows):
        assert r['dataset_index']==i and r['split']=='test'
        assert phash(r['problem'])==r['problem_sha256']
    p=plan['penalty'];audit=ROOT/p['calibration_audit_path']
    assert sha(audit)==p['calibration_audit_sha256']
    f=read(audit)['results']['1.5B']
    assert p['lower_bound']==f['lower_bound'] and p['constant_scale']==f['constant_scale']
    assert plan['assets']['fit']['sha256']==f['fit_sha256']
    assert plan['arm_seconds']=={'full':1500} and plan['process_seconds']==1800
