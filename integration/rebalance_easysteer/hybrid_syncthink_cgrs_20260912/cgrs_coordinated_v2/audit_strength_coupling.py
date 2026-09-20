"""Offline feasibility only. No production controller or sampler changes."""
import ast
import json
import math
from pathlib import Path
import numpy as np
from engineering import HERE, ROOT, save, sha


def penalty(coefficient, lower_bound, mode='scaled'):
    if not math.isfinite(lower_bound) or lower_bound >= 0:
        raise ValueError('Negative finite model-specific bound required')
    if mode not in ('scaled', 'fixed'):
        raise ValueError(mode)
    if not math.isfinite(coefficient) or coefficient >= 0:
        return 0.0
    factor = 1.0 if mode == 'fixed' else min(1.0, coefficient/lower_bound)
    return math.log(2)*factor


def coefficients(c, v, p, source):
    # Scalar curve solver extracted from pinned source, not re-fitted.
    tree = ast.parse(source.read_text(encoding='utf-8'))
    solver = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_solve_k_for_tau']
    assert len(solver)==1
    ns={'math':math}
    exec(compile(ast.Module(body=solver,type_ignores=[]),str(source),'exec'),ns)
    q25,q75=sorted((p['q25c'],p['q75c']));v25,v75=sorted((p['q25v'],p['q75v']))
    k=ns['_solve_k_for_tau'](q25,q75,p['low_val_1'],0.0,p['curve_tau'])
    mid=(q25+q75)/2;intercept=p['low_val_1']/2
    slope=-p['low_val_1']/(2*max(math.tanh(k*max(1e-9,(q75-q25)/2)),1e-12))
    def baseline(x):return intercept+slope*np.tanh(k*(np.clip(x,0,1)-mid))
    def sig(x):return 1/(1+np.exp(-np.clip(x,-60,60)))
    dc=max(1e-12,q75-q25);dv=max(1e-12,v75-v25)
    low=np.minimum(1,sig((q25-c)/dc*1200)*sig((v-v75)/dv*1200)/0.25)
    high=np.minimum(1,sig((c-q75)/dc*1200)*sig((v25-v)/dv*1200)/(sig((1-q75)/dc*12)*0.5))
    return np.clip(baseline(c)+(p['low_val_2']-baseline(q25))*low+(p['high_val_2']-baseline(1))*high,
                   min(p['low_val_1'],p['low_val_2']),max(0.01,p['high_val_2']))


def main():
    out=HERE/'strength_coupling_cpu_20260916';out.mkdir(exist_ok=False)
    root=Path('E:/srtp/srtp-final/.codex_work')
    source=ROOT/'sources/EasySteer/vllm-steer/vllm/steer_vectors/rebalance.py'
    sets={
        '1.5B':(root/'auto_code_v2_500_20260908/fit.json',root/'question_balanced_20260911/original_selected_layer/steps.json'),
        '7B':(root/'qwen7b_validation/auto_code_v2_qwen7b_20260908/fit.json',root/'qwen7b_validation/auto_code_v2_qwen7b_20260908/steps.json')}
    results={};inputs={str(source):sha(source)}
    for model,(fit,steps) in sets.items():
        for path in (fit,steps):inputs[str(path)]=sha(path)
        f=json.loads(fit.read_text());rows=json.loads(steps.read_text());p=f['parameters']
        c=np.array([r['confidence'] for r in rows]);v=np.array([r['variance'] for r in rows])
        assert np.isfinite(c).all() and np.isfinite(v).all()
        # Sanity agreement with archived calibration quartiles; not token-alignment proof.
        assert np.allclose(np.quantile(c,[.25,.75]),[p['q25c'],p['q75c']],atol=1e-8,rtol=0)
        assert np.allclose(np.quantile(v,[.25,.75]),[p['q25v'],p['q75v']],atol=1e-8,rtol=0)
        a=coefficients(c,v,p,source);negative=a<0;lo=min(p['low_val_1'],p['low_val_2'])
        scales=a[negative]/lo
        results[model]=dict(steps=len(rows),questions=len({r['question'] for r in rows}),
            negative_steps=int(negative.sum()),lower_bound=lo,
            scale_quantiles=dict(zip(['min','q25','median','q75','max'],map(float,np.quantile(scales,[0,.25,.5,.75,1])))),
            mean_scale=float(scales.mean()),fraction_negative_below_half_strength=float(np.mean(scales<.5)),
            bound_saturation_fraction=float(np.mean(scales==1)),
            quartiles_match_archived_fit=True)
    save(out/'audit.json',dict(results=results,input_sha256=inputs,source_sha256=sha(Path(__file__),True),
        scope='Existing calibration-step statistics, not online RC14 eligibility or candidate generation.',
        limitations=['NumPy float64 analytical reconstruction; no Torch FP32/GPU parity claimed.',
                     'Calibration rows do not reconstruct clean lexical opening, accepted-token clock, or missing empty-step history.',
                     'Coefficient magnitude is not confidence in answer correctness or intervention benefit.',
                     'Mean-scaled constant comparator must be frozen from actual eligible calibration openings before an efficacy run.'],gpu_ready=False))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
