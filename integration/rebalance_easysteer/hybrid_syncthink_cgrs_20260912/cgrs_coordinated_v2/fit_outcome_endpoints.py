"""Fit distinct endpoint semantics and matching controllers without test feedback."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
import torch

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    complete=read(a.features/'complete.json');assert complete['passed'] and complete['phase']=='full'
    spec=importlib.util.spec_from_file_location('outcome_native_rebalance',a.runtime);rt=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name]=rt;spec.loader.exec_module(rt)
    rows=read(a.features/'manifest.json');data={}
    for r in rows:
        assert sha(a.features/r['file'])==r['sha256']
        x=np.load(a.features/r['file'])['features'];c=np.array([s['confidence'] for s in r['steps']]);v=np.array([s['variance'] for s in r['steps']]);lex=np.array([s['lexical_hit'] for s in r['steps']],bool)
        data[r['question'],r['side']]=(r,x,c,v,lex)
    a.output.mkdir(exist_ok=False);reports={}
    for arm in ['EFFICIENT','UNDER_REFIT']:
        dest=a.output/arm;dest.mkdir()
        pool=[z for z in data.values() if (z[0]['kind']=='CC' if arm=='EFFICIENT' else (z[0]['kind']=='CC' and z[0]['side']=='long') or (z[0]['kind']=='CW' and z[0]['side']=='short'))]
        cl,ch=np.quantile(np.concatenate([z[2] for z in pool]),[.25,.75]);vl,vh=np.quantile(np.concatenate([z[3] for z in pool]),[.25,.75]);assert 0<cl<ch<1 and 0<=vl<vh
        over=[];other=[];op=[];up=[]
        for q in sorted({k[0] for k in data}):
            lr,lx,lc,lv,ll=data[q,'long'];sr,sx,sc,sv,sl=data[q,'short']
            om=ll|(lc<cl);um=(~sl)&(sc>ch)
            if arm=='EFFICIENT':
                if lr['kind']=='CC' and om.any() and um.any():
                    over.append(lx[om].mean(0,dtype=np.float64));other.append(sx[um].mean(0,dtype=np.float64));op.append(q);up.append(q)
            else:
                if lr['kind']=='CC' and om.any():over.append(lx[om].astype(np.float64));op.append(q)
                if sr['kind']=='CW' and um.any():other.append(sx[um].astype(np.float64));up.append(q)
        supported=(len(op)>=30 if arm=='EFFICIENT' else len(op)>=30 and len(up)>=3 and sum(len(x) for x in other)>=20)
        report=dict(supported=bool(supported),over_parents=op,other_parents=up,quantiles=dict(confidence=[float(cl),float(ch)],variance=[float(vl),float(vh)]))
        if not supported:save(dest/'report.json',report);reports[arm]=report;continue
        xo=np.array(over) if arm=='EFFICIENT' else np.concatenate(over);xu=np.array(other) if arm=='EFFICIENT' else np.concatenate(other)
        mo,mu=xo.mean(0),xu.mean(0);d=mo-mu;assert np.isfinite(d).all() and np.linalg.norm(d)>0
        w=d/(xo.var(0)+xu.var(0)+1e-12);proj=float(w@d);threshold=float(.5*w@(mo+mu));assert proj>0
        moderate=float((w@mo-threshold)/proj);aggressive=float(np.max((xo@w-threshold)/proj));assert abs(moderate-.5)<1e-6
        low2=-aggressive if arm=='UNDER_REFIT' else -.5
        tau=min(.01,.25*(1-ch)/(ch-cl));rt.validate_curve_targets(float(cl),float(ch),-.5,float(tau))
        params=dict(q25c=float(cl),q75c=float(ch),q25v=float(vl),q75v=float(vh),low_val_1=-.5,low_val_2=float(low2),high_val_2=.1,initial_coef=0. if arm=='EFFICIENT' else -1.,curve_tau=float(tau))
        control=rt.ReBalanceParams(boundary_token_ids=(1,),think_start_token_id=2,think_end_token_id=3,**params)
        c,v=torch.meshgrid(torch.linspace(0,1,501),torch.linspace(0,.25,251),indexing='ij')
        surface=-((c-cl)/(ch-cl)*-1+1).clamp(0,1) if arm=='EFFICIENT' else rt.compute_rebalance_coefficient(c,v,control)
        assert torch.isfinite(surface).all()
        vector=torch.from_numpy(d.astype(np.float32));torch.save(vector,dest/'auto_vector.pt')
        report.update(raw_norm=float(np.linalg.norm(d)),rows_over=len(xo),rows_other=len(xu),curve_min=float(surface.min()),curve_max=float(surface.max()),midpoint_residual=float(np.max(np.abs(mo-.5*d-(mo+mu)/2))),vector_sha256=sha(dest/'auto_vector.pt'))
        if arm=='EFFICIENT':
            assert np.max(np.abs(mo-d-mu))<1e-10
            report['endpoint_mapping']='alpha=-1 maps long prototype to efficient prototype, not an assertion about every step'
        save(dest/'fit.json',dict(decoder_output_layer=20,hidden_state_index=21,parameters=params,vector_sha256=report['vector_sha256'],method=arm,controller_override='efficient_clip' if arm=='EFFICIENT' else None))
        save(dest/'report.json',report);reports[arm]=report
    save(a.output/'report.json',dict(arms=reports,feature_complete_sha256=sha(a.features/'complete.json'),runtime_sha256=sha(a.runtime),fit_script_sha256=sha(Path(__file__))))
    print(json.dumps({k:{n:v[n] for n in ['supported','raw_norm','rows_over','rows_other'] if n in v} for k,v in reports.items()}))

if __name__=='__main__':main()
