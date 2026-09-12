"""Screen fixed relative-dose geometry using previously saved scalar arrays."""
import argparse
import time
from pathlib import Path
import numpy as np
from mechanism_candidates import ROOT,BASE,read,save,sha,require,read_vector


def quantiles(values):
    return np.quantile(values,[0,.1,.25,.5,.75,.9,.99,1]).tolist()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();require(not a.output.exists(),'Immutable output exists')
    started=time.perf_counter()
    plan_path=ROOT/BASE/'configs/relative_displacement_20260912.json';plan=read(plan_path)
    base=ROOT/'.codex_work/overnight_research_20260912/radial_restoration_cpu'
    original=read(base/'summary.json');arrays=base/'boundaries.npz'
    require(sha(arrays)==original['boundaries_sha256'],'Geometry archive changed')
    backup=ROOT/'.codex_work/question_balanced_20260911/original_selected_layer'
    require(sha(backup/'selected_layer.json')==original['split_sha256'],'Question groups changed')
    selection=read(backup/'selected_layer.json');data=np.load(arrays)
    norm=data['state_norm'];c=data['coefficient'];cos=data['cosine_with_direction'];q=data['question']
    require(len(norm)==83508 and np.all(norm>0),'Invalid support')
    masks={k:np.isin(q,selection[k]) for k in ['training_questions','validation_questions']}
    require(np.all(masks['training_questions'] ^ masks['validation_questions']),'Question groups overlap')
    reference=float(np.median(norm[masks['training_questions']]))
    development_reference=float(np.median(norm[masks['validation_questions']]))
    vector=read_vector(backup/'auto_vector.pt',backup/'auto_vector.pt').astype(np.float64)
    d=float(np.linalg.norm(vector));scaled_c=c*norm/reference
    ratio=norm/reference
    before=abs(c)*d/norm;after=abs(scaled_c)*d/norm
    identity_error=float(np.max(abs(after-abs(c)*d/reference)))
    require(identity_error<1e-12 and np.array_equal(np.sign(c),np.sign(scaled_c)),'Relative-dose or sign identity failed')
    signed=scaled_c*d/norm
    new_radius_ratio=np.sqrt(1+2*signed*cos+signed*signed)
    new_angle=np.arccos(np.clip((1+signed*cos)/new_radius_ratio,-1,1))
    require(np.isfinite(new_radius_ratio).all() and np.isfinite(new_angle).all(),'Degenerate geometry')
    gate=plan['cpu_gate'];lo,hi=gate['scale_interval'];affected=(ratio<lo)|(ratio>hi)
    fit=read(backup/'fit.json')['parameters']
    low=min(fit['low_val_1'],fit['low_val_2']);high=max(.01,fit['high_val_2'])
    summary={}
    for name,mask in {'all':np.ones(len(norm),dtype=bool),**masks}.items():
        summary[name]=dict(count=int(mask.sum()),questions=len(np.unique(q[mask])),
            affected_fraction=float(affected[mask].mean()),affected_questions=len(np.unique(q[mask&affected])),
            state_norm_quantiles=quantiles(norm[mask]),scale_quantiles=quantiles(ratio[mask]),
            effective_coefficient_quantiles=quantiles(scaled_c[mask]),
            fraction_effective_coefficient_outside_original_bounds=float(((scaled_c<low)|(scaled_c>high))[mask].mean()),
            before_relative_displacement_quantiles=quantiles(before[mask]),after_relative_displacement_quantiles=quantiles(after[mask]),
            before_angle_quantiles=quantiles(data['angle_radians'][mask]),after_angle_quantiles=quantiles(new_angle[mask]),
            candidate_radius_ratio_quantiles=quantiles(new_radius_ratio[mask]))
    stable_lo,stable_hi=gate['development_to_training_median_norm_interval']
    passed=(summary['all']['affected_fraction']>=gate['minimum_fraction_scale_outside_reciprocal_interval']
        and summary['all']['affected_questions']>=gate['minimum_affected_questions']
        and all(summary[k]['affected_fraction']>=gate['minimum_affected_fraction_in_each_group'] for k in masks)
        and stable_lo<=development_reference/reference<=stable_hi)
    result=dict(status='CPU_opportunity_only_no_generation',plan_sha256=sha(plan_path,source=True),
        script_sha256=sha(Path(__file__),source=True),input_geometry_sha256=sha(arrays),
        training_reference_norm=reference,development_median_norm=development_reference,
        development_training_reference_ratio=development_reference/reference,
        final_calibration_reference_norm=float(np.median(norm)) if passed else None,
        maximum_relative_displacement=abs(low)*d/reference,
        constant_relative_displacement_identity_error=identity_error,summary=summary,
        passes_fixed_cpu_gate=bool(passed),decision='runtime_preparation_permitted_after_current_radial_batch' if passed else 'stop_no_GPU_no_retuning',
        cpu_seconds=time.perf_counter()-started,model_loads=0,new_answers=0,GPU_calls=0,
        limitations=plan['limitations'])
    save(a.output,result)
    print({k:v for k,v in result.items() if k not in ['summary','limitations']})
    print({k:{f:v[f] for f in ['affected_fraction','affected_questions','fraction_effective_coefficient_outside_original_bounds']} for k,v in summary.items()})


if __name__=='__main__':main()
