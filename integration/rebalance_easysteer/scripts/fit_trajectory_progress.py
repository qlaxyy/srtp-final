"""Local CPU screening of a fixed within-trajectory progress direction."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

import numpy as np

from mechanism_candidates import ROOT, BASE, read, save, sha, require, cosine
from mechanism_candidates import read_vector, write_vector


def difference(early, late):
    return np.asarray(late,dtype=np.float64).mean(0)-np.asarray(early,dtype=np.float64).mean(0)


def pair_auc(early, late):
    d=np.asarray(late)[:,None]-np.asarray(early)[None,:]
    return float(((d>0)+.5*(d==0)).mean())


def cpu_checks():
    early=np.array([[0.,1.],[2.,3.]])
    late=np.array([[4.,5.],[6.,7.]])
    np.testing.assert_array_equal(difference(early,late),[4.,4.])
    np.testing.assert_array_equal(difference(early+100,late+100),difference(early,late))
    np.testing.assert_array_equal(difference(np.repeat(early,3,axis=0),late),difference(early,late))
    assert pair_auc([0.,1.],[1.,2.])==.875 and pair_auc([1.],[1.])==.5
    return 4


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();require(not out.exists(),'Immutable result exists')
    plan_path=ROOT/BASE/'configs/within_trajectory_progress_20260912.json'
    plan=read(plan_path);inputs=plan['inputs'];backup=ROOT/inputs['backup']
    started=time.perf_counter();checks=cpu_checks()
    require(sha(backup/'layer_21.npy')==inputs['hidden_sha256'],'Hidden data changed')
    require(sha(backup/'steps.json')==inputs['steps_sha256'],'Step data changed')
    steps=read(backup/'steps.json');selected=read(backup/'selected_layer.json')
    train=set(selected['training_questions']);val=set(selected['validation_questions'])
    require(len(train)==400 and len(val)==100 and not train&val,'Wrong original group split')
    hidden=np.load(backup/'layer_21.npy',mmap_mode='r')
    require(hidden.shape==(84008,1536) and len(steps)==len(hidden),'Wrong selected-state matrix')
    by_question={q:[] for q in range(500)}
    for i,step in enumerate(steps):by_question[step['question']].append(i)
    rejected=Counter();groups=[];source_hash=hashlib.sha256();count=0
    with tarfile.open(ROOT/inputs['saved_answers']) as archive:
        with archive.extractfile(inputs['generations_member']) as stream:
            for q,line in enumerate(stream):
                source_hash.update(line);answer=json.loads(line);count+=1
                ids=answer['token_ids']
                if answer['finish_reason']!='stop':rejected['not_finished_stop']+=1;continue
                if 151649 not in ids:rejected['missing_think_end']+=1;continue
                length=ids.index(151649)
                early=[i for i in by_question[q] if 0<steps[i]['start']<.25*length]
                late=[i for i in by_question[q] if .75*length<=steps[i]['start']<length]
                if len(early)<2 or len(late)<2:rejected['insufficient_quarter_boundaries']+=1;continue
                # Do not expose or use the answer's mathematical correctness.
                groups.append(dict(question=q,train_index=answer['train_index'],thinking_length=length,
                                   early=early,late=late,split='train' if q in train else 'development'))
    require(count==500 and source_hash.hexdigest()==inputs['generations_sha256'],'Saved answers changed')
    diffs=np.array([difference(hidden[g['early']],hidden[g['late']]) for g in groups])
    train_mask=np.array([g['split']=='train' for g in groups]);val_mask=~train_mask
    require(train_mask.any() and val_mask.any(),'Empty development groups')
    train_direction=diffs[train_mask].mean(0);val_direction=diffs[val_mask].mean(0)
    full_direction=diffs.mean(0)
    original=read_vector(backup/'auto_vector.pt',backup/'auto_vector.pt').astype(np.float64)
    require(np.linalg.norm(full_direction)>0,'Degenerate progress direction')
    candidate=-full_direction/np.linalg.norm(full_direction)*np.linalg.norm(original)
    axes=dict(progress=train_direction/np.linalg.norm(train_direction),
              original_compression=-original/np.linalg.norm(original))
    readouts={name:[] for name in axes}
    for g in groups:
        if g['split']!='development':continue
        early=np.asarray(hidden[g['early']],dtype=np.float64);late=np.asarray(hidden[g['late']],dtype=np.float64)
        for name,axis in axes.items():
            pre=early@axis/np.linalg.norm(early,axis=1);post=late@axis/np.linalg.norm(late,axis=1)
            readouts[name].append(dict(question=g['question'],pair_auc=pair_auc(pre,post),
                                       early_score=float(pre.mean()),late_score=float(post.mean())))
    stats={name:dict(question_mean_pair_auc=float(np.mean([r['pair_auc'] for r in values])),
                    late_higher_question_fraction=float(np.mean([r['late_score']>r['early_score'] for r in values])))
           for name,values in readouts.items()}
    gate=plan['cpu_gate'];stability=cosine(train_direction,val_direction);alignment=cosine(candidate,original)
    conditions=dict(training_support=int(train_mask.sum())>=gate['minimum_training_questions'],
        development_support=int(val_mask.sum())>=gate['minimum_development_questions'],
        direction_stability=stability>=gate['minimum_train_development_direction_cosine'],
        stage_readout=stats['progress']['question_mean_pair_auc']>=gate['minimum_development_question_mean_pair_auc'],
        adds_progress_signal=stats['progress']['question_mean_pair_auc']-stats['original_compression']['question_mean_pair_auc']>=gate['minimum_development_pair_auc_gain_over_original_compression_axis'],
        question_consistency=stats['progress']['late_higher_question_fraction']>=gate['minimum_development_late_higher_question_fraction'],
        distinct_direction=abs(alignment)<=gate['maximum_absolute_candidate_original_cosine'])
    out.mkdir(parents=True);assets=out/'within_trajectory_progress';assets.mkdir()
    write_vector(assets/'auto_vector.pt',candidate.astype(np.float32),backup/'auto_vector.pt')
    fit=read(backup/'fit.json');fit.update(version='within-trajectory-progress-v1',
        vector_sha256=sha(assets/'auto_vector.pt'),vector_norm=float(np.linalg.norm(candidate.astype(np.float32))),
        method='Question-equal first-versus-last temporal-quarter contrast; raw direction scaled to original norm; all original dynamic parameters unchanged.',
        calibration_questions=len(groups),training_questions=int(train_mask.sum()),development_questions=int(val_mask.sum()),
        positives='early-stage mean per question',negatives='late-stage mean per question',
        selection=dict(hidden_state_index=21,decoder_output_layer=20,layer_search=False))
    save(assets/'fit.json',fit)
    result=dict(status='CPU_screen_completed',plan_sha256=sha(plan_path,source=True),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        script_sha256=sha(Path(__file__),source=True),source_answer_sha256=source_hash.hexdigest(),
        eligible_questions=len(groups),rejected=dict(rejected),training_questions=int(train_mask.sum()),development_questions=int(val_mask.sum()),
        original_norm=float(np.linalg.norm(original)),unscaled_progress_norm=float(np.linalg.norm(full_direction)),
        candidate_original_cosine=alignment,train_development_direction_cosine=stability,
        readout_statistics=stats,readouts=readouts,groups=groups,conditions=conditions,
        passes_cpu_gate=all(conditions.values()),cpu_math_checks=checks,
        vector_sha256=sha(assets/'auto_vector.pt'),fit_sha256=sha(assets/'fit.json'),
        seconds=time.perf_counter()-started,model_calls=0,new_answers=0,
        decision='Prepare independent generation screen' if all(conditions.values()) else 'Stop candidate; no temporal-window or threshold tuning',
        limitations=plan['limitations'])
    save(out/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['groups','readouts','limitations']},indent=2))


if __name__=='__main__':main()
