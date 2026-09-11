"""Fixed CPU label and state-feedback diagnostics; never claims generation gains."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from mechanism_candidates import ROOT, BASE, read, save, sha, require, moments, cosine, read_vector, write_vector
from fit_control_points import normalize_like
from audit_control_alignment import load_controller, tensor


def clip_negative(coefficient, score, negative_center, response):
    require(np.isfinite(response) and response > 0, 'Direction does not lower readout')
    room = np.maximum(0, (np.asarray(score)-negative_center)/response)
    return np.where(coefficient < 0, np.maximum(coefficient, -room), coefficient)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); out = a.output.resolve()
    require(not out.exists(), 'Immutable audit output already exists')
    started = time.perf_counter()
    plan_path = ROOT/BASE/'configs/mechanism_followups_20260912.json'
    plan = read(plan_path); plans = {x['id']:x for x in plan['candidates']}
    root_plan = read(ROOT/BASE/'configs/overnight_research_20260912.json')['first_investigation']
    backup = ROOT/root_plan['inputs']['backup']
    for name,digest in root_plan['cpu_result']['inputs_sha256'].items():
        require(sha(backup/name) == digest, 'Original input changed: '+name)
    steps = read(backup/'steps.json'); selection = read(backup/'selected_layer.json')
    train, valid = [set(selection[k]) for k in ('training_questions','validation_questions')]
    require(len(train) == 400 and len(valid) == 100 and not train & valid, 'Wrong original groups')
    q = np.array([s['question'] for s in steps]); c = np.array([s['confidence'] for s in steps])
    variance = np.array([s['variance'] for s in steps]); lex = np.array([s['lexical_hit'] for s in steps])
    original_fit = read(backup/'fit.json'); params = original_fit['parameters']
    original = read_vector(backup/'auto_vector.pt',backup/'auto_vector.pt')
    norm = float(np.linalg.norm(original.astype(float)))
    over = ~lex & (c < params['q25c']); under = ~lex & (c > params['q75c'])
    masks = {'over':over,'under':under}
    for split, qs in [('train',train),('valid',valid)]:
        keep = np.isin(q,list(qs))
        for name,mask in [('over',over),('under',under)]:masks[split+'_'+name] = keep & mask
    stats = moments(np.load(backup/'layer_21.npy',mmap_mode='r'),masks)
    direction = stats['over']['mean'] - stats['under']['mean']
    d_train = stats['train_over']['mean'] - stats['train_under']['mean']
    d_valid = stats['valid_over']['mean'] - stats['valid_under']['mean']
    candidate = normalize_like(direction,norm)
    questions = {k:int(len(np.unique(q[m]))) for k,m in masks.items()}
    stability = cosine(d_train,d_valid)
    difference = float(np.linalg.norm(candidate.astype(float)-original)/norm)
    gate = plans['confidence_only_direction']['cpu_gate']
    passed = (min(questions.values()) >= gate['minimum_class_questions_in_each_original_group']
              and stability >= gate['minimum_train_validation_cosine']
              and difference >= gate['minimum_relative_normed_difference_from_original'])
    out.mkdir(parents=True); folder=out/'confidence_only_direction'; folder.mkdir()
    write_vector(folder/'auto_vector.pt',candidate,backup/'auto_vector.pt')
    fit = dict(original_fit,version='confidence-only-v1',vector_sha256=sha(folder/'auto_vector.pt'),
        vector_norm=float(np.linalg.norm(candidate.astype(float))),positives=int(over.sum()),negatives=int(under.sum()),
        method=plans['confidence_only_direction']['mechanism'],original_fit_sha256=sha(backup/'fit.json'))
    save(folder/'fit.json',fit)
    label_result = dict(status='ready_for_fixed_gpu_screen' if passed else 'stopped_cpu_gate',
        gate=gate,passes_cpu_gate=passed,class_steps={k:int(m.sum()) for k,m in masks.items()},class_questions=questions,
        train_validation_cosine=stability,original_cosine=cosine(direction,original),relative_normed_difference=difference,
        raw_norm=float(np.linalg.norm(direction)),saved_norm=fit['vector_norm'],
        vector_sha256=fit['vector_sha256'],fit_sha256=sha(folder/'fit.json'))

    # Feedback uses the actual pre-content control point, with matched support.
    replay = ROOT/'.codex_work/overnight_research_20260912/control_point_replay'
    prepared = ROOT/'.codex_work/overnight_research_20260912/control_point_prepared'
    ledger = read(replay/'ledger.json'); x_path=replay/'predecessor_layer21.npy'
    require(sha(x_path) == ledger['files_sha256'][x_path.name], 'Predecessor states changed')
    require(ledger['all_original_elements_exact'], 'Replay equivalence failed')
    require(sha(prepared/'matched_mask.npy') == read(prepared/'replay_plan.json')['matched_mask_sha256'], 'Support changed')
    support = np.load(prepared/'matched_mask.npy')
    over = support & (lex | (c < params['q25c'])); under = support & ~lex & (c > params['q75c'])
    masks={'over':over,'under':under}
    for split, qs in [('train',train),('valid',valid)]:
        keep=np.isin(q,list(qs))
        for name,mask in [('over',over),('under',under)]:masks[split+'_'+name]=keep & mask
    x=np.load(x_path,mmap_mode='r'); stats=moments(x,masks)
    def readout(prefix=''):
        pos,neg=stats[prefix+'over'],stats[prefix+'under']
        return (pos['mean']-neg['mean'])/(pos['variance']+neg['variance']+1e-12)
    w=readout(); response=float(w@original); center=float(w@stats['under']['mean'])
    scores=np.empty(len(x))
    for start in range(0,len(x),2048):scores[start:start+2048]=np.asarray(x[start:start+2048],dtype=float)@w
    # Each incoming delimiter has the PREVIOUS completed step's summaries.
    # Prompt predecessors are excluded. No next-step confidence enters control.
    controller=load_controller()['compute_rebalance_coefficient']
    outgoing=np.asarray(controller(tensor(c),tensor(variance),SimpleNamespace(**params)))
    indices=np.flatnonzero(support)
    require(np.array_equal(q[indices],q[indices-1]),'Incoming event crossed question')
    incoming=outgoing[indices-1]
    require(np.isfinite(scores).all() and np.isfinite(incoming).all(),'Nonfinite state/control')
    negative=incoming < -1e-6
    adjusted=clip_negative(incoming,scores[indices],center,response) if response > 0 else None
    changed=(adjusted-incoming > 1e-6) & negative if adjusted is not None else np.zeros(len(indices),dtype=bool)
    fraction=float(changed.sum()/negative.sum()) if negative.any() else 0.0
    stability=cosine(readout('train_'),readout('valid_'))
    questions={k:int(len(np.unique(q[m]))) for k,m in masks.items()}
    gate=plans['latent_feedback_clip']['cpu_gate']
    passed=(response>0 and min(questions.values()) >= gate['minimum_class_questions_in_each_original_group']
        and stability >= gate['minimum_train_validation_readout_cosine']
        and fraction >= gate['minimum_fraction_negative_events_changed'])
    folder=out/'latent_feedback_clip';folder.mkdir()
    np.save(folder/'readout.npy',w.astype(np.float32))
    feedback_result=dict(status='ready_for_runtime_implementation' if passed else 'stopped_cpu_gate',
        gate=gate,passes_cpu_gate=passed,class_questions=questions,readout_train_validation_cosine=stability,
        w_dot_original_d=response,negative_centroid_score=center,valid_incoming_events=len(indices),
        negative_incoming_events=int(negative.sum()),negative_events_changed=int(changed.sum()),fraction_negative_changed=fraction,
        fully_cancelled_negative_events=int(((adjusted==0)&negative).sum()) if adjusted is not None else None,
        incoming_mean_absolute_negative=float(-incoming[negative].mean()),
        adjusted_mean_absolute_negative=float(-adjusted[negative].mean()) if adjusted is not None else None,
        readout_sha256=sha(folder/'readout.npy'),chronology='Previous complete step controls current predecessor state; first prompt excluded')
    save(folder/'fit.json',feedback_result)
    summary=dict(status='completed_cpu_diagnostic',confidence_only_direction=label_result,latent_feedback_clip=feedback_result,
        seconds=time.perf_counter()-started,plan_sha256=sha(plan_path,source=True),source_sha256=sha(Path(__file__),source=True),
        limitation='All hidden data are unsteered saved calibration trajectories; 400/100 groups were already used for layer selection. Offline intervention opportunities and class readouts do not establish real generation savings or semantic safety.')
    save(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=True))


if __name__=='__main__':main()
