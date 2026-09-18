"""Validate explicit episode evidence and feature availability; export no vector."""
import json
import numpy as np
import sympy as s
from prepare_length_vector import HERE,ROOT,read,save,sha

EPISODES=[
 dict(parent=3307,status='valid_repair',error=[73706,73719],repair=[73740,73756],check=[73764,73788],
      note='Recompute squared norm correctly rather than repeating the wrong common denominator. Subsequent unit-norm and bisector checks verify different constraints; not a clean redundant counterpart. Earlier b-is-unit claim73661 and later uniqueness rationale73797 remain wrong.'),
 dict(parent=4543,status='partial_repair_with_recurrence',error=[82269,82272],repair=[82273,82276],check=[82277,82294],
      note='Finite-error approximation clarified, then ambiguity recurs. Cannot label whole later interval redundant or all issues resolved.'),
 dict(parent=3063,status='valid_rechecks_without_prior_error',error=None,repair=None,check=[39224,39266],
      note='Earlier dot product already zero. Later39267-39272 checks nonzero vector, a distinct requirement; do not suppress all late verification.'),
 dict(parent=3104,status='false_repair_correct_final_scalar',error=[63385,63395],repair=[63507,63520],check=None,
      note='Alleged repair changes c cross a from correct minus j to incorrect plus j. Along with wrong signs in other terms this yields correct scalar2. Not positive repair supervision.'),
 dict(parent=4139,status='unresolved_angle_convention',error=[4729,4744],repair=None,check=[4835,4900],
      note='Reflex closing sector is repeatedly substituted for intended smaller AOD; no justified resolution in inspected trajectory.'),
 dict(parent=5977,status='correct_restoration_then_regression',error=[14146,14161],repair=[14183,14184],check=[14185,14732],
      note='Restores known16 then immediately reasserts erroneous multiplier; later repeated loop never resolves multiplier discrepancy within saved steps.')]

def main():
    out=HERE/'repair_episodes_20260918_cpu'
    sp=ROOT/'.codex_work/question_balanced_20260911/original_selected_layer/steps.json'
    steps=read(sp);fp=sp.parent/'layer_21.npy'
    assert sha(fp)=='66fdc33b5ca4d40d76243c1ff062dabdecbe542bcc02580ee0cb1ae2ea31ba66'
    features=np.load(fp,mmap_mode='r');assert features.shape==(84008,1536)
    result=[]
    for spec in EPISODES:
        data=read(out/(str(spec['parent'])+'.json'));by={r['step_index']:r for r in data['steps']};q=data['identity']['question']
        evidence={};boundaries=[]
        for role in ('error','repair','check'):
            span=spec[role]
            if span is None:continue
            lo,hi=span;assert lo<=hi and lo in by and hi in by
            evidence[role]=[by[i] for i in range(lo,hi+1)]
            for phase,i in [('before_span',lo),('after_span',hi+1)]:
                if i not in by:continue
                assert steps[i]['question']==q and steps[i]['start']==by[i]['start']
                assert np.isfinite(features[i]).all()
                boundaries.append(dict(role=role,phase=phase,feature_row=i,token_index=steps[i]['start'],
                    content_available_through=steps[i]['start'],includes_current_first_token=True,
                    label_available_online=False,reason='Semantic episode label requires annotation/verifier; stored feature alone does not supply it.'))
        result.append(dict(**spec,evidence=evidence,feature_boundaries=boundaries,
            problem_sha256=data['identity']['problem_sha256'],source_sha256=sha(out/(str(spec['parent'])+'.json'))))
    # A concrete failure of final-answer-derived process labels.
    i=s.Matrix([1,0,0]);j=s.Matrix([0,1,0]);kb=s.Matrix([0,0,1]);c=-i-kb
    assert kb.cross(i)==j and c.cross(i)==-j and kb.cross(c)==-j
    assert 2*kb.cross(i)+kb.cross(c)+c.cross(i)==s.zeros(3,1)
    # All signs inverted also sum to zero at2: correct result is not valid repair.
    assert 2*(-j)+j+j==s.zeros(3,1)
    save(out/'decision.json',dict(episodes=result,checks='Feature identities/finite values and cross-product counterexample passed',
        clean_matched_pairs=0,fit_performed=False,gpu_ready=False,
        decision='Close cue-derived semantic protection/vector candidate at present evidence level; do not keep growing isolated examples or GPU-test an untrained oracle.',
        next_requirement='A reusable process verifier or independently audited training episode labels with parent-disjoint validation. No current implementation supplies this. Preserve L27.',
        interface=dict(offline_record=['parent_hash','claim_id','evidence_step_ids','status_before','status_after','repair_validity','regression_step_ids','uncertain'],
            possible_states=['unresolved','locally_resolved','regressed','unknown'],
            transitions='Only verified proposition-level evidence changes state. A phrase, repeated answer, or final correctness never changes state alone.',
            deployment='No shared controller change. Future predictor must consume prefix-only first-token features, reset per request, abstain to original L27 on unknown. Train/evaluate by parent, never split sibling steps across sets.',
            feature_warning='First token at start is stored, not completed-step hidden state. After-span feature must use next step row and predicts subsequent behavior; it cannot justify an intervention before the repair.'),
        limits='Six previously exposed training parents, adaptive full-context follow-up; zero clean pairs is restricted to this audit, not a population impossibility. Single-reviewer role judgments; no causal deletion-safety.'))
    print(json.dumps(dict(parents=len(result),clean_pairs=0,gpu_ready=False,decision_sha256=sha(out/'decision.json'))))

if __name__=='__main__':main()
