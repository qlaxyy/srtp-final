"""Record content judgments before revealing confidence and lexical identities.

This is a single-reviewer diagnostic, not independently validated training labels.
"""
import json
from prepare_length_vector import HERE, read, save, sha

NOTES = {
 '4160': [('V','valid','Combine post widths and gaps to obtain side length.'),('V','valid','Move from known side length to four-side perimeter; But introduces progress.'),('M','rounding_check','Decimal check restates known rational perimeter; distinct verification value uncertain.')],
 '7392': [('V','valid','Substitute computed powers into outer operation.'),('R','valid','Recompute 3^4=81 already established in quarter context.'),('M','partly_incorrect','Claim operator occurs only in exponents overlooks outer operator; no actual revised answer here.')],
 '3716': [('V','valid','Substitute distance and speed.'),('V','valid','Compute difference between established travel times.'),('R','valid','Same time subtraction and minutes conversion already in preceding context.')],
 '4174': [('V','valid','Set up next subproblem for smallest three-digit multiple of four.'),('V','valid','Add established operands.'),('R','valid','Repeat already computed 12+100=112.')],
 '871': [('V','valid','Square given identity as a valid intermediate plan, not necessarily shortest.'),('V','valid','Expand cube and regroup algebraically.'),('R','valid','Recap previously established derivation and result76.')],
 '7216': [('V','valid','Compute initial acid amount.'),('V','valid','Multiply concentration equation by positive final volume.'),('M','no_mathematical_claim','Intention to check again does not itself establish redundant proof or necessary correction.')],
 '482': [('R','valid_counterexample','Same counterexample to same proposed row formula recurs without resolution.')]*3,
 '2166': [('R','valid','Repeat already enumerated integer solutions without summing.'),('R','valid','Repeat complete divisor checking agenda already executed in context.'),('R','valid','Repeat already enumerated integer solutions without summing.')],
 '5169': [('R','partly_incorrect','Repeated discussion admits impossible hypotenuse4 with side5 and omits both-known-sides-as-legs case.'),('R','incomplete_proof','Repeated valid hypotenuse5 calculation and rejection of hypotenuse4; target alone omits both-legs case.'),('R','partly_incorrect','Same invalid hypotenuse discussion repeated.')],
 '2030': [('R','segment_incomplete','Identity product720 already established. Step boundary truncates factorial punctuation; do not treat as false 720=6 claim.'),('R','incorrect_premise','Repeated assertion identity terms are all equal is false: terms are1 through6.'),('R','unjustified_premise','Repeated product-change assertion supplies no adequate argument; distinct terms premise is not established.')],
 '552': [('R','valid','Repeated statement that digit order matters supplies no counting progress.'),('R','valid','Repeated description of requested guess count supplies no counting progress.'),('R','valid','Repeated statement that digit order matters supplies no counting progress.')],
 '2769': [('M','valid_calculations_incomplete_relevance','LCMs are valid, but parity alone does not address upper bound; necessity of extra cases uncertain.'),('M','valid_but_entailed','New numeric LCM case already follows from general even-intersection rule; value as verification uncertain.'),('M','subgoal_only','Sets up counting odd multiples of25; limited context cannot establish novelty or redundancy.')],
}

def main():
    out=HERE/'process_review_20260918_cpu'
    assert not (out/'annotations.json').exists(), 'Do not overwrite recorded review'
    views=read(out/'reading_view.json'); by={v['case_id']:v for v in views}
    labels={'V':'valid_advancement','R':'redundant_restatement','M':'mixed_or_uncertain'}
    rows=[]
    for v in views:
        parent,stage=v['case_id'].split('_')
        kind,validity,reason=NOTES[parent][['quarter','middle','late'].index(stage)]
        support=[dict(case_id=v['case_id'],field='before',text=v['before'])]
        if v['earlier_exact_match']:
            support.append(dict(case_id=v['case_id'],field='earlier_exact_match',**v['earlier_exact_match']))
        if v['case_id'] in ('7392_middle','871_late'):
            prior=parent+('_quarter' if parent=='7392' else '_middle')
            support.append(dict(case_id=prior,field='before_and_target',text=by[prior]['before']+by[prior]['target']))
        rows.append(dict(case_id=v['case_id'],label=labels[kind],local_validity=validity,reason=reason,target=v['target'],support=support))
    assert len(rows)==36 and len({r['case_id'] for r in rows})==36
    save(out/'annotations.json',dict(review_type='single assistant content review; confidence/lexical fields not consulted; strata not fully blinded',
        protocol_sha256=sha(out/'protocol.json'),reading_sha256=sha(out/'reading_view.json'),rows=rows,
        limitations=['Not independent human gold','No deletion-safety or causal claim','No direction fitting','250-token local context with earlier exact matches; selected cross-case prior context']))
    print(json.dumps(dict(annotation_sha256=sha(out/'annotations.json'),rows=len(rows))))

if __name__=='__main__':main()
