"""Content evidence and exact CPU checks; no proxy-label fitting."""
import json
import sympy as s
from prepare_length_vector import HERE,read,save,sha

REVIEWS={
4139:dict(status='error_detected_no_repair_in_window',old='Smaller angle180-x substituted into a full360-degree partition.',replacement=None,
    reason='The closing sector must instead be the reflex180+x. Target admits error; next3 steps supply no corrected equation.',
    repetition='Next approach considers the same four rays; insufficient evidence of a correction-versus-repetition pair.'),
4543:dict(status='mixed_partial_repair_in_window',old='20 percent relative diameter error becomes10 percent relative radius error; then linear approximation treated as exact.',
    replacement='Following3 recognizes radius error20 percent and that linearization is approximate for finite errors.',
    reason='Exact44/36 percent errors were already correct before the doubt. Following1 repeats them, following2 repeats the mistaken10 percent premise, following3 partially resolves the conflict.',
    repetition='Following1 repeats established areas; following3 provides a valid approximation caveat but mixes a previously known radius error. Not a clean single-step semantic pair.'),
3063:dict(status='valid_calculation_rechecked',old=None,replacement=None,
    reason='All three preceding vector components are algebraically correct. Following1-3 repeat those subtractions; no wrong proposition repaired.',
    repetition='Three explicit correct restatements, but no necessary-correction counterpart.'),
3104:dict(status='wrong_calculation_repeated',old='k_basis cross i_basis equals minus j_basis.',replacement=None,
    reason='Correct cross product is plus j_basis. Following2 repeats wrong sign and following3 propagates it; admission of error does not repair it.',
    repetition='Repeated incorrect computation, no valid replacement in window.'),
3307:dict(status='error_detected_no_repair_in_window',old='1/k^2 rewritten as25k^2/(25k^2).',replacement=None,
    reason='Numerator should be25; correct equation is75-10k=0 for k nonzero. Following3 ends at restated vector equation, before denominator repair.',
    repetition='Local window is too short to determine later repair; do not label the entire trajectory unsuccessful.'),
5977:dict(status='restoration_of_previously_known_result',old='Fixed4-dollar reduction treated as multiplier0.84 even after price becomes20.',
    replacement='Following3 computes25*0.8-4=16.',
    reason='Valid repair of the most recent mistaken computation, but16 was already explicitly known in prior context. Restoration is distinct from novel information.',
    repetition='Local correct repair revisits earlier correct result. A novelty-only suppression rule could wrongly block recovery.')}

def main():
    out=HERE/'correction_evidence_20260918_cpu';views=read(out/'reading_view.json')
    assert {v['train_index'] for v in views}==set(REVIEWS)
    x=s.symbols('x');k=s.symbols('k',nonzero=True)
    a=s.Matrix([1,-2,-5]);b=s.Matrix([s.sqrt(7),4,-1]);c=s.Matrix([13,-4,17])
    v=a.dot(c)*b-a.dot(b)*c
    assert s.simplify(v-s.Matrix([39-77*s.sqrt(7),-268+4*s.sqrt(7),115-17*s.sqrt(7)]))==s.zeros(3,1)
    assert s.simplify(a.dot(v))==0 and v!=s.zeros(3,1)
    ih=s.Matrix([1,0,0]);kh=s.Matrix([0,0,1]);jh=s.Matrix([0,1,0])
    assert kh.cross(ih)==jh
    assert 2*kh.cross(ih)+kh.cross(-ih-kh)+(-ih-kh).cross(ih)==s.zeros(3,1)
    ah=s.Matrix([s.Rational(3,5),s.Rational(4,5),0]);b2=s.Matrix([-1,1,-1]);v2=s.Rational(2,15)*b2-ah
    assert v2.dot(v2)==1 and ah+v2==s.Rational(2,15)*b2
    assert s.simplify(((-1/k-s.Rational(3,5))**2+(1/k-s.Rational(4,5))**2+1/k**2-1)*25*k**2)==75-10*k
    assert s.solve(s.Eq(180-x,s.Rational(7,2)*x),x)==[40]
    assert (s.Rational(6,5)**2-1)*100==44 and (1-s.Rational(4,5)**2)*100==36
    assert (25-4)*s.Rational(4,5)-(25*s.Rational(4,5)-4)==s.Rational(4,5)
    assert len([n for n in range(100,1000) if sum(map(int,str(n)))==5])==15
    rows=[dict(train_index=v['train_index'],**REVIEWS[v['train_index']],evidence=v) for v in views]
    save(out/'review.json',dict(reading_sha256=sha(out/'reading_view.json'),reviewer='single assistant; not independent human gold',rows=rows,
        exact_checks=dict(vector3063='components valid; dot product zero',cross3104='basis cross sign positive; scalar2 satisfies',
            unit3307='correct denominator gives k=15/2; vector(-11/15,-2/3,-2/15) has norm1',
            angle4139='using diagram order: BOC40 and AOD140',area4543='44 and36 percent exact; equal relative diameter/radius error',discount5977='16.8 versus16; difference80 cents',digits4042='enumerated15'),
        prior_anchor_correction=dict(train_index=4042,old_category='necessary_constraint_adjustment',refinement='valid preventive constraint handling, not demonstrated repair of an already asserted false proposition; A>=1 was known before target'),
        conclusion='Zero clean novel necessary-correction plus repetition pairs in the reviewed windows. One restoration of previously known correct information; one mixed partial repair. Does not imply no later repairs or no feasible pairs among500.',
        design_implication='Novelty alone is insufficient: restoring a valid proposition after a later error may be useful even when not new. Need a status distinction between active erroneous claim and historical resolved claim. No cue-only semantic labels, no feature fitting, no GPU.',
        limitations=['Repair-cue-enriched sample, not prevalence','Three following steps often too short; non-repair labels restricted to window','Outcome interpretation cannot be assigned as knowledge available at first-token feature','No quantitative effect or causal deletion-safety claim']))
    print(json.dumps(dict(checks='PASS',review_sha256=sha(out/'review.json'))))

if __name__=='__main__':main()
