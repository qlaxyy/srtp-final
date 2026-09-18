"""Independently verify numeric diagnostic counts; preserve author labels."""
import ast,collections,json,math,re,typing
from prepare_length_vector import HERE,ROOT,read,save,sha
from numeric_format_diagnostic import value

def main():
    out=HERE/'numeric_format_diagnostic_20260918_cpu';result=read(out/'result.json');rows=read(out/'rows.json')
    assert len(rows)==5276 and len({(r['arm'],r['dataset_index']) for r in rows})==5276
    for n,h in result['output_sha256'].items():assert sha(out/n)==h
    for arm,s in result['summaries'].items():
        rs=[r for r in rows if r['arm']==arm];assert len(rs)==1319
        assert sum(r['original_correct'] for r in rs)==s['original_correct']
        assert sum(r['supplemental_correct'] is None for r in rs)==s['unresolved']
        assert sum(r['supplemental_correct'] is True for r in rs)==s['resolved_correct']
        for r in rs:
            if r['supplemental_correct'] is not None:assert (value(r['extraction']['value'])==value(r['gold']))==r['supplemental_correct']
    # Execute original numeric branch without loading unavailable symbolic dependencies.
    grader=ROOT/'sources/ReBalance/utils/grader.py';tree=ast.parse(grader.read_text(encoding='utf8'))
    names={'parse_digits','is_digit','numeric_equal','math_equal'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='single_choice_patterns' for t in n.targets)]
    ns=dict(Union=typing.Union,re=re,regex=re,isclose=math.isclose)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(grader),'exec'),ns)
    tests=[]
    for a,b in [('3','300'),('18','1800'),('24500','245')]:
        default=ns['math_equal'](a,b);strict=ns['math_equal'](a,b,include_percentage=False)
        assert default and not strict;tests.append(dict(prediction=a,gold=b,default_percentage_true=default,percentage_false=strict))
    disputes=read(out/'disagreements.json');unresolved=read(out/'unresolved.json')
    review=dict(local_checks_passed=True,numeric_branch_tests=tests,grader_sha256=sha(grader),
        reviewed_original_true_to_literal_false=[dict(arm=r['arm'],dataset_index=r['dataset_index'],value=r['extraction']['value'],gold=r['gold']) for r in disputes if r['original_correct']],
        interpretation=['Author numeric comparison defaults to accepting gold/100 and gold*100 without inspecting whether the question asks for a percentage.',
            'U951 answers3 pages instead of300; R/L27 at1182 answer18 books instead of1800. Original labels true are reproduced by percentage scaling.',
            'R1067 gives24500 cents versus gold245 dollars; this is unit-sensitive and not automatically a semantic error.',
            'All four1113 outputs show -3 degrees but our narrow grammar does not parse the degree symbol. Keep all four unresolved under the frozen diagnostic; no one-arm exception.',
            'Literal numeric equality neither handles every unit conversion nor proves reasoning validity. Do not publish resolved-only accuracy or overwrite author scores.'],
        unresolved_reason_counts=dict(collections.Counter(r['extraction']['reason'] for r in unresolved)),
        decision='Retain L27; do not revive AND or run GPU on the premise that formatting explains all accuracy loss. Under this fixed diagnostic AND still trails L27 even at its favorable unresolved assignment; the2pp margin remains undetermined by unresolved ranges.',
        next_evaluation_spec='For future experiments retain author grading for historical comparability and separately freeze a unit-aware final-answer protocol on training examples before new benchmark generation. Adjudicate all unsupported outputs symmetrically. Never tune extraction against benchmark gold.',
        limits='Source numeric branch reproduced using standard regex for the simple numeric-only cases, not the full symbolic grader. One assistant content review, no new formal accuracy claim.',
        result_sha256=sha(out/'result.json'),gpu_calls=0,new_answers=0)
    save(out/'review.json',review);print(json.dumps(review,ensure_ascii=True))
if __name__=='__main__':main()
