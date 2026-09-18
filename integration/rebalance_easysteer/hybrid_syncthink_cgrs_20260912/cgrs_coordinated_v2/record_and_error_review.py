"""Record one-assistant evidence review separately from frozen author scores."""
from prepare_length_vector import HERE,read,save,sha

def main():
    out=HERE/'and_error_audit_20260918_cpu'
    tests={r['dataset_index']:r for r in read(out/'test_cases.json')['cases']}
    cal={r['train_index']:r for r in read(out/'calibration_cases.json')['cases']}
    notes=[
        (459,'arithmetic_error','new_text','which amounts to 3,300 pounds.','42*75=3150, not3300. Candidate is longer, not simply cut short.'),
        (841,'answer_extraction_issue','new_text','Gabriel still needs **\\$5,600**.','Both derive5600. Candidate has no boxed answer; original extractor returns empty. Do not alter frozen label.'),
        (1292,'harmful_revision_visible','new_text',"So, before summer, Jackie was 67 inches tall.",'Candidate first reaches67, then repeatedly reverses shorter-than and ends71. Extra reflection is visible; not evidence of missing reflection.'),
        (694,'missing_or_wrong_operation','new_text','48 - 8 = 40','Candidate fails to use the57 attendees in final subtraction. Old trace explicitly corrects from available48 seats to required57; local necessary adjustment is visible.'),
        (322,'missing_or_wrong_operation','new_text','3x + 2 = 26','Candidate uses loose-pencil condition and obtains9. Old output incorrectly treats box capacity as unspecified and answers7.'),
        (626,'answer_format_sensitivity','old_text','\\boxed{72\\ \\text{Mb/h}}','Both compute72. Old/new author labels differ despite same numeric answer; units/spacing diagnostic, actual grader root cause not executed locally.'),
        (1075,'necessary_correction_visible','new_text','15 + 6 = $21','Candidate notices unaffordable30, uses dozen pricing, corrects quarter count to16. Intermediate half-dozen and unit statements remain wrong; correct final answer is not valid entire reasoning.'),
        (680,'answer_extraction_issue','old_text','exceeded the maximum load by 20 kg.','Both compute20. Old final response lacks boxed content and extracts empty; candidate boxes20.')]
    reviewed=[]
    for i,label,arm,quote,note in notes:
        assert quote in tests[i][arm],(i,quote)
        reviewed.append(dict(dataset_index=i,problem_sha256=tests[i]['problem_sha256'],original_flip=tests[i]['status'],category=label,evidence_arm=arm,evidence_quote=quote,interpretation=note))
    cnotes=[
        (4158,'valid_new_inference','0 to any positive exponent is still 0','Applies positive exponent to already derived zero; not a false calculation or purely repeated premise.'),
        (4042,'necessary_constraint_adjustment','A has a minimum value of 1','Identifies restriction required for stars-and-bars, followed by A-prime=A-1 and15 solutions.'),
        (563,'ordinary_task_restatement','some other pet','Opening restatement of pet categories. Phrase is in the problem itself, not evidence of reflection.'),
        (7012,'unfounded_self_correction','Wait, no, that\'s not right.','The preceding reciprocal multiplication (-5/3)*(-5/3) is correct; unsupported rejection is followed by repeated doubt. Mixed step, not a clean redundant-only label.'),
        (458,'valid_new_inference','the two draws are independent events','Replacement justifies independence; different is ordinary description of colors.'),
        (465,'ordinary_task_restatement','two different probabilities','Organizes given conditional probabilities before total-probability calculation. Different is not a metacognitive trigger here.'),
        (2152,'valid_plan','convert all the binary numbers to decimal','A valid initial solution strategy; maybe alone does not establish unnecessary exploration.'),
        (3789,'conclusion_interpretation','Steve needs to keep 4 people','Maps derived x=4 to requested headcount. Rechecking follows, but target itself cannot safely be called pure redundancy.')]
    reviewed_cal=[]
    for i,label,quote,note in cnotes:
        assert quote in cal[i]['target'],(i,quote)
        reviewed_cal.append(dict(train_index=i,problem_sha256=cal[i]['problem_sha256'],step_index=cal[i]['step_index'],category=label,evidence_quote=quote,interpretation=note))
    assert 42*75==3150 and (2+48+3150+1300)//2250==2 and 10800-5200==5600
    assert 2*36-2-3==67 and 42+15-(8+5*8)==9 and (26+46)/((26-2)/3)==9
    assert 20/1000*3600==72 and (25-(15+3*2))/.25==16 and 9*80-700==20
    result=dict(test_review=reviewed,calibration_review=reviewed_cal,
        sources={n:sha(out/n) for n in ('protocol.json','test_registry.json','test_cases.json','calibration_cases.json','surface_audit.json')},
        conclusions=['No support for a single blanket missing-reflection explanation; observed losses include arithmetic error, omitted condition, harmful revision and answer formatting.',
            'Do not retrofit empty extraction to correct: 23 candidate-empty losses and18 baseline-empty gains only identify formatting-sensitive pools.',
            'All10636 strict-AND lexical flags reproduce exactly; mixed labels reflect proxy semantics, not discovered tokenizer or index corruption.',
            'Eight earliest matching calibration steps include useful reasoning and ordinary phrases. Selection favors early steps, so do not estimate population purity.',
            'No reviewed calibration category is automatically a safe deletion intervention. No new vector, semantic classifier or GPU release is justified by these16 reviewed cases.'],
        next_spec=dict(evaluation='Before causal claims about accuracy, audit all U/R/L27/AND saved GSM final answers with one predefined supplemental extraction policy; preserve frozen author scores and publish all unresolved cases, not only flips. No regeneration.',
            process_labels='Train-side annotation must record prior premise, claimed new inference, mathematical validity, and explicit earlier support for any alleged repetition. Distinguish necessary correction, harmful revision, valid advancement, restatement and uncertain.',
            alignment='Each annotation binds train_index, normalized problem hash, generation hash, step start/stop and saved feature row. New on-policy prefixes cannot inherit old calibration features.',
            fitting='No semantic direction from this early-step sample. Establish cross-question coverage and annotation consistency before specifying training/development fit. Test-flip IDs excluded from extraction or threshold tuning.',
            gpu='None queued; current L27 stays frozen. Do not rerun AND full datasets or previous engineering.'),
        limitations=['Single assistant review, not independent human adjudication. Original correctness labels and method identities were visible.',
            'Local evidence spans establish examples, not complete proof validity or controlled causal effects. Test cohort decomposition is posthoc accounting.',
            'One initial CPU attempt failed before parsing due to utf8-sig codec spelling; fixed to utf-8-sig without changing selection or data.'],gpu_calls=0,new_answers=0)
    save(out/'review.json',result);print('Recorded8 test pairs and8 calibration steps; no score updates or vector fitting.')
if __name__=='__main__':main()
