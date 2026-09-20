"""Run original pure extraction/lexicon functions locally; do not revise scores."""
import ast,collections,hashlib,json,re,tarfile
from prepare_length_vector import HERE,ROOT,read,save,sha
from review_wsc_native_cpu import Decoder

def pure(path,names,assignments=()):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'));nodes=[]
    for n in tree.body:
        if isinstance(n,ast.FunctionDef) and n.name in names:nodes.append(n)
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in assignments for t in n.targets):nodes.append(n)
    namespace={'re':re};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace);return namespace
def main():
    out=HERE/'and_error_audit_20260918_cpu';parser=ROOT/'sources/ReBalance/utils/parser.py';author=ROOT/'sources/ReBalance/hidden_analysis_auto.py'
    extract=pure(parser,{'extract_answer'})['extract_answer']
    lex=pure(author,{'_normalize_text','_token_pattern','_compile_lexicon_regex','has_lexicon_hit'},['LEXICON_BASE','LEXICON_RE'])
    old=HERE.parents[3]/'.codex_work/label_alignment_20260917/label_alignment_gsm1319_20260917_run1'
    refs=read(HERE/'strict_and_gsm_20260918_run1/historical_compact.json')
    assert sha(parser)==refs['grader_sha256']['parser.py']
    baseline=read(old/'L27_L27/result.json')['records']
    with tarfile.open(HERE.parents[3]/'.codex_work/strict_and_gsm_20260918_run1.completed.tar.gz') as t:
        candidate=json.load(t.extractfile('strict_and_gsm_20260918_run1/full/AND_NORM_L27/result.json'))['records']
    registry=read(out/'test_registry.json')['rows'];rows=[]
    for r,a,b in zip(registry,baseline,candidate):
        assert r['dataset_index']==a['dataset_index']==b['dataset_index']
        pa,pb=extract(a['text']),extract(b['text'])
        rows.append(dict(dataset_index=r['dataset_index'],status=r['status'],old_extracted=pa,new_extracted=pb,old_empty=not pa,new_empty=not pb))
    counts={group:dict(count=sum(r['status']==group for r in rows),old_empty=sum(r['status']==group and r['old_empty'] for r in rows),new_empty=sum(r['status']==group and r['new_empty'] for r in rows)) for group in ('loss','gain','both_correct','both_wrong')}
    # All strict-AND stored labels must reproduce from original token intervals.
    with tarfile.open(ROOT/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz') as t:raw=t.extractfile('generations.jsonl').read()
    assert hashlib.sha256(raw).hexdigest()=='4d0194b30bb97d1ef0779b13d24ba47ba0aa7ccf38803e6a68f5c233ceb756a3'
    gens=[json.loads(x) for x in raw.splitlines()];steps=read(ROOT/'.codex_work/question_balanced_20260911/original_selected_layer/steps.json')
    lo=read(ROOT/'.codex_work/auto_code_v2_500_20260908/protocol.json')['confidence_quantiles'][0]
    dec=Decoder(ROOT/'.codex_work/label_audit_30_20260910/tokenizer.json');hits=collections.Counter();selected=[]
    selected_indices={r['step_index'] for r in read(out/'calibration_cases.json')['cases']}
    for i,s in enumerate(steps):
        if not(s['lexical_hit'] and s['confidence']<lo):continue
        text=dec.decode(gens[s['question']]['token_ids'][s['start']:s['stop']]);matches=[m.group(0) for m in lex['LEXICON_RE'].finditer(lex['_normalize_text'](text).lower())]
        assert matches,(i,text);hits.update(set(matches))
        if i in selected_indices:selected.append(dict(step_index=i,matched_strings=matches))
    assert sum(s['lexical_hit'] and s['confidence']<lo for s in steps)==10636
    save(out/'surface_audit.json',dict(extraction_counts=counts,extraction_rows=rows,selected_calibration_matches=selected,
        strict_steps=10636,distinct_matches_by_step=dict(hits.most_common()),source_sha256=dict(parser=sha(parser),author_lexicon=sha(author)),
        limitations='Empty extraction is a format finding, not a correctness label. All old labels retained. Lexical ordinary-language hits are not automatically safe-to-delete reflection. No new grading or GPU.'))
    print(json.dumps(dict(extraction_counts=counts,selected_calibration_matches=selected,most_common=hits.most_common(12))))
if __name__=='__main__':main()
