"""Fixed lexical-category substitution. CPU only, reuses frozen trajectories."""
import argparse,collections,hashlib,json,re,tarfile,time,unicodedata
from pathlib import Path
import numpy as np
from prepare_label_alignment import load,save,sha
from review_wsc_native_cpu import Decoder
from lexicon_automaton import Automaton,expanded_terms,compact_tables

HERE=Path(__file__).resolve().parent
GROUPS={'CHECK7':['wait','think again','make sure','let me check','hold on','double check','any errors'],
        'SWITCH5':['alternatively','alternative','another','differently','alternate']}

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--main-root',type=Path,required=True)
    p.add_argument('--raw-root',type=Path,required=True);a=p.parse_args()
    out=HERE/'type_split_20260917';out.mkdir(exist_ok=False);start=time.monotonic()
    root=a.main_root;author=load('type_author',root/'sources/ReBalance/hidden_analysis_auto.py')
    terms=set(sum(GROUPS.values(),[]));assert len(terms)==12 and terms<=set(author.LEXICON_BASE)
    ignored=[x for x in author.LEXICON_BASE if x not in terms]
    save(out/'partition.json',dict(groups=GROUPS,unclassified_not_suppressed=ignored,
        status='Fixed before outcome inspection; lexical categories are weak proxies, not semantic truth',
        source='ATLAS A.2 motivates separation; this exact partition is our adaptation, not its algorithm',
        ambiguity="wait/another/differently can be ordinary discourse; no claim that checking is redundant or switching useful"))
    tokenizer=root/'.codex_work/label_audit_30_20260910/tokenizer.json';dec=Decoder(tokenizer)
    pieces=[dec.decode([i]) for i in range(max(set(dec.vocab)|set(dec.added))+1)]
    with tarfile.open(root/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz') as t:
        raw=t.extractfile('generations.jsonl').read()
    assert hashlib.sha256(raw).hexdigest()=='4d0194b30bb97d1ef0779b13d24ba47ba0aa7ccf38803e6a68f5c233ceb756a3'
    rows=[json.loads(s) for s in raw.splitlines()];del raw
    steps=read(root/'.codex_work/question_balanced_20260911/original_selected_layer/steps.json')
    regex={k:author._compile_lexicon_regex(v) for k,v in GROUPS.items()}
    opening={k:re.compile(r'^\s*(?:'+r'|'.join(author._token_pattern(x) for x in v)+r')',re.I) for k,v in GROUPS.items()}
    counts=collections.Counter();examples={k:[] for k in GROUPS};registry=[]
    for row in rows:
        q=row.get('problem',row.get('question'));h=hashlib.sha256(''.join(unicodedata.normalize('NFKC',q).split()).encode()).hexdigest()
        registry.append(dict(train_index=row['train_index'],normalized_problem_sha256=h,purpose='Existing calibration lexical CPU audit only'))
    for step in steps:
        text=author._normalize_text(dec.decode(rows[step['question']]['token_ids'][step['start']:step['stop']]))
        hits={k:bool(rx.search(text)) for k,rx in regex.items()}
        for k in GROUPS:
            counts[k+'_whole']+=hits[k];op=bool(opening[k].search(text));counts[k+'_opening']+=op
            if op and len(examples[k])<12:examples[k].append(dict(question=step['question'],start=step['start'],text=text[:350]))
        counts['whole_both']+=all(hits.values());counts['L27_whole']+=bool(author.has_lexicon_hit(text))
    save(out/'calibration_audit.json',dict(questions=len(rows),steps=len(steps),counts=dict(counts),examples=examples,
        caveat='Corpus co-occurrence and lexical coverage only; not semantic validity, intervention frequency or efficacy'))
    save(out/'calibration_registry.json',registry)
    print('Calibration coverage',dict(counts),flush=True)
    for name,words in GROUPS.items():
        autom=Automaton(expanded_terms(words),opening=True);tr,hit=autom.compile_tokens(pieces)
        table=compact_tables(tr,hit,autom.final,True)
        assert np.array_equal(table['transition'][:,table['token_classes']],tr)
        assert np.array_equal(table['hits'][:,table['token_classes']],hit)
        # Exhaustive vocabulary checks from start and representative phrase prefixes.
        for prefix in ['', ' ', '\n', 'let ', 'let me ', 'think ', 'double ', 'another ', 'word ']:
            state,_=autom.consume(0,prefix)
            if autom.final[state] or opening[name].search(prefix):continue
            expected=np.array([bool(opening[name].search(prefix+s)) for s in pieces])
            assert np.array_equal(expected,hit[state]),(name,prefix)
        with (out/(name+'.npz')).open('xb') as f:np.savez_compressed(f,**table)
        print(name,'candidate tokens',len(table['candidate_ids']),flush=True)
    old=HERE/'label_alignment_20260917';plan=read(old/'plan_v2_math500.json')
    plan.pop('factorial_comparisons',None)
    plan.update(run_id='type_split_1p5b_math500_20260917_run1',status='Fixed GPU experiment; results pending',
        experiment_kind='type_split_v1',arms=[dict(name=k,calibration='frozen L27',controller='unchanged native ReBalance',
        suppression_table='type_split_20260917/'+k+'.npz') for k in GROUPS],
        hard_stop_seconds_per_arm=1800,process_hard_stop_seconds=4200,
        execution=dict(max_num_seqs=256,max_num_batched_tokens=32768,max_model_len=32768,gpu_memory_utilization=.9,chunked_prefill=False,sync_replay=False),
        decision=dict(primary_reference='L27_L27',max_accuracy_loss_pp=2,
        advance_to_gsm='Both thinking and total mean tokens lower than L27, caps no higher, observed accuracy loss <=2pp versus L27 AND R. If both pass choose lower total tokens, then higher accuracy. No retuning.',
        uncertainty='20000 paired bootstrap; nominal CIs descriptive on repeatedly exposed test set. Report observed pass separately from CI confirmation.',
        stop='Engineering mismatch, async preemption, OOM, asset mismatch or fixed timeout stops batch; preserve partials. No partial efficacy claim.'),
        interpretation='New exploratory module substitution; no independent confirmation, no causal synergy claim')
    save(out/'plan.json',plan)
    refs=read(old/'historical_compact.json');result=read(a.raw_root/'L27_L27/result.json')
    labels=[json.loads(s) for s in (a.raw_root/'L27_L27/author_partial.jsonl').read_text().splitlines()]
    assert len(labels)==len(result['records'])==500
    compact=[]
    for label,r,q in zip(labels,result['records'],plan['rows']):
        assert label['problem_sha256']==r['problem_sha256']==q['problem_sha256']
        assert label['text_sha256']==hashlib.sha256(r['text'].encode()).hexdigest()
        compact.append(dict(label,tokens=r['tokens'],thinking_tokens=r['thinking_tokens']))
    refs['groups']['L27_L27']=dict(records=compact,generation_seconds=result['generation_seconds'],
        source_result_sha256=sha(a.raw_root/'L27_L27/result.json'),source_labels_sha256=sha(a.raw_root/'L27_L27/author_partial.jsonl'))
    save(out/'historical_compact.json',refs)
    release=read(old/'release_v2.json')
    release.update(status='CPU prepared; native engineering required',plan_relative_path='type_split_20260917/plan.json',
        plan_sha256=sha(out/'plan.json'),artifact_root='type_split_20260917',references_artifact='historical_compact.json')
    release['source_sha256']={n:hashlib.sha256((HERE/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in release['source_sha256']}
    release['source_sha256']['prepare_type_split.py']=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    release['artifact_sha256']={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    release['cpu_checks']=dict(calibration_steps=len(steps),tables_exhaustive_representative_prefixes=True,seconds=time.monotonic()-start)
    for key in ['commands','table_audit','tests']:release.pop(key,None)
    save(out/'release.json',release)
    print('READY',time.monotonic()-start,flush=True)

if __name__=='__main__':main()
