"""CPU compile and exhaustive calibration audit; no model/GPU import."""
import argparse,hashlib,json,tarfile,time
from pathlib import Path
import numpy as np
from prepare_label_alignment import load,save,sha
from review_wsc_native_cpu import Decoder
from lexicon_automaton import Automaton,expanded_terms,compact_tables


def main():
    p=argparse.ArgumentParser();p.add_argument('--main-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root,out=args.main_root,args.output;out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    try:
        author=load('compile_author',root/'sources/ReBalance/hidden_analysis_auto.py')
        tokenizer=root/'.codex_work/label_audit_30_20260910/tokenizer.json'
        decoder=Decoder(tokenizer)
        size=max(set(decoder.vocab)|set(decoder.added))+1
        pieces=[decoder.decode([i]) for i in range(size)]
        terms=expanded_terms(author.LEXICON_BASE)
        search=Automaton(terms)
        with tarfile.open(root/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz') as tar:
            raw=tar.extractfile('generations.jsonl').read()
        assert hashlib.sha256(raw).hexdigest()=='4d0194b30bb97d1ef0779b13d24ba47ba0aa7ccf38803e6a68f5c233ceb756a3'
        rows=[json.loads(s) for s in raw.splitlines()];del raw
        steps=json.loads((root/'.codex_work/question_balanced_20260911/original_selected_layer/steps.json').read_text())
        checked=0
        for step in steps:
            ids=rows[step['question']]['token_ids'][step['start']:step['stop']]
            state,hit=search.consume(0,decoder.decode(ids))
            found=hit or bool(search.final[state])
            if found!=step['lexical_hit']:raise AssertionError(('Regex parity',step))
            # Check streamed single-token decoding separately; partial UTF8 is
            # harmless only if it preserves the lexical match on these inputs.
            state=0;hit=False
            for token in ids:
                state,h=search.consume(state,pieces[token]);hit|=h
            assert (hit or bool(search.final[state]))==found,('Token parity',step)
            checked+=1
        print('Calibration regex and token-stream checks passed:',checked,flush=True)
        meta={}
        for name,automaton in [('search',search),('opening',Automaton(terms,opening=True))]:
            tr,hits=automaton.compile_tokens(pieces)
            compact=compact_tables(tr,hits,automaton.final,name=='opening')
            assert np.array_equal(compact['transition'][:,compact['token_classes']],tr)
            assert np.array_equal(compact['hits'][:,compact['token_classes']],hits)
            with (out/(name+'.npz')).open('xb') as f:
                np.savez_compressed(f,**compact)
            meta[name]=dict(states=len(automaton.states),vocabulary=size,
                candidate_columns=len(compact['candidate_ids']),token_classes=compact['transition'].shape[1],sha256=sha(out/(name+'.npz')),
                device_table_bytes=sum(x.nbytes for x in compact.values()))
            print(name,meta[name],flush=True)
        save(out/'complete.json',dict(calibration_steps_checked=checked,token_stream_parity=True,
            tokenizer_sha256=sha(tokenizer),tables=meta,seconds=time.monotonic()-started,
            source_sha256={n:sha(Path(__file__).with_name(n)) for n in
                ['lexicon_automaton.py','compile_label_tables.py']},
            limitation='Native device acceptance and async history checks remain pending.'))
    except BaseException as e:
        save(out/'failure.json',dict(error=repr(e),seconds=time.monotonic()-started));raise


if __name__=='__main__':main()
