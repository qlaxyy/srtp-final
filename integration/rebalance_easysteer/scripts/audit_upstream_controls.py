"""Bounded CPU source/label audit; no upstream script or model is executed."""
import argparse
import ast
import json
from pathlib import Path
import tarfile
import tempfile
from collections import Counter
from mechanism_candidates import ROOT,BASE,read,save,sha,require


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    require(not out.exists(),'Audit exists');work=ROOT/'.codex_work/overnight_research_20260912';up=work/'upstream'
    convergence=up/'reasoning_earlystop/src/early_stopping_via_consistency.py'
    tree=ast.parse(convergence.read_text(encoding='utf-8'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='process_file')
    scope={'json':json};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(convergence),'exec'),scope)
    with tempfile.TemporaryDirectory() as temporary:
        p=Path(temporary);source=p/'input.jsonl';target=p/'output.jsonl'
        source.write_text(''.join(json.dumps(dict(question='synthetic',answer='',position=i))+'\n' for i in range(12)),encoding='utf-8')
        scope['process_file'](str(source),str(target),10);record=read(target)
        require(record['answer']=='' and record['position']==9,'Upstream empty-answer behavior changed')
    inputs=read(ROOT/BASE/'configs/overnight_research_20260912.json')['first_investigation']['inputs']
    with tarfile.open(ROOT/inputs['archive']) as archive:raw=archive.extractfile(inputs['member']).read()
    rows=[json.loads(line) for line in raw.splitlines()];maps=read(work/'seal_prepared/positions.json')
    original=read(ROOT/inputs['backup']/'positions.json');vocab=read(ROOT/inputs['tokenizer'])['model']['vocab'];boundary={i for token,i in vocab.items() if 'ĊĊ' in token}
    extra=Counter();missing=0;examples=[]
    for q,(row,mapping,old) in enumerate(zip(rows,maps,original,strict=True)):
        prompt=len(row['prompt_token_ids']);previous={p-1 for p in old['positions'] if p-1>=prompt}
        current=set(mapping['positions']);missing+=len(previous-current)
        for pos in current-previous:
            i=pos-prompt
            if i+1>=old['think_stop'] or row['token_ids'][i+1] in boundary:kind='empty_following_segment'
            elif len(row['token_ids'])==16000 and old['think_stop']==16000:kind='capped_final_partial_step'
            else:kind='other_inspect'
            extra[kind]+=1;examples.append(dict(question=q,input_position=pos,kind=kind))
    result=dict(status='completed_cpu_source_and_label_audit',new_answers=0,model_forwards=0,
        answer_convergence=dict(commit=read(up/'reasoning_earlystop/source_commit.json')['sha'],
            implementation='collect_partial_reasoning_res.py first generates probes for all sentence prefixes of saved full traces; early_stopping_via_consistency.py then selects a record from the saved JSONL. This workflow does not measure an online stopping engine or its prefix/probe cost.',
            synthetic_empty_answer_threshold10_stops_at_index=record['position'],
            limitation='Synthetic parser edge case only; does not establish empty-answer frequency or published benchmark impact.'),
        reflctrl=dict(commit=read(up/'ReflCtrl/source_commit.json')['sha'],
            calibration='collect_activation.py:95-118 selects offset+delimiter_index, and offset-1 for the first step. Stored module-output features are at the preceding delimiter, not the first content input token.',
            online='hook_utils.py:363-375 matches current input IDs; llm_server.py:263-275 supplies delimiter IDs. The public step_begin_only path targets delimiter input positions.',
            limitation='Paper Eq5 first-content notation and public implementation differ. This explains terminology only; does not alter the failed local position-candidate result.'),
        seal_position_difference=dict(original_generated_complete_predecessors=83508,author_delimiters=83667,missing_original=missing,extra_counts=dict(extra),extra_positions=examples),
        source_sha256={str(path.relative_to(ROOT)):sha(path,source=True) for repo in ['ReflCtrl','reasoning_earlystop'] for path in (up/repo).rglob('*') if path.suffix in ('.py','.sh','.md')},
        audit_script_sha256=sha(Path(__file__),source=True))
    save(out,result);print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','seal_position_difference']}));print(dict(extra))


if __name__=='__main__':main()
