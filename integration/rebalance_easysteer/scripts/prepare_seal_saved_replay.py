"""Prepare author-SEAL labels on exact saved500 token streams, without a model."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile
from mechanism_candidates import ROOT,BASE,read,save,sha,require


def author_index_function(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='generate_index')
    namespace={};exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['generate_index']


class SavedTokenAdapter:
    """Use saved token IDs; decode byte-level BPE for the author's text labels."""
    def __init__(self,tokenizer):
        vocab=tokenizer['model']['vocab'];self.by_id={i:t for t,i in vocab.items()}
        for item in tokenizer.get('added_tokens',[]):self.by_id[item['id']]=item['content']
        byte_values=list(range(33,127))+list(range(161,173))+list(range(174,256))
        chars=byte_values.copy();extra=0
        for b in range(256):
            if b not in byte_values:byte_values.append(b);chars.append(256+extra);extra+=1
        self.byte_for={chr(c):b for b,c in zip(byte_values,chars)}
        self.special={item['id']:item['content'] for item in tokenizer.get('added_tokens',[])}
        self.marker_ids={s:[next(i for i,t in self.by_id.items() if t==s)] for s in ['<think>','</think>']}
        self.ids=[]

    def encode(self,text,add_special_tokens=True):
        if text in self.marker_ids:return self.marker_ids[text]
        require(text=='saved_token_sequence','Unexpected retokenization request')
        return self.ids

    def decode(self,ids):
        out=bytearray()
        for i in ids:
            if i in self.special:out.extend(self.special[i].encode('utf-8'))
            else:out.extend(self.byte_for[c] for c in self.by_id[i])
        return out.decode('utf-8',errors='replace')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    require(not out.exists(),'Preparation exists')
    config=read(ROOT/BASE/'configs/overnight_research_20260912.json');inputs=config['first_investigation']['inputs']
    source=ROOT/'.codex_work/overnight_research_20260912/upstream/SEAL/hidden_analysis.py'
    upstream=read(source.parent/'source_commit.json')
    author=author_index_function(source);tokenizer=read(ROOT/inputs['tokenizer']);adapter=SavedTokenAdapter(tokenizer)
    boundary={i for t,i in tokenizer['model']['vocab'].items() if 'ĊĊ' in t}
    with tarfile.open(ROOT/inputs['archive']) as archive:raw=archive.extractfile(inputs['member']).read()
    require(hashlib.sha256(raw).hexdigest()==inputs['generations_sha256'],'Saved answers changed')
    rows=[json.loads(s) for s in raw.splitlines()];require(len(rows)==500,'Expected saved500')
    original=read(ROOT/inputs['backup']/'positions.json');maps=[];counts=Counter();support={i:set() for i in range(3)}
    for q,row in enumerate(rows):
        adapter.ids=row['prompt_token_ids']+row['token_ids']
        positions,check,switch=author('saved_token_sequence',adapter,boundary,think_only=True)
        classes=[1 if i in set(check) else 2 if i in set(switch) else 0 for i in range(len(positions))]
        require(all(p<len(row['prompt_token_ids'])+original[q]['think_stop'] for p in positions),'Position past reasoning end')
        for value in classes:counts[value]+=1;support[value].add(q)
        maps.append(dict(question=q,positions=positions,classes=classes,prompt_tokens=len(row['prompt_token_ids']),think_stop=original[q]['think_stop'],reference_first=original[q]['positions']))
    require(all(counts[i]>0 for i in range(3)),'Missing SEAL class')
    out.mkdir(parents=True);save(out/'positions.json',maps)
    replay=read(ROOT/'.codex_work/overnight_research_20260912/control_point_prepared/replay_plan.json')
    plan=dict(status='prepared_saved_answer_class_means_only',questions=500,classes=['execution','reflection','transition'],
        class_steps={str(i):counts[i] for i in range(3)},class_questions={str(i):len(support[i]) for i in range(3)},
        positions_sha256=sha(out/'positions.json'),original_boundary_count=sum(len(m['positions']) for m in maps),
        prompt_boundary_count=sum(p<m['prompt_tokens'] for m in maps for p in m['positions']),
        model=replay['model'],model_files_sha256=replay['model_files_sha256'],generations=replay['generations'],generations_sha256=replay['generations_sha256'],
        original=replay['original'],original_files_sha256=replay['original_files_sha256'],
        decoder_output_layer=19,reference_output_layer=20,source_hidden_state_index=20,
        original_source=upstream,author_label_source_sha256=sha(source,source=True),
        author_label_scope='Author generate_index function unchanged, but encode is supplied the exact saved prompt+answer token IDs. Server rechecks all labels using the real tokenizer before model forward.',
        vector_formula='Pooled execution mean minus pooled union(reflection,transition) mean; static alpha1 at delimiter input after decoder output19, matching author layer20 input. No normalization or parameter sweep.',
        input_scope='Original500 MATH-train greedy saved responses. Not the authors500correct+500incorrect selection from10000; not EasySteer three-mean example.',
        calibration_limitations=['Includes the author parser final tail, even when the saved answer was capped.','Original calibration remains development data; no independent efficacy evidence from mean separation.'],
        timeout_seconds=900,expected_gpu_minutes=[2,5],expected_new_bytes=500*3*1536*8+1000000,
        outputs=['class_sums.npy float64 500x3x1536','class_counts.npy int64 500x3','original_feature_checks.json'],
        safety='No model.generate and no new calibration answers. Only missing selected-layer class sums; all original first-content features checked in same replay.',
        source_sha256={BASE+'scripts/prepare_seal_saved_replay.py':sha(Path(__file__),source=True)})
    save(out/'replay_plan.json',plan);print(json.dumps({k:v for k,v in plan.items() if k not in ['model_files_sha256','original_files_sha256']}))


if __name__=='__main__':main()
