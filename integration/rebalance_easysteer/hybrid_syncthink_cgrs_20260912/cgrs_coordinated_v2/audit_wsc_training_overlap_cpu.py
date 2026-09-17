"""Identity-only reconciliation of released WSC training questions.

Do not inspect or grade evaluation responses, select thresholds, or change old
benchmark rows. Exact normalized matches are a lower bound on contamination.
"""
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path
import unicodedata


def question_hash(text):
    return hashlib.sha256(''.join(unicodedata.normalize('NFKC',text).split()).encode()).hexdigest()


def audit(source, data_root):
    raw=source.read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='85604aef7d620de9afacbd207c5ac901526b95ff333a647bcb597710aa912677'
    rows=json.loads(raw);assert len(rows)==1000
    lookup=defaultdict(list);ledger=[]
    for source_id,row in rows.items():
        digest=question_hash(row['question']);lookup[digest].append(source_id)
        ledger.append(dict(source_id=source_id,normalized_question_sha256=digest,
            source_type=row['source_type'],purpose='Released1.5B probe training-source exclusion, not independent development/confirmation'))
    result={}
    for path in sorted(data_root.glob('*/test.jsonl')):
        hits=[];invalid=[];count=0;missing=[]
        for i,line in enumerate(path.read_text(encoding='utf8').splitlines()):
            if not line.strip():continue
            try:r=json.loads(line)
            except json.JSONDecodeError as exc:
                invalid.append(dict(line=i+1,error=str(exc)));continue
            count+=1
            field=next((k for k in ('problem','question','Question') if isinstance(r.get(k),str)),None)
            if field is None:missing.append(i);continue
            digest=question_hash(r[field])
            if digest in lookup:hits.append(dict(zero_based_line_index=i,normalized_question_sha256=digest,wsc_source_ids=lookup[digest]))
        result[path.parent.name]=dict(file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            valid_rows=count,invalid_lines=invalid,missing_question_indices=missing,
            complete=not invalid and not missing,exact_overlap_questions=len(hits),matches=hits)
    return dict(training_revision='f1f189bcf8baaba316cfc8ed78e845d83902765c',training_source_sha256=hashlib.sha256(raw).hexdigest(),
        model_source='DeepSeek-R1-Distill-Qwen-1.5B_s1; sevenB source not yet independently reconciled',
        rows=1000,unique_normalized_questions=len(lookup),source_types=dict(Counter(r['source_type'] for r in rows.values())),
        dataset_checks=result,training_ledger=ledger,
        exclusions='Matching future evaluation questions cannot be called independent of this published training source. Do not delete or recalculate historical frozenRC14 results.',
        limitations=['Exact text normalization only; paraphrases, images and translated duplicates may remain',
        'Published source is not a checkpoint-bound train_split receipt; treat all1000as potentially exposed',
        'Incomplete dataset parsing is reported, never treated as zero overlap'],gpu_used=False,new_answers=0)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.source,a.data_root)
    with a.output.open('x',encoding='utf8',newline='\n') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:{'complete':v['complete'],'exact_overlap_questions':v['exact_overlap_questions'],'invalid_lines':len(v['invalid_lines'])} for k,v in result['dataset_checks'].items()},indent=2))
