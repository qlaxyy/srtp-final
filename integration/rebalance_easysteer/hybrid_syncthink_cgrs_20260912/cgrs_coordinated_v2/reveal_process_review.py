"""Descriptive proxy-label comparison after locking content review."""
import json
from collections import Counter
from prepare_length_vector import HERE, ROOT, read, save, sha

def main():
    out=HERE/'process_review_20260918_cpu'
    assert sha(out/'annotations.json')=='73e8268453e605c2377ef3557a69a0feccf976399194a5c73eefc2c8fd145201'
    assert not (out/'comparison.json').exists()
    annotations=read(out/'annotations.json')['rows']
    identities={r['case_id']:r for r in read(out/'identities.json')}
    q25,q75=read(ROOT/'.codex_work/auto_code_v2_500_20260908/protocol.json')['confidence_quantiles']
    rows=[]
    for a in annotations:
        r=identities[a['case_id']]; c=r['confidence']; lex=r['lexical_hit']
        rows.append(dict(**r,content_label=a['label'],local_validity=a['local_validity'],
            original_class='over' if lex or c<q25 else 'under' if not lex and c>q75 else 'unused',
            strict_and_over=bool(lex and c<q25)))
    summary={}
    for label in sorted({r['content_label'] for r in rows}):
        rs=[r for r in rows if r['content_label']==label]
        summary[label]=dict(steps=len(rs),parents=len({r['question'] for r in rs}),
            original_classes=dict(Counter(r['original_class'] for r in rs)),strict_and_over=sum(r['strict_and_over'] for r in rs),
            lexical_hits=sum(bool(r['lexical_hit']) for r in rs),capped_steps=sum(r['thinking_tokens']==16000 for r in rs),
            confidence_min=min(r['confidence'] for r in rs),confidence_max=max(r['confidence'] for r in rs))
    result=dict(annotation_sha256=sha(out/'annotations.json'),identities_sha256=sha(out/'identities.json'),
        q25=q25,q75=q75,summary=summary,rows=rows,
        limitations='Descriptive selected-step counts, not purity estimates; six capped and six uncapped parents; three dependent steps each; one reviewer; no necessary-correction examples; no vector fit or threshold selection.')
    save(out/'comparison.json',result)
    print(json.dumps(dict(summary=summary,comparison_sha256=sha(out/'comparison.json')),indent=2))

if __name__=='__main__':main()
