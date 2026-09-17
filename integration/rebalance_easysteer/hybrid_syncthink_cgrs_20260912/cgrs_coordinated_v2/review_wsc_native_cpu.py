"""Deterministic context export for qualitative native observation review.

Exports every trigger plus highest-score nontrigger per request. Exact repeats
are diagnostics, never semantic gold labels. No parameters or outputs edited.
"""
import argparse
import hashlib
import json
from pathlib import Path
from audit_repeat_continuations_cpu import ExactCopy


class Decoder:
    def __init__(self,path):
        raw=Path(path).read_bytes()
        assert hashlib.sha256(raw).hexdigest()=='88145e3c3249adc2546ede277e9819d6e405e19072456e4b521cbc724bd60773'
        tok=json.loads(raw);assert tok['decoder']['type']=='ByteLevel'
        self.vocab={i:s for s,i in tok['model']['vocab'].items()}
        self.added={r['id']:r['content'] for r in tok['added_tokens']}
        bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs[:];n=0
        for i in range(256):
            if i not in bs:bs.append(i);cs.append(256+n);n+=1
        self.bytes={chr(c):b for b,c in zip(bs,cs)}

    def decode(self,ids):
        chunks=[]
        for i in ids:
            if i in self.added:chunks.append(self.added[i].encode('utf8'))
            else:chunks.append(bytes(self.bytes[c] for c in self.vocab[i]))
        return b''.join(chunks).decode('utf8',errors='replace')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--scores',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    plan=json.loads((a.run/'plan.json').read_text());result=json.loads((a.run/'RC14_wsc_shadow/result.json').read_text())
    scores=json.loads(a.scores.read_text());decoder=Decoder(a.tokenizer);rows=[]
    for r,question in zip(result['records'],plan['rows']):
        idx=r['train_index'];assert idx==question['train_index'];ids=r['token_ids']
        thought=ids[:ids.index(151649)] if 151649 in ids else ids
        d=ExactCopy();hits=[i for i,t in enumerate(thought) if d.accept(t)]
        events=scores['results'][str(idx)]['events'];contexts=[]
        chosen=[e for e in events if e['would_trigger']]
        nontriggers=[e for e in events if not e['would_trigger']]
        if nontriggers:chosen.append(max(nontriggers,key=lambda e:e['score']))
        for e in chosen:
            pos=e['position'];contexts.append(dict(event=e,start=max(0,pos-256),end=min(len(ids),pos+257),
                text=decoder.decode(ids[max(0,pos-256):pos+257])))
        rows.append(dict(train_index=idx,problem_sha256=r['problem_sha256'],problem=question['problem'],
            generated_tokens=len(ids),thinking_tokens=len(thought),finish_reason=r['finish_reason'],
            exact_copy_first=hits[0] if hits else None,exact_copy_completions=len(hits),
            scored_boundaries=len(events),high_score_steps=sum(e['score']>.5 for e in events),
            trigger_positions=[e['position'] for e in events if e['would_trigger']],contexts=contexts,text=decoder.decode(ids)))
    with a.output.open('x',encoding='utf8') as f:json.dump(dict(rows=rows,
        scope='Qualitative exposed engineering cases, no semantic gold labels or paired efficacy estimate'),f,ensure_ascii=False,indent=2)


if __name__=='__main__':main()
