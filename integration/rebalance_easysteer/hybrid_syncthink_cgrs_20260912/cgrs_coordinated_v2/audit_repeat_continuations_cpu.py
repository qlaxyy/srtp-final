"""Causal exact-copy detector, a CPU diagnostic, not WSC or compression evidence.

Fixed before audit: 64-token copies, require two earlier non-overlapping copies.
No semantic similarity, confidence thresholds, answer labels, or model forward.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tarfile
import unicodedata


class ExactCopy:
    def __init__(self, width=64):
        if width < 2:raise ValueError(width)
        self.width=width;self.tokens=[];self.index=defaultdict(dict)

    def candidates(self):
        """Return next IDs that would complete a third disjoint exact copy."""
        n=self.width;start=len(self.tokens)-(n-1)
        if start<0:return set()
        return {token for token,positions in self.index.get(tuple(self.tokens[-(n-1):]),{}).items()
                if len(positions)==2 and positions[1]+n<=start}

    def accept(self, token):
        hit=token in self.candidates();self.tokens.append(token);n=self.width
        if len(self.tokens)>=n:
            start=len(self.tokens)-n;key=tuple(self.tokens[-n:-1])
            positions=self.index[key].setdefault(token,[])
            if len(positions)<2 and (not positions or positions[-1]+n<=start):positions.append(start)
        return hit


def inspect(row):
    ids=row['token_ids'];ids=ids[:ids.index(151649)] if 151649 in ids else ids
    detector=ExactCopy();hits=[];covered=set()
    for i,token in enumerate(ids):
        if detector.accept(token):
            hits.append(i);covered.update(range(i-63,i+1))
    return dict(train_index=row['train_index'],problem_sha256=hashlib.sha256(''.join(unicodedata.normalize('NFKC',row['problem']).split()).encode()).hexdigest(),
        purpose='Existing exposed training/calibration data, diagnostic only; never independent confirmation',
        thinking_tokens=len(ids),matching_completions=len(hits),covered_tokens=len(covered),first_match_end=hits[0] if hits else None,
        finish_reason=row['finish_reason'])


def self_test():
    # Exhaustive naive reference verifies overlap exclusion and causality.
    import random
    rng=random.Random(42)
    for width in (2,3,4):
        for _ in range(25):
            seq=[rng.randrange(3) for _ in range(80)];d=ExactCopy(width)
            for i,x in enumerate(seq):
                if i>=width-1:
                    cur=seq[i-width+1:i+1];occ=[j for j in range(i-2*width+2) if seq[j:j+width]==cur]
                    expected=any(b>=a+width for a in occ for b in occ)
                else:expected=False
                assert (x in d.candidates())==expected
                assert d.accept(x)==expected
    d=ExactCopy(4)
    for x in [1,2,3,4]*2:d.accept(x)
    for x in [1,2,3]:d.accept(x)
    assert d.candidates()=={4}
    assert not ExactCopy(4).candidates()
    return dict(randomized_naive_comparisons=6000,nonoverlap_and_prefix_causality=True,new_request_reset=True)


def main():
    root=Path(__file__).resolve().parents[4];here=Path(__file__).resolve().parent
    output=here/'repetition_pivot_20260917';output.mkdir(exist_ok=False)
    tests=self_test();inputs=[];sets=[]
    archive=Path('E:/srtp/srtp-final/.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz')
    fit=Path('E:/srtp/srtp-final/.codex_work/auto_code_v2_500_20260908/fit.json')
    with tarfile.open(archive) as t:raw=t.extractfile('generations.jsonl').read();manifest=json.load(t.extractfile('manifest.json'))
    assert hashlib.sha256(raw).hexdigest()==json.loads(fit.read_text())['calibration_source_sha256']
    assert manifest['temperature']==0 and manifest['count']==500
    sets.append(('1p5b_greedy_unsteered_calibration500',[json.loads(x) for x in raw.splitlines()]))
    inputs.append(dict(path=str(archive),member='generations.jsonl',sha256=hashlib.sha256(raw).hexdigest()))
    archive=root/'.codex_work/history_screen100_20260914_r2/complete.tar.gz'
    with tarfile.open(archive) as t:
        plan=json.load(t.extractfile('results/resolved_plan.json'))
        assert plan['assets']['model']=='DeepSeek-R1-Distill-Qwen-7B' and plan['runtime']['temperature']==.7
        for arm in ('R','RC14'):
            name=f'results/{arm}/result.json';raw=t.extractfile(name).read();d=json.loads(raw)
            assert len(d['records'])==100 and all(r['split']=='train' for r in d['records'])
            sets.append((f'7b_temp07_mathtrain100_{arm}',d['records']))
            inputs.append(dict(path=str(archive),member=name,sha256=hashlib.sha256(raw).hexdigest()))
    summary={};ledger={}
    for name,rows in sets:
        records=[inspect(r) for r in rows];ledger[name]=records
        summary[name]=dict(questions=len(records),thinking_tokens=sum(r['thinking_tokens'] for r in records),
            hit_questions=sum(r['matching_completions']>0 for r in records),matching_completions=sum(r['matching_completions'] for r in records),
            covered_tokens=sum(r['covered_tokens'] for r in records),capped_questions=sum(r['finish_reason']=='length' for r in records),
            hit_capped_questions=sum(r['matching_completions']>0 and r['finish_reason']=='length' for r in records))
    for file,data in [('data_usage.json',ledger),('cpu_audit.json',dict(inputs=inputs,tests=tests,summary=summary,
        gpu_used=False,new_answers=0,width=64,required_prior_nonoverlapping_copies=2,
        limitations=['Coverage is not saved tokens, causal recovery, or a correctness guarantee',
        'Greedy unsteered calibration differs from RC14 sampling', 'No1.5B RC14 long sampled training trace was audited in this batch',
        '7B training100 is already exposed development data; no frozen test results were read',
        'Detecting exact repetitions misses paraphrased loops and may catch useful repeated calculations']))]:
        with (output/file).open('x',encoding='utf8',newline='\n') as f:json.dump(data,f,ensure_ascii=False,indent=2)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
