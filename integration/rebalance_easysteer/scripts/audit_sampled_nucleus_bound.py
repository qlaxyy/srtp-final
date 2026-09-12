"""Sufficient modal-only nucleus bound on old greedy calibration, no replay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

import numpy as np

from mechanism_candidates import ROOT, read, save, sha, require


def audit():
    started = time.perf_counter(); temperature=.7; top_p=.95
    threshold = 1/(1+((1-top_p)/top_p)**temperature)
    rng = np.random.default_rng(20260929); checks=0
    for vocabulary in (2, 3, 37, 151936):
        for maximum in (.7, threshold-1e-5, threshold+1e-5, .95):
            tail = rng.random(vocabulary-1); tail *= (1-maximum)/tail.sum()
            probabilities = np.concatenate(([maximum], tail))
            transformed = probabilities**(1/temperature); transformed /= transformed.sum()
            lower = maximum**(1/temperature)/(maximum**(1/temperature)+(1-maximum)**(1/temperature))
            require(transformed[0]+1e-13 >= lower, 'Power-sum bound failed')
            from_logits = np.exp(np.log(probabilities)/temperature-
                                 np.max(np.log(probabilities)/temperature))
            from_logits /= from_logits.sum()
            require(abs(from_logits[0]-transformed[0])<1e-13, 'Temperature reference failed')
            if maximum>threshold:
                require(transformed[0]>top_p, 'Sufficient modal-only condition failed')
            if vocabulary==2 and maximum<threshold:
                require(transformed[0]<top_p, 'Binary extremal construction failed')
            checks+=1
    archive=ROOT/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz'
    tokenizer=ROOT/'.codex_work/label_audit_30_20260910/tokenizer.json'
    require(sha(tokenizer)=='88145e3c3249adc2546ede277e9819d6e405e19072456e4b521cbc724bd60773', 'Tokenizer changed')
    data=read(tokenizer); boundaries={value for text,value in data['model']['vocab'].items() if 'ĊĊ' in text}
    qrows=[]; digest=hashlib.sha256()
    with tarfile.open(archive) as tar:
        manifest=json.load(tar.extractfile('manifest.json'))
        require(manifest['count']==500 and manifest['temperature']==0 and
                manifest['generation']=='vLLM greedy, raw selected-token log probabilities', 'Wrong calibration provenance')
        provenance=json.load(tar.extractfile('artifact_provenance.json'))
        require(provenance['snapshot_commit']=='e28c401fd55ad5186f80775f4541123fb8966a41',
                'Unexpected preserved calibration recipe')
        require(provenance['files']['generations.jsonl']['sha256']==
                '4d0194b30bb97d1ef0779b13d24ba47ba0aa7ccf38803e6a68f5c233ceb756a3', 'Wrong provenance binding')
        original_source=subprocess.check_output(['git','show',provenance['snapshot_commit']+
            ':integration/rebalance_easysteer/scripts/calibrate_own_vector.py'],cwd=ROOT)
        for line in tar.extractfile('generations.jsonl'):
            digest.update(line); row=json.loads(line); ids=np.array(row['token_ids'])
            require(len(ids)==len(row['logprobs'])<=16000, 'Malformed saved generation')
            end=np.flatnonzero(ids==151649); count=int(end[0]) if len(end) else len(ids)
            ids=ids[:count]; probabilities=np.exp(np.array(row['logprobs'][:count],dtype=np.float64))
            require(np.isfinite(probabilities).all() and np.all((probabilities>=0)&(probabilities<=1)), 'Invalid probabilities')
            content=~np.isin(ids,list(boundaries))
            modal_only=probabilities>threshold+1e-6
            qrows.append(dict(train_index=row['train_index'],thinking_tokens=count,
                content_tokens=int(content.sum()),modal_only_sufficient_tokens=int(modal_only.sum()),
                modal_only_sufficient_content_tokens=int((content&modal_only).sum()),
                capped=row['finish_reason']=='length'))
    require(digest.hexdigest()=='4d0194b30bb97d1ef0779b13d24ba47ba0aa7ccf38803e6a68f5c233ceb756a3', 'Saved answers changed')
    require(len(qrows)==500 and sum(q['capped'] for q in qrows)==149, 'Wrong calibration population')
    totals={key:sum(row[key] for row in qrows) for key in ('thinking_tokens','content_tokens',
        'modal_only_sufficient_tokens','modal_only_sufficient_content_tokens')}
    totals['not_ruled_out_content_fraction']=1-totals['modal_only_sufficient_content_tokens']/totals['content_tokens']
    return dict(status='CPU_nucleus_bound_on_saved_greedy_prefixes_not_stochastic_opportunity',
        temperature=temperature,top_p=top_p,sufficient_raw_maximum=threshold,numeric_margin=1e-6,
        derivation='For r=1/T>1 and raw maximum m, sum(tail p_j^r)<=(1-m)^r. Hence post-temperature modal mass>=m^r/(m^r+(1-m)^r). If this exceeds top_p, nucleus keeps only the modal token. Solve for m to obtain1/(1+((1-top_p)/top_p)^T).',
        independent_probability_law_cases=checks,questions=500,capped_answers_included=149,
        totals=totals,per_question=qrows,calibration_manifest=manifest,
        original_generation_source_sha256=hashlib.sha256(original_source).hexdigest(),
        recipe_snapshot_commit=provenance['snapshot_commit'],generation_base_commit=manifest['commit'],
        historical_provenance_note=provenance['generation_started_from_commit'],
        generations_sha256=digest.hexdigest(),
        source_sha256=sha(Path(__file__),source=True),tokenizer_sha256=sha(tokenizer),
        limits='Only temperature and top_p transformations, no penalties or grammar. Ideal probability arithmetic plus a numerical margin is not a cross-kernel guarantee. These are old greedy prefixes, not trajectories generated with temperature0.7. Below the bound does not imply a nonmodal choice. Step-mean changes cannot be inferred from this token count.',
        decision='Keep the fixed100saved stochastic probability replay and all gates; this analysis neither substitutes for it nor estimates compression or accuracy.',
        cpu_seconds=time.perf_counter()-started,model_loads=0,GPU_calls=0,new_answers=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require(not args.output.exists(),'Immutable audit exists')
    result=audit();save(args.output,result)
    print({key:result[key] for key in ('status','sufficient_raw_maximum','totals','cpu_seconds','GPU_calls')})


if __name__=='__main__':
    main()
