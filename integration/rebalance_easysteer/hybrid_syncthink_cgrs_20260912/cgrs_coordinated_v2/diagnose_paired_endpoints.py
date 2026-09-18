"""CPU-only descriptive diagnostics; no fitted deployment vector or test feedback."""
import hashlib
import io
import json
import tarfile
from pathlib import Path
import numpy as np


def sha(b):
    return hashlib.sha256(b).hexdigest()


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    home = Path(__file__).resolve().parent
    root = next(p for p in home.parents if (p / '.codex_work').is_dir())
    old = home / 'outcome_endpoints_20260918_run1'
    out = home / 'paired_endpoint_diagnostic_20260918_run1'
    out.mkdir(exist_ok=False)
    protocol = dict(kind='CPU descriptive diagnostics, no new performance claim',
        variants=['old_selection', 'post_divergence_all', 'post_divergence_matched'],
        matching='Within question, after exact common token prefix, confidence difference <=0.025, normalized token position difference <=0.10, same lexical-hit flag; greedy smallest standardized distance, no reuse',
        weights='Equal question weights, equal selected steps within each question',
        stability='Leave-one-question-out cosine and 200 random equal splits; source-stratified long-U versus long-L27 cosine',
        limits=['Position/confidence matching does not establish semantic equivalence or causal redundancy.',
                'These are diagnostics on previously used training data, not an independent confirmation.',
                'No thresholds are selected using MATH500 outcomes; no GPU run or deployable vector.'])
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    archive = root / '.codex_work/outcome_endpoints_20260918_evidence.tar.gz'
    raw = archive.read_bytes()
    assert sha(raw) == 'ba391343b8b86e28514f4dc638a7bc189979883727a17b3fe0f1b757bfce22ff'
    tf = tarfile.open(fileobj=io.BytesIO(raw))
    prefix = 'outcome_endpoints_20260918_run1/features/'
    manifest = json.load(tf.extractfile(prefix+'manifest.json'))
    data = {}
    for r in manifest:
        if r['kind'] != 'CC':
            continue
        b = tf.extractfile(prefix+r['file']).read()
        assert sha(b) == r['sha256']
        x = np.load(io.BytesIO(b))['features'].astype(np.float64)
        assert len(x) == len(r['steps']) and np.isfinite(x).all()
        data[r['question'], r['side']] = (r, x)
    records = json.loads((old/'rows.json').read_text())
    tokens = {(r['question'], r['side']): r['token_ids'] for r in records}
    fit = json.loads((old/'fitted_audited/EFFICIENT/report.json').read_text())
    cl, ch = fit['quantiles']['confidence']
    variants = {k: [] for k in protocol['variants']}
    selected = {k: [] for k in variants}
    prefix_rows = []
    for q in sorted({k[0] for k in data}):
        lr, lx = data[q, 'long']; sr, sx = data[q, 'short']
        lt, st = tokens[q, 'long'], tokens[q, 'short']
        common = next((i for i, (a,b) in enumerate(zip(lt,st)) if a != b), min(len(lt),len(st)))
        ls, ss = lr['steps'], sr['steps']
        oi = [i for i,s in enumerate(ls) if s['lexical_hit'] or s['confidence'] < cl]
        ui = [i for i,s in enumerate(ss) if not s['lexical_hit'] and s['confidence'] > ch]
        li = [i for i,s in enumerate(ls) if s['start'] >= common]
        si = [i for i,s in enumerate(ss) if s['start'] >= common]
        prefix_rows.append(dict(question=q, problem_sha256=lr['problem_sha256'], long_source=lr['source'],
            common_tokens=common, long_tokens=len(lt), short_tokens=len(st),
            old_long_common_states=sum(ls[i]['start']<common for i in oi),
            old_short_common_states=sum(ss[i]['start']<common for i in ui)))
        edges=[]
        for i in li:
            for j in si:
                dc=abs(ls[i]['confidence']-ss[j]['confidence'])
                dp=abs(ls[i]['start']/len(lt)-ss[j]['start']/len(st))
                if dc <= .025 and dp <= .10 and ls[i]['lexical_hit']==ss[j]['lexical_hit']:
                    edges.append((dc/.025+dp/.10,i,j))
        used_l=set(); used_s=set(); matched=[]
        for _,i,j in sorted(edges):
            if i not in used_l and j not in used_s:
                used_l.add(i);used_s.add(j);matched.append((i,j))
        indexes={'old_selection':(oi,ui), 'post_divergence_all':(li,si),
                 'post_divergence_matched':([i for i,j in matched],[j for i,j in matched])}
        for name,(ii,jj) in indexes.items():
            if not ii or not jj: continue
            delta=lx[ii].mean(0)-sx[jj].mean(0)
            variants[name].append((q,lr['source'],delta))
            selected[name].append(dict(question=q,problem_sha256=lr['problem_sha256'],long_source=lr['source'],
                long_step_indices=ii,short_step_indices=jj,
                long_confidence=float(np.mean([ls[i]['confidence'] for i in ii])),
                short_confidence=float(np.mean([ss[j]['confidence'] for j in jj]))))
    summary={}; means={}
    for name, rows in variants.items():
        x=np.stack([r[2] for r in rows]); d=x.mean(0); means[name]=d
        loo=np.array([cosine(d,(x.sum(0)-v)/(len(x)-1)) for v in x])
        rng=np.random.default_rng(20260918); splits=[]
        for _ in range(200):
            ix=rng.permutation(len(x)); half=len(ix)//2
            splits.append(cosine(x[ix[:half]].mean(0),x[ix[half:]].mean(0)))
        groups={s:x[[r[1]==s for r in rows]] for s in ['U','L27']}
        summary[name]=dict(parents=len(rows),raw_norm=float(np.linalg.norm(d)),
            long_steps=sum(len(s['long_step_indices']) for s in selected[name]),
            short_steps=sum(len(s['short_step_indices']) for s in selected[name]),
            confidence_question_mean=[float(np.mean([s[k] for s in selected[name]])) for k in ['long_confidence','short_confidence']],
            loo_cos_min=float(loo.min()),loo_worst_question=rows[int(loo.argmin())][0],
            split_cos_quantiles=np.quantile(splits,[0,.05,.5,.95,1]).tolist(),
            source_counts={s:len(z) for s,z in groups.items()},
            source_mean_cos=cosine(groups['U'].mean(0),groups['L27'].mean(0)),
            source_norms={s:float(np.linalg.norm(z.mean(0))) for s,z in groups.items()})
    for name in summary:
        summary[name]['cosine_with_old_selection']=cosine(means[name],means['old_selection'])
    result=dict(status='CPU diagnostics complete; no generation or GPU',source_archive_sha256=sha(raw),
        script_sha256=sha(Path(__file__).read_bytes()),rows_sha256=sha((old/'rows.json').read_bytes()),
        common_prefix=dict(pairs=len(prefix_rows),token_quantiles=np.quantile([r['common_tokens'] for r in prefix_rows],[0,.5,.9,1]).tolist(),
            selected_long_states=sum(r['old_long_common_states'] for r in prefix_rows),
            selected_short_states=sum(r['old_short_common_states'] for r in prefix_rows)),variants=summary)
    for name,obj in [('results',result),('question_selection',selected),('prefix_audit',prefix_rows)]:
        (out/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
