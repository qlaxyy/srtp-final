"""Prepare two endpoint/control hypotheses from frozen training feedback only."""
import ast,hashlib,io,json,tarfile
from pathlib import Path
from prepare_length_vector import HERE,read,save,sha


def main():
    run='outcome_endpoints_20260918_run1';out=HERE/run;out.mkdir(exist_ok=False)
    old=HERE/'self_feedback_train500_20260918_run1'
    assert sha(old/'labels.json')=='194fcb92205dd3a7275be0cbd9116ffa028d8ec57e64a245c826fbda1a5a1869'
    labels=read(old/'labels.json');plan=read(old/'plan.json')
    archive=HERE.parents[3]/'.codex_work/self_feedback_train500_20260918_evidence.tar.gz'
    assert sha(archive)=='cd38a0f003ed26d53037dbc9c74483e1943d8032bf150ae237e8cdb8c2b443ca'
    with tarfile.open(archive) as t:
        r=json.load(t.extractfile('self_feedback_train500_20260918_run1/merged/L27_L27/result.json'))['records']
    sources={'U':read(old/'baseline.json'),'L27':r};pairs=[];rows=[]
    for i,label in enumerate(labels):
        a,b=label['U'],label['L27'];la,lb=a['thinking_tokens'],b['thinking_tokens']
        if la==lb:continue
        ln,sn=('U','L27') if la>lb else ('L27','U');long,short=label[ln],label[sn]
        if not(long['closed'] and short['closed'] and long['correct']):continue
        if long['thinking_tokens']-short['thinking_tokens']<128 or 1-short['thinking_tokens']/long['thinking_tokens']<.1:continue
        kind='CC' if short['correct'] else 'CW'
        pairs.append(dict(question=i,kind=kind,long_source=ln,short_source=sn,problem_sha256=label['problem_sha256']))
        for side,source in [('long',ln),('short',sn)]:
            rec=sources[source][i];assert rec['problem_sha256']==label['problem_sha256']
            ids=rec['token_ids'];n=ids.index(151649)
            rows.append(dict(question=i,kind=kind,side=side,source=source,problem_sha256=label['problem_sha256'],
                prompt_token_ids=plan['rows'][i]['prompt_token_ids'],token_ids=ids[:n]))
    assert sum(p['kind']=='CC' for p in pairs)==231 and sum(p['kind']=='CW' for p in pairs)==8
    protocol=dict(status='Fixed hypotheses before feature collection or benchmark outcomes',
        arms={
          'EFFICIENT':dict(endpoints='Within-question CC long O candidates minus CC short efficient candidates; average question differences',
             control='Raw vector. Initial0. alpha(c)=-clip((q75c-c)/(q75c-q25c),0,1); q25 maps -1, q75 maps0. No variance boost, positive branch, or overshoot.'),
          'UNDER_REFIT':dict(endpoints='O from CC long candidates; U from CW short wrong candidates. Step-weighted class means, raw vector.',
             control='Original public-code adapted controller form, newly fit O/U LDA midpoint/max crossing, confidence/variance quantiles; initial-1, high2+.1. Midpoint is geometric, not guaranteed efficient reasoning.')},
        selection='Natural termination, long correct, thinking gap>=128 AND >=10% of long thinking length; CC231/CW8, no test outcomes used',
        features='Replay both sides with same unsteered BF16 HF SDPA full forward; decoder output20 step-first-token states; full-vocabulary RAW max probability at preceding causal position; not teacher-forced token probability',
        steps='Original blank-line token boundaries; arithmetic step max-prob means, adjacent-step (difference)^2/4; final natural partial step included',
        E_quantiles='All steps in CC long and short trajectories; original pooled-step weighting for quantiles',
        B_quantiles='All steps in CC long and CW short trajectories; pooled-step weighting',
        labels='O=lexicon OR c<q25, efficient/U=no lexicon AND c>q75, restricted to specified trajectory sides',
        gates='E at least30 CC pairs with nonempty O/E; B at least3 CW parent questions and20 U states; B always small-sample exploratory. Finite nonzero raw vectors, nondegenerate quantiles, valid native curve anchors/grid, no automatic fallback labels.',
        engineering='First4 CC pairs, truncate each thinking prefix to512 for8 replay rows; repeat same full-forward schedule exactly, alignment counts and finite checks. No regenerated answers.',
        evaluation='1.5B MATH500 two new arms, seed42 temperature.7 top_p.95 max16000; existing L27/R reused. Advance each only if both mean token types decrease vs L27, caps no higher, observed accuracy loss<=2pp vs both L27/R; report paired intervals, no retuning. Exposed benchmark, not independent confirmation.',
        limits='Feature collection<=1800s. GPU sampling announced before start. No multi-seed, original calibration regeneration, environment/shared-runtime edits.',
        confounds='Endpoint selection is an outcome proxy, not causal step labels. E and B compare complete vector/controller designs, not an isolated controller ablation. HF reconstruction differs numerically from online vLLM; no historical probability equivalence claimed.')
    save(out/'protocol.json',protocol);save(out/'pairs.json',pairs);save(out/'rows.json',rows)
    engineering=[dict(r,token_ids=r['token_ids'][:512]) for r in rows[:8]];save(out/'engineering.json',engineering)
    (out/'.gitignore').write_text('rows.json\nfeatures/\npackage.tar.gz\n');(out/'.gitattributes').write_text('*.json -text\n*.pt binary -text\n*.npz binary -text\n')
    prior=read(HERE/'self_feedback_math500_20260918_run1/release.json')
    files={n:(HERE/n).read_bytes().replace(b'\r\n',b'\n') for n in ['collect_outcome_endpoints.py','fit_outcome_endpoints.py']}
    for n,b in files.items():ast.parse(b,filename=n)
    author=Path('E:/srtp/srtp-final/sources/ReBalance/hidden_analysis_auto.py')
    files['author_labels.py']=author.read_bytes().replace(b'\r\n',b'\n')
    release=dict(assets=prior['assets'],artifact_sha256={n:sha(out/n) for n in ['rows.json','pairs.json','engineering.json','protocol.json']},
        source_sha256={n:hashlib.sha256(b).hexdigest() for n,b in files.items()},label_sha256=sha(old/'labels.json'))
    save(out/'release.json',release)
    for n in ['rows.json','pairs.json','engineering.json','protocol.json','release.json']:files[n]=(out/n).read_bytes()
    with tarfile.open(out/'package.tar.gz','w:gz') as t:
        for n,b in files.items():
            info=tarfile.TarInfo(n);info.size=len(b);t.addfile(info,io.BytesIO(b))
    print(json.dumps(dict(rows=len(rows),thinking_tokens=sum(len(r['token_ids']) for r in rows),package_sha256=sha(out/'package.tar.gz'))))


if __name__=='__main__':main()
