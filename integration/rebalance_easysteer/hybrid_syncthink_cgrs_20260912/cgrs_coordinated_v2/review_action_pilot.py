"""Independent archive integrity, paired lengths, masks and label review."""
import collections,hashlib,json,tarfile
from prepare_length_vector import HERE,read,save,sha
def main():
    run='counterfactual_labels_20260918_run1';p=HERE.parents[3]/('.codex_work/'+run+'.completed.tar.gz')
    assert sha(p)=='479dc20cc6ff0fcf7b0bdd8d07c048430aaf545c587775e75adebf2207c28bc5'
    with tarfile.open(p) as t:
        def raw(n):return t.extractfile(run+'/'+n).read()
        def get(n):return json.loads(raw(n))
        g=get('graded.json');assert g['plan_sha256']==sha(HERE/run/'plan.json')
        arms={n:get(n+'_complete.json') for n in ('apply_a','skip')}
        for n,h in g['source_sha256'].items():assert hashlib.sha256(raw(n+'_complete.json')).hexdigest()==h
        a=get('apply_a_prefill_masks.json');b=get('skip_prefill_masks.json')
        assert len(a)==len(b)==8
        for x,y in zip(a,b):assert x[:-1]==y[:-1] and x[-1]!=0 and y[-1]==0
    plan=read(HERE/run/'plan.json');counts=collections.Counter();summaries={}
    for i,r in enumerate(g['rows']):
        assert r['train_index']==plan['rows'][i]['train_index'] and r['problem_sha256']==plan['rows'][i]['problem_sha256']
        prefixes=[]
        for n in arms:
            x=arms[n][str(i)];prefixes.append(x['prefix_token_ids']);full=x['prefix_token_ids']+x['token_ids'];s=r['arms'][n]
            assert x['train_index']==r['train_index'] and len(full)==x['total_tokens']==s['total']<=16000
            assert (full.index(151649) if 151649 in full else len(full))==x['thinking_tokens']==s['thinking']
            assert x['control']['tokens']==len(full)
        assert prefixes[0]==prefixes[1]
        ap,sk=[r['arms'][n] for n in ('apply_a','skip')]
        label='inconclusive'
        if all(x['finish_reason']=='stop' and x['closed'] for x in (ap,sk)):
            if ap['correct'] and sk['correct']:
                if ap['total']<sk['total'] and ap['thinking']<sk['thinking']:label='apply_shorter_both_correct'
                elif sk['total']<ap['total'] and sk['thinking']<ap['thinking']:label='skip_shorter_both_correct'
                elif ap['total']==sk['total'] and ap['thinking']==sk['thinking']:label='tie_both_correct'
                else:label='mixed_length_both_correct'
            elif ap['correct']!=sk['correct']:label='apply_only_correct' if ap['correct'] else 'skip_only_correct'
        assert label==r['label'];counts[label]+=1
    assert dict(counts)==g['counts']
    for n in arms:
        xs=[r['arms'][n] for r in g['rows']]
        summaries[n]=dict(correct=sum(x['correct'] for x in xs),total=sum(x['total'] for x in xs),thinking=sum(x['thinking'] for x in xs),caps=sum(x['total']==16000 for x in xs))
    result=dict(archive_sha256=sha(p),local_checks_passed=True,counts=dict(counts),summaries=summaries,rows=g['rows'],completion=g['completion'],grading_seconds=g['grading_seconds'],
        decision='Five directional labels among eight early negative-action states are insufficient for a new vector. No positive-state coverage or original calibration feature alignment. Do not fit or automatically enlarge this label-collection pilot.',
        correctness_source='Frozen author parser/grader labels verified against saved identities, not independent manual proofs.')
    save(HERE/run/'verified_result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':main()
