"""Separate unbiased-within-question primary selection from enriched R-high subset."""
import argparse,json
from pathlib import Path
import numpy as np

def report(rows):
    valid=[r for r in rows if r['cgrs_certainty'] is not None]
    table={f'R{int(r)}_C{int(c)}':sum(x['rebalance_high']==r and x['cgrs_high']==c for x in valid) for r in (False,True) for c in (False,True)}
    return dict(total=len(rows),valid=len(valid),invalid=len(rows)-len(valid),table=table,
        both_high_fraction=table['R1_C1']/len(valid) if valid else None,
        C_high_given_R_high=table['R1_C1']/(table['R1_C0']+table['R1_C1']) if table['R1_C0']+table['R1_C1'] else None,
        R_high_given_C_high=table['R1_C1']/(table['R0_C1']+table['R1_C1']) if table['R0_C1']+table['R1_C1'] else None,
        C_high_R_positive=sum(x['cgrs_high'] and x['native_extend'] for x in valid),
        R_positive=sum(x['native_extend'] for x in valid),
        certainty_range=[min(x['cgrs_certainty'] for x in valid),max(x['cgrs_certainty'] for x in valid)] if valid else None)

def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args()
    complete=json.loads((a.folder/'complete.json').read_text());assert complete['status']=='complete'
    rows=[json.loads(s) for s in (a.folder/'probes.jsonl').read_text().splitlines()]
    assert len(rows)==174 and len({(r['train_index'],r['boundary_generated_index']) for r in rows})==174
    primary=[r for r in rows if r['sample_role']=='primary_question_balanced'];high=[r for r in rows if r['rebalance_high']]
    assert len(primary)==128 and len(high)==47
    ids=sorted({r['train_index'] for r in rows})
    result=dict(primary=report(primary),all_R_high=report(high),enriched_union_not_prevalence=report(rows),
        by_question={i:dict(primary=report([r for r in primary if r['train_index']==i]),R_high=report([r for r in high if r['train_index']==i])) for i in ids},
        interpretation_allowed=complete['interpretation_allowed'],scope='Eight exposed question clusters; no efficacy or correctness claims')
    # Rank correlation only on prespecified primary sample, never enriched union.
    valid=[r for r in primary if r['cgrs_certainty'] is not None]
    def ranks(v):
        v=np.array(v);return np.array([sum(v<x)+(sum(v==x)+1)/2 for x in v])
    if len(valid)>2:
        x=ranks([r['rebalance_confidence'] for r in valid]);y=ranks([r['cgrs_certainty'] for r in valid])
        result['primary_spearman']=float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None
    rng=np.random.default_rng(42);boots=[]
    for _ in range(5000):
        sampled=rng.choice(ids,len(ids),replace=True);rr=[r for i in sampled for r in primary if r['train_index']==i]
        boots.append(report(rr)['both_high_fraction'])
    result['primary_both_high_cluster_bootstrap_95']=np.quantile([b for b in boots if b is not None],[.025,.975]).tolist()
    result['bootstrap_warning']='Descriptive only: 8 clusters, 1 R-high primary point, unstable/degenerate intervals possible'
    result['secondary_interior_high_disagreements']=sum((r['boxed_interior_certainty']>.9)!=r['cgrs_high'] for r in rows if r['boxed_interior_certainty'] is not None and r['cgrs_high'] is not None)
    with (a.folder/'analysis.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='by_question'},indent=2))

if __name__=='__main__':main()
