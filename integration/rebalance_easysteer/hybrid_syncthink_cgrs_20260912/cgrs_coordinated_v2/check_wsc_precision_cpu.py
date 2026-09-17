"""Prospective precision diagnostic criterion; never relabel old BF16 gate."""
import argparse
import json
from pathlib import Path


def check(old,new):
    baseline={r['train_index']:r for r in old['partition']}
    assert set(baseline)=={5353,26}
    assert {r['train_index'] for r in new['partition']}==set(baseline)
    rows=[]
    for row in new['partition']:
        delta=row['max_score_difference'];previous=baseline[row['train_index']]['max_score_difference']
        rows.append(dict(train_index=row['train_index'],old_delta=previous,new_delta=delta,
            passed=bool(row['passed'] and delta<=1e-4 and delta<=previous/10)))
    return dict(cases=rows,numerical_convergence=all(r['passed'] for r in rows),
        original_bf16_gate='failed_unchanged',
        interpretation='Convergence supports precision/kernel-path sensitivity only; not semantic accuracy, BF16 deployment readiness, or compression gains.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True)
    p.add_argument('--new',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=check(json.loads(a.old.read_text()),json.loads(a.new.read_text()))
    with a.output.open('x',encoding='utf8') as f:json.dump(r,f,indent=2)
    if not r['numerical_convergence']:raise ValueError('No convergence; no automatic expansion')
