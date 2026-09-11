"""One CPU replay of the experimental complete-step applied-scale gate."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

from audit_control_alignment import load_controller, scalar_reference
from replay_cycle_monitor import token_bytes

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'integration/rebalance_easysteer/eval'))
from repeat_positive_gate import CompleteStepRepeat, applied_coefficient, byte_pieces

PLAN = dict(date='2026-09-11', purpose='Implement and inspect a complete-step positive gate on saved unsteered traces, not measure its generation effect.',
    sample='All original 1.5B 500 MATH training calibration answers, greedy, seed42, cap16000; all capped/incorrect outputs retained.',
    units='Close a unit only on the original runtime boundary token ID. Keep every byte including the boundary token (which can contain math symbols or the next paragraph opening). Strip outer ASCII whitespace only; ignore empty units. Thus units differ explicitly from the earlier byte-paragraph monitor. No artificial close at EOF or </think>.',
    repetition='Use the shortest exact adjacent repeated block of complete units. Confirm only after two copies are complete; extend while later complete units follow the same period, release on mismatch. No length/count/similarity threshold search.',
    action='At original ready boundaries, if repetition holds, apply min(original coefficient,0). Preserve original confidence, variance and controller state. Off/shadow return the original scale. No new injection position or forced ending.',
    numeric='CPU float64 projection of the frozen controller. Actual action uses alpha>0; additionally report alpha>1e-6 using the earlier numeric tolerance, not a detector threshold.',
    review='Inspect all previously stopped counterexamples 274,348,469 and the atom false-action case74. Inspect all other stopped answers with any positive cancellation if <=12, otherwise a fixed seed20260911 sample of12. Join all76 previous reviewed checkpoints without relabeling them.',
    limitations=['All500 are exposed development data, not held-out validation.',
        'Text repetition is not semantic overthinking truth. A normal stop or cap is not a gold label.',
        'Projected applied-scale differences are not changed outputs, preserved accuracy or measured speedup.',
        'The runtime bridge is opt-in, single-process V2 only, and adds host/device synchronization. CPU contract tests do not validate CUDA, graph replay or async throughput.'],
    gpu_tasks=0,new_generations=0,model_forwards=0)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def run(assets, output):
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();save(output/'plan.json',PLAN)
    config=assets/'integration/rebalance_easysteer/configs'
    frozen=read(config/'final_results_20260909.json')
    fit=read(config/'auto_code_v2_1p5b_20260908.json')
    old=read(config/'cycle_monitor500_20260911.json')
    atom=read(config/'cycle_atoms500_20260911.json')
    alignment=read(config/'calibration_control_alignment30_20260910.json')
    evidence=read(config/'calibration_label_evidence30_20260910.json')
    path=assets/'.codex_work/label_audit_30_20260910/tokenizer.json'
    assert sha(path)==old['provenance']['tokenizer_sha256']
    pieces=byte_pieces(path)
    assert pieces==token_bytes(read(path))
    p=SimpleNamespace(**fit['parameters'])
    params=frozen['benchmarks'][0]['protocol']['dynamic_params']
    assert all(params[k]==v for k,v in fit['parameters'].items())
    boundaries=set(params['boundary_token_ids']);end=params['think_end_token_id']
    assert boundaries=={i for i,b in pieces.items() if b'\n\n' in b}
    runtime_path='sources/EasySteer/vllm-steer/vllm/steer_vectors/rebalance.py'
    blob=subprocess.check_output(['git','show','HEAD:'+runtime_path],cwd=ROOT)
    assert hashlib.sha256(blob).hexdigest()==frozen['source_sha256'][runtime_path]
    assert blob.decode().replace('\r\n','\n')==(ROOT/runtime_path).read_text(encoding='utf-8')
    fn=load_controller();constants=fn['_curve_constants'](p.q25c,p.q75c,p.low_val_1,'cpu','float64',p.curve_tau)
    records,details=[],[];source_hash=hashlib.sha256()
    with tarfile.open(assets/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz') as archive:
        manifest=json.load(archive.extractfile('manifest.json'))
        assert manifest['count']==500 and manifest['temperature']==0 and manifest['max_tokens']==16000
        for index,line in enumerate(archive.extractfile('generations.jsonl')):
            source_hash.update(line);raw=json.loads(line);ids=raw['token_ids']
            assert raw['train_index']==old['records'][index]['train_index']==manifest['train_indices'][index]
            assert len(ids)==old['records'][index]['total_tokens']<=16000
            monitor=CompleteStepRepeat(boundaries,params['think_start_token_id'],end)
            probabilities=[];previous=None;rows=[];first=None;expected_units=[];unit_start=0
            for n,token in enumerate(ids,1):
                event=monitor.observe(token,pieces[token])
                if token==end: break
                if token==params['think_start_token_id']: unit_start=n
                if token not in boundaries:
                    probabilities.append(math.exp(raw['logprobs'][n-1]));continue
                text=b''.join(pieces[t] for t in ids[unit_start:n]).strip()
                unit_start=n
                if text: expected_units.append(text)
                if not probabilities:
                    assert event is None or not event['eligible_boundary']
                    continue
                confidence=math.fsum(probabilities)/len(probabilities)
                variance=0. if previous is None else (confidence-previous)**2/4
                coefficient=scalar_reference(confidence,variance,p,constants)['coefficient']
                repeat=event is not None
                assert not repeat or event['eligible_boundary']
                applied=applied_coefficient(coefficient,repeat,'cancel_positive')
                assert applied_coefficient(coefficient,repeat,'shadow')==coefficient
                assert applied==coefficient or repeat and coefficient>0 and applied==0
                row=dict(boundary_token_1based=n,confidence=confidence,variance=variance,
                    original_coefficient=coefficient,candidate_applied_coefficient=applied,
                    repeat=repeat,would_cancel=applied!=coefficient,event=event)
                rows.append(row)
                if repeat and first is None:
                    start,period=event['start_unit'],event['period']
                    first=dict(**event,blocks=[[s.decode('utf-8',errors='replace') for s in
                        monitor.units[start+k*period:start+(k+1)*period]] for k in range(2)])
                previous=confidence;probabilities=[]
            assert monitor.units==expected_units
            positive=[r for r in rows if r['would_cancel']]
            record=dict(calibration_index=index,train_index=raw['train_index'],exposure_group=old['records'][index]['exposure_group'],
                finish_reason=raw['finish_reason'],total_tokens=len(ids),runtime_boundaries=len(rows),
                original_positive_boundaries=sum(r['original_coefficient']>1e-6 for r in rows),
                completed_units=len(monitor.units),repeat_boundaries=sum(r['repeat'] for r in rows),
                first_repeat=first,would_cancel_count=len(positive),
                would_cancel_above_tolerance=sum(r['original_coefficient']>1e-6 for r in positive),
                first_cancel=positive[0] if positive else None)
            records.append(record)
            details.append(dict(**record,problem=raw['problem'],text=raw['text'],boundaries=rows,
                units=[u.decode('utf-8',errors='replace') for u in monitor.units],unit_token_spans=monitor.spans,
                unit_sha256=[hashlib.sha256(u).hexdigest() for u in monitor.units]))
    assert len(records)==500 and source_hash.hexdigest()==fit['calibration_source_sha256']==old['provenance']['source_sha256']
    assert sum(r['runtime_boundaries'] for r in records)==sum(r['runtime_boundaries'] for r in atom['records'])
    assert sum(r['original_positive_boundaries'] for r in records)==sum(r['original_positive_boundaries'] for r in atom['records'])
    by_question={q['question']:q for q in evidence['questions']};joined=[]
    by_case={p['case_id']:p for q in evidence['questions'] for p in q['checkpoints']}
    for point in alignment['checkpoints']:
        q=by_question[point['question']];source=by_case[point['case_id']]
        boundary=next((r for r in details[q['calibration_index']]['boundaries']
                       if r['boundary_token_1based']==source['stop']+1),None)
        if boundary is not None:
            for field in ('confidence','variance'):
                assert abs(boundary[field]-point['outgoing'][field])<1e-12
            assert abs(boundary['original_coefficient']-point['outgoing']['coefficient'])<1e-12
        else:
            assert point['outgoing']['coefficient'] is None
        joined.append(dict(**{k:point[k] for k in ['case_id','question','train_index','step','progress_evidence','math_validity']},boundary=boundary))
    counts={}
    for group in ('all500','reviewed30','boundary20','remaining450'):
        subset=[r for r in records if group=='all500' or r['exposure_group']==group]
        counts[group]={reason:dict(count=sum(r['finish_reason']==reason for r in subset),
            repeat_questions=sum(r['finish_reason']==reason and r['first_repeat'] is not None for r in subset),
            positive_gate_questions=sum(r['finish_reason']==reason and r['would_cancel_count']>0 for r in subset),
            positive_gate_positions=sum(r['would_cancel_count'] for r in subset if r['finish_reason']==reason))
            for reason in ('stop','length')}
    known=[74,274,348,469]
    new_stopped=[r['calibration_index'] for r in records if r['finish_reason']=='stop' and r['would_cancel_count'] and r['calibration_index'] not in known]
    selected=sorted(random.Random(20260911).sample(new_stopped,min(12,len(new_stopped))))
    summary=dict(status='completed_cpu_shadow_gate_pending_review',plan=PLAN,
        execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        question_count=500,new_generations=0,model_forwards=0,gpu_tasks=0,
        cpu_seconds=time.perf_counter()-started,outcome_cross_tab=counts,records=records,reviewed_checkpoints=joined,
        review_selection=dict(known_counterexamples=known,new_stopped_positive=selected,all_new_stopped_positive_count=len(new_stopped)),
        verification=dict(token_byte_entries_compared=len(pieces),complete_units_independently_checked=True,
            original_boundary_count=sum(r['runtime_boundaries'] for r in records),
            original_positive_count=sum(r['original_positive_boundaries'] for r in records),
            original_checkpoint_positions_checked=len(joined)),
        provenance=dict(source_sha256=source_hash.hexdigest(),tokenizer_sha256=sha(path),
            script_sha256=sha(__file__),adapter_sha256=sha(ROOT/'integration/rebalance_easysteer/eval/repeat_positive_gate.py'),
            frozen_runtime_sha256=hashlib.sha256(blob).hexdigest(),
            input_reports={name:sha(config/name) for name in ['final_results_20260909.json','auto_code_v2_1p5b_20260908.json',
                'cycle_monitor500_20260911.json','cycle_atoms500_20260911.json','calibration_control_alignment30_20260910.json','calibration_label_evidence30_20260910.json']}))
    save(output/'summary.json',summary);save(output/'details.json',details)
    print(json.dumps({k:summary[k] for k in ['outcome_cross_tab','review_selection','verification','cpu_seconds']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.assets.resolve(),args.output.resolve())
