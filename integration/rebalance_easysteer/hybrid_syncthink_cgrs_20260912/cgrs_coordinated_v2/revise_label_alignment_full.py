"""User-requested full MATH-500 matrix; preserve the superseded train proposal."""
import hashlib
import json
from pathlib import Path
import unicodedata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE/'label_alignment_20260917'


def read(p):
    return json.loads(p.read_text(encoding='utf8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def phash(s):
    return hashlib.sha256(''.join(unicodedata.normalize('NFKC',s).split()).encode()).hexdigest()


def save(p,x):
    with p.open('x',encoding='utf8',newline='\n') as f:
        json.dump(x,f,ensure_ascii=False,indent=2)
        f.write('\n')


def main():
    old_path=next((ROOT/'.codex_work/cgrs_v2_full_run1_verified').rglob('resolved_plan.json'))
    old=read(old_path)
    rcpath=old_path.parent/'math_test/RCnegative/result.json'
    rc=read(rcpath)
    rows=old['datasets']['math_test']['rows']
    assert len(rows)==500 and len(rc['records'])==500 and rc['status']=='complete'
    hashes=[phash(r['problem']) for r in rows]
    assert len(set(hashes))==500
    for r in rows+rc['records']:
        assert phash(r['problem'])==r['problem_sha256']
    assert set(hashes)=={r['problem_sha256'] for r in rc['records']}
    source=ROOT/'sources/ReBalance/Data/Math_Math500/test.jsonl'
    originals=[json.loads(x) for x in source.read_text(encoding='utf8').splitlines()]
    assert hashes==[phash(r['problem']) for r in originals]
    frozen=ROOT.parent/'srtp-final/.codex_work/auto_code_v2_500_20260908'
    evalpath=frozen/'math500_eval.json'
    grading=frozen/'math500_author_grading.json'
    ev=read(evalpath)
    expected=old['frozen_benchmarks']['math_test']
    assert sha(evalpath)==expected['artifacts']['evaluation_sha256']
    assert sha(grading)==expected['artifacts']['grading_sha256']
    for k in ('seed','temperature','top_p','max_tokens','max_model_len'):
        assert ev['protocol'][k]==old['runtime'][k]
    assert old['runtime']['seed']==42
    plan=read(OUT/'plan.json')
    plan.update(run_id='label_alignment_1p5b_math500_v2_20260917',
        supersedes='plan.json: train200 proposal; preserved, not executed',
        status='Full benchmark scope fixed by user; native adapters pending, not launched',
        dataset='MATH-500 test', proposed_unique_questions=500, rows=rows,
        new_answers_if_no_matching_references=2500,
        reuse_only_arms=['U','R','RC14'],
        arms=[a for a in plan['arms'] if a['name'] not in ('R','RC14')],
        runtime_target=dict(old['runtime'],engine='existing vLLM/EasySteer',
            prefix_caching=False,chunked_prefill=False,gpu_memory_utilization=.90),
        comparison='Frozen configurations evaluated on same 500 test questions; reuse U/R/RC14 without regeneration',
        selection='User-directed exploratory full-test ablation, not an untouched independent confirmation. No threshold search on these outputs.',
        data_status='500 canonical source problems and both saved references verified; no new train reservation',
        required_before_gpu=['Pinned server model/runtime/source hashes and no conflicting GPU job',
            'Complete native L27 phrase and lexical-controller adapters with exact default-off path',
            'CPU state/reset/multi-token checks, then bounded native acceptance checks',
            'Verify reused grading identity and export paired per-question joins'],
        expected_time='Roughly 35-60 minutes for five 500-answer groups on prior 4090D settings; estimate only, native engineering separate',
        hard_stop_seconds_per_arm=1500,batch_generation_hard_stop_seconds=7500)
    plan['arms'].insert(2,dict(name='T14_L27',calibration='T14 vector and LDA fit',
        controller='original confidence+variance',suppression='L27 opening phrase completion'))
    assert len(plan['arms'])==5
    plan['decision']['multiplicity']='Five exploratory candidates; report all, no winner-selected confirmation claim.'
    plan['factorial_comparisons']={
        'lexicon_2x2':[['RC14','L27_L27'],['T14_T14','T14_L27']],
        'rows':'Calibration vocabulary L27/T14; columns inference suppression T14/L27',
        'extraction_effect_small_suppression':'T14_T14 versus RC14',
        'extraction_effect_large_suppression':'T14_L27 versus L27_L27',
        'suppression_effect_large_extraction':'L27_L27 versus RC14',
        'suppression_effect_small_extraction':'T14_L27 versus T14_T14',
        'interaction_tokens':'Mean(T14_L27)-Mean(T14_T14)-Mean(L27_L27)+Mean(RC14); paired question bootstrap, raw mean token scale',
        'interaction_accuracy':'Same difference-in-differences on correct fractions, percentage-point scale',
        'signal_controls':'CV_CV and L27_L27_control each compared with RC14, keeping T14 suppression fixed',
        'limitation':'Historical RC14 runtime/source differences may confound strict causal interaction; same seed alone does not prove numerical equivalence.'}
    plan['reference_files']={str(p):sha(p) for p in (old_path,rcpath,source,evalpath,grading)}
    save(OUT/'plan_v2_math500.json',plan)
    save(OUT/'full500_reference_audit.json',dict(count=500,unique_hashes=500,
        canonical_source_order_equal=True,rc14_question_hash_set_equal=True,
        original_eval_and_grading_sha_match=True,seed=42,
        sampling_keys_equal=['temperature','top_p','max_tokens','max_model_len'],
        rc14_generation_seconds=rc['generation_seconds'],runtime=old['runtime'],
        files=plan['reference_files'],new_generation=False,
        limitation='Current server runtime not checked; no assertion that seed implies identical execution.'))
    p=ROOT.parent/'srtp-final/docs/research/00-研究交接.md'
    raw=p.read_bytes()
    text='''\n\n**B线词表对齐实验改为全量MATH-500（用户更新，2026-09-17）。**

替代尚未执行的train200提案：1.5B，沿用seed42、temperature0.7、top_p0.95、max_tokens16000；无干预U、单ReBalance R、原RC14均只读复用旧结果。完整词表2×2为L27提取/T14抑制（旧RC14）、L27/L27、T14/T14、新补T14/L27。再单独比较CV校准＋原CV在线控制，以及原L27校准＋词表置信度在线控制；后二者固定T14抑制。共五个新候选，每个500题，总计2500份新答案，不自动扩到GSM8K或7B。

已核验旧RC14完整500条、题面规范化哈希与源题集顺序、原U/R evaluation和grading文件SHA，以及seed/采样/上限一致。旧RC14纯生成400.054秒，运行配置异步、并发256、batch tokens32768、上下文32768，将作为配置参照；同seed不保证不同运行时数值路径等价，历史差异须披露，未擅自重跑对照。四格按固定抑制表比较提取因素，按固定提取表比较抑制因素；交互在平均token的差中差及正确率百分点尺度报告。全量题已反复暴露，本轮是固定候选探索性消融，不称独立确认，不按结果重新调阈值。CPU新向量资产不变，原生接入仍待完成，尚未启动GPU。规格及只读核查见B线原目录plan_v2_math500.json和full500_reference_audit.json，旧plan.json保留为被替代提案。
'''
    pos=raw.find(b'\n')+1
    insert=text.replace('\n','\r\n').encode()
    p.write_bytes(raw[:pos]+insert+raw[pos:])
    save(OUT/'full500_handoff_receipt.json',dict(before_sha256=hashlib.sha256(raw).hexdigest(),after_sha256=sha(p),prior_bytes_preserved=True))
    print('Full500 matrix saved: five new arms; historical U/R/RC14 reused; seed42 verified.')


if __name__=='__main__':main()
