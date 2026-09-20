# -*- coding: utf-8 -*-
"""Pin CPU-validated runtime package and unchanged historical references."""
import hashlib,json,subprocess,sys
from pathlib import Path
from prepare_label_alignment import save,sha

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
OUT=HERE/'label_alignment_20260917'


def read(p):return json.loads(p.read_text(encoding='utf8'))


def main():
    tests=['test_label_alignment.py','test_lexicon_automaton.py','test_alignment_device_cpu.py','test_native.py','test_phrase_cpu.py']
    checks={}
    for name in tests:
        r=subprocess.run([sys.executable,'-X','utf8',str(HERE/name)],capture_output=True,text=True)
        checks[name]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
        assert r.returncode==0,checks[name]
    tables=ROOT/'.codex_work/label_alignment_20260917/tables_run2'
    meta=read(tables/'complete.json');assert meta['calibration_steps_checked']==84008
    for name,h in meta['source_sha256'].items():assert sha(HERE/name)==h,name
    dst=OUT/'tables';dst.mkdir(exist_ok=False)
    for name in ('search.npz','opening.npz','complete.json'):
        with (dst/name).open('xb') as f:f.write((tables/name).read_bytes())
    native=read(next((ROOT/'.codex_work/wsc_native_long_complete_20260917/results').rglob('engineering_gate.json')).parent/'plan.json')
    assert len(native['rows'])==8
    prior=next((ROOT/'.codex_work/cgrs_v2_full_run1_verified').rglob('resolved_plan.json'))
    old=read(prior);rows=old['datasets']['math_test']['rows']
    frozen=ROOT.parent/'srtp-final/.codex_work/auto_code_v2_500_20260908'
    ev=read(frozen/'math500_eval.json');grades=read(frozen/'math500_author_grading.json')
    assert grades['input_sha256']==sha(frozen/'math500_eval.json')
    groups={}
    for arm,key in [('U','baseline'),('R','rebalance_dynamic')]:
        compact=[]
        for i,(rec,label,row) in enumerate(zip(ev[key]['records'],grades['groups'][key]['records'],rows)):
            assert rec['dataset_index']==label['index']==i and rec['problem']==row['problem']
            assert len(rec['token_ids'])==rec['tokens']
            compact.append(dict(dataset_index=i,problem_sha256=row['problem_sha256'],correct=label['author_correct'],tokens=rec['tokens'],thinking_tokens=rec['thinking_tokens']))
        assert len(compact)==500
        groups[arm]=dict(records=compact,generation_seconds=old['frozen_benchmarks']['math_test']['groups'][key]['generation_seconds'])
    rcpath=prior.parent/'math_test/RCnegative/result.json';rc=read(rcpath)
    lp=rcpath.parent/'author_partial.jsonl';labels=[json.loads(x) for x in lp.read_text(encoding='utf8').splitlines()]
    assert len(labels)==len(rc['records'])==500
    recs={r['dataset_index']:r for r in rc['records']};compact=[]
    for i,(label,row) in enumerate(zip(labels,rows)):
        rec=recs[i];assert label['dataset_index']==i and rec['problem_sha256']==label['problem_sha256']==row['problem_sha256']
        assert hashlib.sha256(rec['text'].encode()).hexdigest()==label['text_sha256']
        compact.append(dict(dataset_index=i,problem_sha256=row['problem_sha256'],correct=label['correct'],tokens=rec['tokens'],thinking_tokens=rec['thinking_tokens']))
    groups['RC14']=dict(records=compact,generation_seconds=rc['generation_seconds'])
    save(OUT/'historical_compact.json',dict(groups=groups,
        grader_sha256={k:grades[k+'_sha256'] for k in ('parser.py','grader.py')},
        original_files={str(p):sha(p) for p in [prior,rcpath,lp,frozen/'math500_eval.json',frozen/'math500_author_grading.json']}))
    files=['adapter.py','policy.py','label_alignment_adapter.py','lexicon_automaton.py',
        'run_label_alignment.py','grade_label_alignment.py','prepare_label_alignment.py',
        'review_wsc_native_cpu.py','audit_repeat_continuations_cpu.py']
    release=dict(status='CPU integration complete; SSH unavailable; native engineering pending',
        plan_sha256=sha(OUT/'plan_v2_math500.json'),table_schema='compact-token-classes-v1',
        assets=native['assets'],engineering_rows=native['rows'],
        engineering_scope='Already exposed 8 training questions; RC14, extension-off and five candidates; 512 tokens each; 56 prefixes maximum; no grading or parameter search',
        full_scope='After native gate only: five new candidates x MATH500, seed42; reuse U/R/RC14',
        runtime_root='/root/autodl-tmp/projects/hybrid_wsc_native_long_20260917',
        runtime_source_sha256={n:h for n,h in native['source_sha256'].items() if n.startswith(('sources/','integration/rebalance_easysteer/eval/'))},
        source_sha256={n:sha(HERE/n) for n in files},
        artifact_sha256={str(p.relative_to(OUT)).replace('\\','/'):sha(p) for p in
            [OUT/'historical_compact.json',OUT/'tables/search.npz',OUT/'tables/opening.npz',OUT/'tables/complete.json',
             OUT/'T14/auto_vector.pt',OUT/'T14/fit.json',OUT/'CV/auto_vector.pt',OUT/'CV/fit.json']},
        actual_gpu_generation=False,tests=checks,table_audit=meta,
        commands=dict(engineering='python run_label_alignment.py --release label_alignment_20260917/release.json --runtime-root /root/autodl-tmp/projects/hybrid_wsc_native_long_20260917 --phase engineering --output /root/autodl-tmp/results/easysteer/hybrid_syncthink_cgrs_20260912/label_alignment_math500_v2_20260917/engineering_run1',
            full='python run_label_alignment.py --release label_alignment_20260917/release.json --runtime-root /root/autodl-tmp/projects/hybrid_wsc_native_long_20260917 --phase full --engineering-result /root/autodl-tmp/results/easysteer/hybrid_syncthink_cgrs_20260912/label_alignment_math500_v2_20260917/engineering_run1 --output /root/autodl-tmp/results/easysteer/hybrid_syncthink_cgrs_20260912/label_alignment_math500_v2_20260917/full_run1',
            grade='python grade_label_alignment.py --runtime-root /root/autodl-tmp/projects/hybrid_wsc_native_long_20260917 --run /root/autodl-tmp/results/easysteer/hybrid_syncthink_cgrs_20260912/label_alignment_math500_v2_20260917/full_run1'))
    save(OUT/'release.json',release)
    p=ROOT.parent/'srtp-final/docs/research/00-研究交接.md';raw=p.read_bytes()
    text='''\n\n**B线词表对齐原生接入已完成CPU准备，SSH仍失败，未启动GPU（2026-09-17）。**

新增有限状态机、实例级AlignmentAdapter和固定批次runner/判分入口。原Adapter仅增加可选回调，未启用时保留原路径；共享vLLM控制器没有改动。大词表惩罚只在干净步骤开头的词／短语完成处应用；词表控制在原生observe_sample完成后、_record_scales写入历史之前替换已完成步骤系数。状态在新请求／边界／think结束重置，拒绝抢占，不逐token回传CPU。

匹配器全量核验原84008校准步骤：完整文本与逐token流均复现作者词表标签。修复Unicode大写带点I先lower再regex的差异，失败测试及初版表保留。235状态的表按等价token转移无损合并：搜索470类／1159145字节，开头466类／1155785字节，335个候选惩罚token；校准大表仍是27项，不把335叫335个词。22项CPU检查通过（原路径3、旧短语4、分组5、自动机5、设备状态6实际合计23项）；具体测试数量以release中输出为准。没有用CPU结果冒充真实GPU验证。

release.json固定源码、模型、向量、表与历史逐题对照哈希，配套生成、判分和20000次配对bootstrap入口。原生工程先用已暴露8训练题、512上限，原RC14／扩展关闭／五候选，最多56条短前缀；通过关闭路径逐token及控制历史、首次可能干预前前缀检查后才跑五组各500全量。正式仍并发256、异步、batch tokens32768、seed42、上限16000。20403本轮连接仍在SSH握手阶段失败，没有远程进程、GPU前向或新生成；恢复后首先核对固定运行时源码并执行工程门槛。完整状态在B线原label_alignment_20260917目录，无新增Markdown。
'''
    text=text.replace('22项CPU检查通过（原路径3、旧短语4、分组5、自动机5、设备状态6实际合计23项）','23项CPU检查通过（原路径3、旧短语4、分组5、自动机5、设备状态6）')
    pos=raw.find(b'\n')+1;insert=text.replace('\n','\r\n').encode();p.write_bytes(raw[:pos]+insert+raw[pos:])
    save(OUT/'release_handoff_receipt.json',dict(before_sha256=hashlib.sha256(raw).hexdigest(),after_sha256=sha(p),prior_bytes_preserved=True))
    print('Release prepared: 23 CPU tests, 84008 steps, five full candidates. No GPU run.')


if __name__=='__main__':main()
