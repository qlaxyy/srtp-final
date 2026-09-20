"""Archive CPU results and pending experiment specification; never launch GPU."""
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT/'.codex_work/label_alignment_20260917/run6'
OUT = HERE/'label_alignment_20260917'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, v):
    with p.open('x', encoding='utf8') as f:
        json.dump(v, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    assert json.loads((SOURCE/'complete.json').read_text())['inputs_unchanged']
    OUT.mkdir(exist_ok=False)
    for name in ('report.json', 'inputs.json', 'data_usage.json', 'complete.json'):
        with (OUT/name).open('xb') as f:
            f.write((SOURCE/name).read_bytes())
    report = json.loads((SOURCE/'report.json').read_text())
    variants = report['variants']
    assets = {}
    for name in ('T14', 'CV'):
        folder = OUT/name
        folder.mkdir()
        for file in ('auto_vector.pt', 'fit.json'):
            with (folder/file).open('xb') as f:
                f.write((SOURCE/name/file).read_bytes())
            assets[f'{name}/{file}'] = sha(folder/file)
    save(OUT/'plan.json', dict(run_id='label_alignment_1p5b_train200_v1_20260917',
        status='CPU assets prepared; native adapters and new data registration pending; not GPU-ready',
        model='DeepSeek-R1-Distill-Qwen-1.5B', dataset='MATH train', proposed_unique_questions=200,
        seed=42, max_new_tokens=16000, temperature=.7, top_p=.95,
        arms=[
            dict(name='R', calibration='frozen L27', controller='original confidence+variance', suppression='off'),
            dict(name='RC14', calibration='frozen L27', controller='original confidence+variance', suppression='original T14'),
            dict(name='L27_L27', calibration='frozen L27', controller='original confidence+variance', suppression='L27 opening phrase completion'),
            dict(name='T14_T14', calibration='T14 vector and LDA fit', controller='original confidence+variance', suppression='original T14'),
            dict(name='CV_CV', calibration='confidence+variance vector and LDA fit', controller='original confidence+variance', suppression='original T14'),
            dict(name='L27_L27_control', calibration='frozen L27', controller='lexical+confidence reference', suppression='original T14')],
        new_answers_if_no_matching_references=1200,
        comparison='Each candidate versus RC14 and R on exactly the same new questions; not a synergy test',
        selection='Screen only; candidate selection and independent confirmation must use disjoint registered data',
        data_status='No new IDs reserved. Reconcile local/remote usage and A-line reservations before freezing.',
        assets_sha256=assets,
        runtime_target=dict(engine='Existing vLLM/EasySteer; no environment installation',
            dtype='bfloat16', max_num_seqs=128, max_num_batched_tokens=32768,
            gpu_memory_utilization=.90, async_scheduling='Only after native acceptance tests',
            prefix_caching=False, chunked_prefill=False,
            context='Round actual largest prompt + 16000 up to 512; no truncation'),
        inference_order=['Native R observes raw pre-penalty max probability.',
            'R updates coefficient after accepting a complete step.',
            'Optional lexical controller uses only that completed step; preserve prefill and initial behavior.',
            'At next clean opening, apply penalty only when current coefficient is negative.',
            'Subtract log(2) before temperature/top-p; no claim final probability halves.'],
        lexical_control_spec=dict(rule='O=lexical OR c<q25; U=not lexical AND c>q75; else middle',
            coefficients='O=low_val_2; U=high_val_2; middle=original confidence baseline',
            warning='New hard-gated controller, not an equivalent curve refit; no author formula implied',
            reset='Per request; clear lexical prefix/hit at each accepted boundary and think end; reject preemption until validated'),
        large_lexicon_spec=dict(policy='Same author 27 regex entries including morphology; clean opening only',
            phrases='Penalize a completing token, not every constituent token; do not blanket-penalize I/Let/make',
            limitation='Calibration detects anywhere in a complete step; suppression only prospectively at opening. Same inventory is not identical observation scope.',
            integration='Compile finite-state transitions on CPU; device state per request. CPU reference is implemented, native hook pending.'),
        required_before_gpu=['Data reconciliation and exact row hashes', 'Pinned model/runtime/source hashes',
            'Instance-only native adapter with default-off exact behavior; do not fork the shared controller',
            'CPU phrase automaton versus regex oracle including token segmentation and resets',
            'Native short-prefix engineering checks before efficacy; do not tune on them'],
        expected_time='Provisional 10-25 minutes generation for 1200 answers on prior 4090D profile; add native engineering and grading; not a measured estimate',
        hard_stop_seconds_per_arm=600, batch_generation_hard_stop_seconds=3600,
        stop_conditions=['Any input hash mismatch, prefix-history gate failure, NaN, unsupported preemption, OOM, or time cap: preserve partials and stop; no automatic retries'],
        decision=dict(accuracy_margin_pp=-2, margin_reference='R; also report RC14 difference',
            observed_candidate='Both mean thinking and total tokens lower than RC14; cap count not higher; observed accuracy difference versus R >= -2pp',
            uncertainty='20000 paired question bootstraps for token ratios and paired accuracy interval; label unconfirmed if interval crosses accuracy margin or zero compression',
            report=['accuracy','thinking tokens','total tokens','caps','wrong-to-right/right-to-wrong',
                    'generation time','new forward passes','judgment/adapter overhead','all partial/failure costs'],
            multiplicity='Four exploratory candidates; no confirmed superiority claim from choosing the best; independent confirmation required'),
        ssh_status='20403 handshake failed; no remote process started',
        source_parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    save(OUT/'cpu_checks.json', dict(tests_passed=5, original_vector_elementwise_equal=True,
        original_parameters_max_abs_error=report['original_parameters_max_abs_error'],
        grid_points_per_fit=125751, fits=3, inputs_unchanged=True,
        attempts='run1-4 preserve CPU assertions for cross-platform FP64 exact equality; run5 succeeded; run6 recomputes cosine in FP64 to avoid values slightly above one. No GPU used.',
        raw_run=str(SOURCE), local_sources_sha256={n:sha(HERE/n) for n in
            ('prepare_label_alignment.py','test_label_alignment.py','large_lexicon_reference.py')}))
    p = ROOT.parent/'srtp-final/docs/research/00-研究交接.md'
    raw = p.read_bytes()
    marker = '**B线词表／校准标签对齐消融：CPU资产已完成，尚无生成结果（2026-09-17）。**'
    assert marker.encode() not in raw
    text = f'''\n\n{marker}

用户要求比较统一27项词表／14-token表，以及统一置信度＋方差／词表＋置信度信号。先纠正机制：当前向量提取与LDA强度拟合本来使用同一混合标签；置信度＋方差是在线控制输入，不是另一次拟合分类。14项实际为token变体，列表在CGRS附录A表3；L27含多token短语且无Hmm，不是T14的严格超集。

复用原500题、84008步骤和layer_21隐藏特征，无模型加载、新回答或GPU。原向量逐元素复现；参数复算最大差4.44e-16，正式对照继续用原文件。原L27类数48251/11309；T14类数47985/11427，方向余弦0.999809、范数比1.004779；CV联合类数12778/14087，方向余弦0.439179、范数比1.136782。三个125751点曲面均有限。以上是校准几何，不是效果比较。保存源文件哈希、校准题号／题面哈希及所有失败CPU断言；原资产未改。

新增大词表短语完成匹配CPU参考及词表控制参考，5项检查通过。词表控制是明确的新硬门控设计：混合O组取low_val_2，U组取high_val_2，其余保留置信度基础曲线；不能冒充作者曲线。原生设备状态／异步接入尚未完成，不是GPU就绪。大词表只在短语完成时惩罚，禁止简单降低I/Let等组成token。保留原clean opening范围，校准全步骤检测与推理开头抑制并非完全同一观察范围。

候选比较规格为1.5B、新MATH训练200题、seed42、每题16000、R/RC14及四候选共6组（最多1200答案）；尚未冻结新题号，先对账，不能用冻结测试集挑参数。准确率相对R下降最多2个百分点，同时报告相对RC14；思考/总token、触顶、逐题翻转、区间与开销全部记录。预估纯生成10–25分钟仅是旧配置估算，原生工程验收另计。当前20403 SSH握手失败，未启动远程作业。计划、CPU结果及T14/CV新资产位于B线`integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912/cgrs_coordinated_v2/label_alignment_20260917/`；原始CPU结果在`.codex_work/label_alignment_20260917/run6`。没有新增Markdown报告。
'''
    # Insert after heading, retaining every original byte.
    pos = raw.find(b'\n') + 1
    insert = text.replace('\n','\r\n').encode('utf8')
    new = raw[:pos] + insert + raw[pos:]
    assert new[:pos] + new[pos+len(insert):] == raw
    p.write_bytes(new)
    save(OUT/'handoff_receipt.json',dict(path=str(p),before_sha256=hashlib.sha256(raw).hexdigest(),
        after_sha256=sha(p),prior_bytes_preserved=True))
    print('Recorded CPU assets, pending plan, and existing handoff.')


if __name__ == '__main__':
    main()
