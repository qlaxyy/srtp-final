"""Select every frozen short trajectory contributing under-class states."""
import hashlib,json,tarfile,unicodedata
from prepare_length_vector import HERE,ROOT,read,save,sha

def main():
    train_path=ROOT/'sources/ReBalance/Data/Math_Train/test.jsonl'
    assert sha(train_path)=='4bd7f5266f6c5c4fc729da2d9cbefbe2453cf8df7bd6ba93dc02b9e11512cefd'
    train=[json.loads(s) for s in train_path.read_text(encoding='utf8').splitlines()]
    audit=read(HERE/'trajectory_length_labels_20260918_run1/result.json')
    with tarfile.open(ROOT/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz') as tf:
        raw=tf.extractfile('generations.jsonl').read()
    assert hashlib.sha256(raw).hexdigest()==audit['source_sha256']
    rows=[json.loads(s) for s in raw.splitlines()];mean=audit['variants']['thinking']['mean'];selected=[]
    for r,s in zip(rows,audit['questions']):
        assert r['train_index']==s['train_index']
        if s['thinking_tokens']>=mean or not s['under_original']:continue
        gold=train[r['train_index']]
        norm=lambda x:''.join(unicodedata.normalize('NFKC',x).split())
        assert norm(gold['problem'])==norm(r['problem'])
        assert hashlib.sha256(norm(gold['problem']).encode()).hexdigest()==s['problem_sha256']
        selected.append(dict(train_index=r['train_index'],problem_sha256=s['problem_sha256'],
            thinking_tokens=s['thinking_tokens'],under_steps=s['under_original'],
            text=r['text'],text_sha256=hashlib.sha256(r['text'].encode()).hexdigest(),gold_row=gold))
    assert len(selected)==166 and sum(r['under_steps'] for r in selected)==854
    refs=read(HERE/'length_sign_ablation_20260918_run1/historical_compact.json')
    target=HERE/'length_state_alignment_20260918_run1/under_parent_input.json'
    save(target,dict(scope='CPU grading of existing calibration answers only; no generation or benchmark selection',
        source_sha256=audit['source_sha256'],train_sha256=sha(train_path),grader_sha256=refs['grader_sha256'],rows=selected))
    print(json.dumps(dict(input=str(target),sha256=sha(target),questions=len(selected),under_steps=854)))

if __name__=='__main__':main()
