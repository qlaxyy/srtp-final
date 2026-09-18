"""Persist the assistant's exposed qualitative judgments with exact excerpts."""
import json
from prepare_length_vector import HERE,read,save,sha

def main():
    out=HERE/'repeat_content_review_20260918_cpu';cases=read(out/'cases.json')['cases']
    judgments={
        2152:('uncertain','Wait, this is getting confusing. Maybe I should use a different approach.',
              '重复竖式出现在借位错误后的重新核验中；局部混有十进制/二进制混淆，不能认证删掉该重复可保留纠错。'),
        5201:('mechanical_restatement','But that seems too big',
              '重复相同缩放混淆和过大质疑，附近来回切换已出现的尺度，没有明确新增依据；不认证其数学正确性。'),
        5417:('verification_or_reuse','First, expand the first sum:',
              '第二次写同一多项式后展开括号并合并项，重复表达式承担后续代数操作的输入；不是单纯循环。'),
        1667:('verification_or_reuse','Now, if I square',
              '从原s平方等式重新开始，但随后平方交叉项之和并使用ab+ac+bc=11，是新的推导尝试；不以最终触顶否定这一局部推进。'),
        556:('mechanical_restatement','As I did earlier',
             '已得到2/9，随后重写相同模2定义及相同四格枚举；本地没有新增检查方法。核验意图存在，但内容重复。'),
        2430:('uncertain','Let me think about the modular arithmetic approach.',
              '同一答案句再次出现，随后换到模12的代数刻画。答案句冗余与后续新推导混合，不能把这一状态整体标成无效。'),
        5972:('mechanical_restatement','So, each step is correct, and the result is 4.',
              '局部已澄清周期3并得到4，随后再逐项重算同一四步。此前有周期2的错误表述，判定仅限被抽中的这次重复。'),
        2962:('verification_or_reuse','cofactor expansion might be more straightforward here.',
              '重写输入矩阵后开始余子式计算行列式，是对同一对象的新操作；存在局部措辞错误不改变重复的用途。')}
    rows=[]
    for c in cases:
        label,quote,reason=judgments[c['train_index']]
        text='\n'.join(v for x in c['contexts'] for k,v in x.items() if k in ('before','target','after'))
        assert quote in text,(c['train_index'],quote)
        rows.append(dict(train_index=c['train_index'],problem_sha256=c['problem_sha256'],label=label,evidence_quote=quote,reason=reason))
    summary={label:sum(r['label']==label for r in rows) for label in ('mechanical_restatement','verification_or_reuse','uncertain')}
    result=dict(reviewer='Single assistant, exposed problem/context and cap strata; no independent human review',
        cases_sha256=sha(out/'cases.json'),protocol_sha256=sha(out/'protocol.json'),counts=summary,rows=rows,
        block_followup=dict(reviewed_train_indices=[2152,5201,5417,1667,556],
            note='Three-step criterion was chosen after sentence review. 5201/5417/556 show repeated local derivation; 2152 is followed by an attempted borrowing correction; 1667 reuses a faulty relation to attempt a polynomial derivation. Not an independently validated detector.'),
        decision='Do not use isolated exact sentence repetition as semantic overthinking label. Three-step blocks have broader coverage but remain confidence/position confounded; do not replace the frozen vector or force ending from these labels.',
        next_requirement='A candidate must separate useful reuse from stagnation on held-out questions and use a deployment rule that preserves necessary correction. Repeated benchmark threshold search is not authorized by this audit.',
        limits='Eight stratified examples do not estimate population prevalence or establish step correctness. First occurrence need not be valid, repeat need not be harmful. No GPU, new answer or forward.',
        engineering_note='Initial console print failed under Windows GBK after complete UTF8 JSON had been saved; JSON was read directly. Console now prints ASCII-only summary; no resampling or overwriting.')
    save(out/'content_review.json',result);print(json.dumps(summary))

if __name__=='__main__':main()
