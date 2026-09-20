"""Read-only coverage tests on exposed native RC14 trajectories, not efficacy."""
import ast,hashlib,json,re
from fractions import Fraction
from pathlib import Path
import numpy as np
from policy import next_opening,TRIGGERS
from review_wsc_native_cpu import Decoder

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent

def numeric(s):
    node=ast.parse(s,mode='eval').body
    def ev(n):
        if isinstance(n,ast.Constant) and type(n.value) is int:return Fraction(n.value)
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):
            return ev(n.operand)*(1 if isinstance(n.op,ast.UAdd) else -1)
        if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
            a,b=ev(n.left),ev(n.right)
            if isinstance(n.op,ast.Add):return a+b
            if isinstance(n.op,ast.Sub):return a-b
            if isinstance(n.op,ast.Mult):return a*b
            return a/b
        raise ValueError('Outside exact rational arithmetic grammar')
    return ev(node)

def equations(step):
    # Restrict to complete delimited math or a whole line, not arbitrary substrings.
    spans=re.findall(r'(?<!\\)\$([^$\n]+)\$|\\\((.*?)\\\)|\\\[(.*?)\\\]',step,re.S)
    candidates=[next(s for s in v if s) for v in spans if any(v)]
    candidates+=step.splitlines();out=[]
    for text in dict.fromkeys(candidates):
        s=text.strip().rstrip('.').strip()
        if not re.fullmatch(r'[0-9 ()+*/\-]+=[0-9 ()+*/\-]+',s):continue
        left,right=s.split('=')
        try:a,b=numeric(left),numeric(right)
        except (ValueError,SyntaxError,ZeroDivisionError,RecursionError):continue
        out.append(dict(text=text,left=str(a),right=str(b),false=a!=b))
    return out

def tests():
    assert numeric('2*(3+4)/7')==2
    assert equations('$2+3=6$')[0]['false']
    assert not equations('\\(2+3=5\\)')[0]['false']
    for text in ['x=2','2+3=5+','2**3=8','1/0=2','abc 2+3=6','2.5=3']:
        assert not equations(text)
    assert equations('Suppose $2+3=6$ for contradiction.')[0]['false'] # needs manual assertion-scope review

def main():
    tests();out=HERE/'local_refinements_20260917'
    plan=json.loads((out/'plan.json').read_text(encoding='utf8'))
    raw=ROOT/'.codex_work/wsc_native_long_complete_20260917'
    run=next((raw/'results').rglob('engineering_gate.json')).parent
    records=json.loads((run/'RC14_wsc_shadow/result.json').read_text())['records']
    decoder=Decoder('E:/srtp/srtp-final/.codex_work/label_audit_30_20260910/tokenizer.json')
    boundaries={i for i,p in decoder.vocab.items() if 'ĊĊ' in p}
    rows=[];hashes={}
    for r in records:
        i=r['train_index'];ids=r['token_ids'];ids=ids[:ids.index(151649)] if 151649 in ids else ids
        cp=run/'RC14_wsc_shadow'/f'{i}_control.npy';control=np.load(cp)
        hashes[str(cp.relative_to(ROOT))]=hashlib.sha256(cp.read_bytes()).hexdigest()
        opening=False;text='';previous_boundary=0;false_pending=[];cases=[];gaps=[];mixed=[];actual=0
        for pos,tid in enumerate(ids):
            # control[pos] is the state governing prediction of this generated token.
            coef=float(control[pos,0]);mean=float(control[pos,1])
            negative=np.isfinite(coef) and np.isfinite(mean) and coef<0
            semantic_open=bool(re.search(r'\n\n[^\S\n]*$',text))
            if opening and negative:
                actual+=1
                if false_pending:
                    cases.append(dict(position=pos,coefficient=coef,equations=false_pending,
                        prefix=text[-1600:],next_token=decoder.decode([tid])))
                    false_pending=[] # one recorded first opportunity per completed step
            if semantic_open and not opening and negative:
                gaps.append(dict(position=pos,coefficient=coef,prefix=text[-200:],next_token=decoder.decode([tid])))
            piece=decoder.decode([tid]);text+=piece
            if tid in boundaries:
                if piece.rsplit('\n\n',1)[-1].strip():mixed.append(dict(position=pos,piece=piece))
                step=text[previous_boundary:]
                false_pending=[e for e in equations(step) if e['false']]
                previous_boundary=len(text)
            opening=next_opening(opening,piece,tid in boundaries,tid)
        rows.append(dict(train_index=i,problem_sha256=r['problem_sha256'],thinking_tokens=len(ids),
            eligible_sampling_positions=actual,arithmetic_release_cases=cases,
            split_newline_missed_positions=gaps,mixed_boundary_tokens=mixed))
    result=dict(plan_sha256=hashlib.sha256((out/'plan.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows,
        arithmetic_coverage_questions=sum(bool(r['arithmetic_release_cases']) for r in rows),
        split_boundary_coverage_questions=sum(bool(r['split_newline_missed_positions']) for r in rows),
        input_sha256=hashes,new_generations=0,new_forwards=0,
        limitation='Eight exposed engineering trajectories only. No counterfactual savings. False arithmetic can be quoted or a reductio premise; manual prefix-only assertion review is required before intervention.')
    with (out/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','input_sha256')},indent=2))
    for r in rows:print(r['train_index'],r['eligible_sampling_positions'],len(r['arithmetic_release_cases']),len(r['split_newline_missed_positions']),len(r['mixed_boundary_tokens']))

if __name__=='__main__':main()
