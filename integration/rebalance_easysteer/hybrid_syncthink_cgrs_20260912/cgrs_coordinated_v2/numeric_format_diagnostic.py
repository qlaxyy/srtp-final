"""Conservative supplemental GSM numeric extraction; never replace author grading."""
import re
from fractions import Fraction

NUMBER=r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*\d+)?'
def value(s):
    try:return Fraction(s.replace(',','').replace(' ',''))
    except (ValueError,ZeroDivisionError):return None
def extract(text,finish_reason):
    if finish_reason!='stop' or '</think>' not in text:return dict(status='unresolved',reason='unfinished_thinking_or_generation')
    final=text.rsplit('</think>',1)[1].strip()
    final=re.sub(r'<\|[^>]*\|>','',final).strip()
    boxes=[]
    for m in re.finditer(r'\\boxed\s*\{',final):
        depth=1;end=m.end()
        while end<len(final) and depth:
            depth+=(final[end]=='{')-(final[end]=='}');end+=1
        if depth:return dict(status='unresolved',reason='unbalanced_box')
        boxes.append(final[m.end():end-1])
    if boxes:
        span=boxes[-1];kind='final_box'
        cleaned=re.sub(r'\\(?:dfrac|tfrac|frac)\{([+-]?\d+)\}\{(\d+)\}',r'\1/\2',span)
        cleaned=re.sub(r'\\(?:text|mathrm)\{([A-Za-z /._%-]+)\}',r'\1',cleaned)
        for token in ('\\,','\\!','\\;','\\ ','\\$','$','\\%'):cleaned=cleaned.replace(token,'' if token!='\\%' else '%')
        # No algebra, multiple numbers, scientific notation, radicals, or alternatives.
        m=re.fullmatch(r'\s*('+NUMBER+r')\s*(?:[%A-Za-z /._-]*)\s*',cleaned)
        if not m:return dict(status='unresolved',reason='non_scalar_box',span=span)
        n=value(m.group(1))
    else:
        paras=[s.strip() for s in re.split(r'\n\s*\n',final) if s.strip()]
        if not paras:return dict(status='unresolved',reason='empty_final')
        span=paras[-1];kind='last_paragraph_unique_number'
        if '?' in span or re.search(r'\b(?:not|either|between|or)\b',span,re.I):return dict(status='unresolved',reason='non_assertive_final',span=span)
        ns=[value(m.group()) for m in re.finditer(NUMBER,span)]
        if not ns or None in ns or len(set(ns))!=1:return dict(status='unresolved',reason='ambiguous_final_numbers',span=span)
        n=ns[0]
    if n is None:return dict(status='unresolved',reason='invalid_numeric',span=span)
    return dict(status='resolved',rule=kind,value=str(n),span=span)

def checks():
    assert extract('20</think>Answer: **$5,600**.','stop')['value']=='5600'
    assert extract('</think>\\boxed{72\\ \\text{Mb/h}}','stop')['value']=='72'
    assert extract('</think>\\boxed{72\\, \\text{Mb/h}}','stop')['value']=='72'
    assert extract('</think>\\boxed{\\frac{3}{4}}','stop')['value']=='3/4'
    for t,reason in [('20','stop'),('</think>Answer20','length'),('</think>20 or30','stop'),('</think>9*80-700=20','stop'),('</think>\\boxed{2+3}','stop'),('</think>\\boxed{1/0}','stop')]:
        assert extract(t,reason)['status']=='unresolved'
    assert extract('\\boxed{100}</think>Final:20','stop')['value']=='20'
    return 11
