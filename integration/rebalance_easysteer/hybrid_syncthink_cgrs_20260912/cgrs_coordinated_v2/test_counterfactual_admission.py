from types import SimpleNamespace as NS
from counterfactual_admission import seed_request
from counterfactual_snapshot import capture,fork
from test_counterfactual_snapshot import fixture

def main():
    assets={'identity':'fixed'};a=fixture();snap=fork(capture(a,'parent',[1,2],[3,4,9],asset_identity=assets),action='apply')
    def request():
        r=NS(prompt_token_ids=[1,2],num_computed_tokens=0,num_output_tokens=0,num_output_placeholders=0,max_tokens=13,all_token_ids=[1,2])
        def append(ids):r.all_token_ids.extend(ids);r.num_output_tokens+=len(ids)
        r.append_output_token_ids=append;return r
    r=request();s=NS(requests={'child':r},running=[])
    seed_request(s,'child',snap,remaining_tokens=10,asset_identity=assets)
    assert r.all_token_ids==[1,2,3,4,9] and r.prompt_token_ids==[1,2] and r.num_output_tokens==3
    for variant in ('started','running','budget','prompt','asset','placeholder'):
        r=request();s=NS(requests={'child':r},running=[]);ident=assets
        if variant=='started':r.num_computed_tokens=1
        if variant=='running':s.running=[r]
        if variant=='budget':r.max_tokens=10
        if variant=='prompt':r.prompt_token_ids=[1,7]
        if variant=='asset':ident={'identity':'changed'}
        if variant=='placeholder':r.num_output_placeholders=1
        before=list(r.all_token_ids)
        try:seed_request(s,'child',snap,remaining_tokens=10,asset_identity=ident)
        except ValueError:pass
        else:raise AssertionError(variant)
        assert r.all_token_ids==before
    print('PASS 7 CPU admission contracts; native engine integration requires GPU gate')

if __name__=='__main__':main()
