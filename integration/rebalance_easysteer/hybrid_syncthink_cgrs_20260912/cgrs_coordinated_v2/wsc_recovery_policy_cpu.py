"""Paper Algorithm1 streak semantics; proposed bounded recovery budget.

Pure state machine only, not an active sampler/controller or official WSC.
Chunk length is the accepted-token span excluding its trailing delimiter.
Budget includes discarded generations; recovery never replenishes the cap.
"""
import math


class LoopState:
    def __init__(self):
        self.long = self.short = 0
        self.disabled = self.fired = False

    def observe(self, score, chunk_tokens):
        if self.disabled or self.fired:return False
        if not math.isfinite(score) or not 0 <= score <= 1 or chunk_tokens < 0:
            self.disabled=True
            self.long=self.short=0
            return False
        if score <= .5:
            self.long=self.short=0
        elif chunk_tokens >= 10:
            self.long+=1;self.short=0
        else:
            self.short+=1;self.long=0
        self.fired = self.long >= 2 or self.short >= 5
        return self.fired


def recovery_allowance(generated_so_far, inserted_cue_tokens, cap=16000):
    """Our budget, not paper's32k+4k: max4k within original global16k."""
    if generated_so_far < 0 or inserted_cue_tokens < 0:raise ValueError('Negative accounting')
    available=cap-generated_so_far-inserted_cue_tokens
    return max(0,min(4096,available))


def self_test():
    p=LoopState();assert not p.observe(.9,3);assert not p.observe(.9,3);assert not p.observe(.9,20)
    assert p.observe(.9,20);assert not p.observe(.9,20) # one rescue only
    p=LoopState();assert [p.observe(.9,3) for _ in range(5)]==[False]*4+[True]
    p=LoopState();assert not p.observe(.9,20);assert not p.observe(.5,20);assert not p.observe(.9,20)
    p=LoopState();assert not p.observe(float('nan'),20);assert not p.observe(.9,20);assert p.disabled
    assert recovery_allowance(15000,20)==980
    assert recovery_allowance(15990,20)==0
    assert recovery_allowance(2000,20)==4096
    assert LoopState().__dict__==dict(long=0,short=0,disabled=False,fired=False)
    return dict(mixed_lengths_keep_individual_history=True,short_and_long_thresholds=True,
        exact_half_score_resets=True,one_rescue_only=True,invalid_score_disables=True,
        discarded_tokens_not_refunded=True,new_request_reset=True,gpu_used=False)


if __name__=='__main__':
    import json
    print(json.dumps(self_test(),indent=2))
