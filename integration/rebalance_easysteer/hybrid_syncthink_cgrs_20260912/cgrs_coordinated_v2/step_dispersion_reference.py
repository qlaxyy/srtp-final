"""CPU-only signal prototype, no gate, controller, or correctness predictor."""
import math
class StepDispersion:
    def __init__(self):
        self.thinking=False;self.last=None;self._reset()
    def _reset(self):
        self.n=0;self.mean=0.;self.m2=0.;self.minimum=1.
    def accept(self,probability,*,boundary=False,start=False,end=False):
        if start:
            self.thinking=True;self.last=None;self._reset();return
        if end:
            self.thinking=False;self.last=None;self._reset();return
        if not self.thinking:return
        if boundary:
            if self.n:self.last=dict(count=self.n,mean=self.mean,variance=max(0.,self.m2/self.n),minimum=self.minimum)
            self._reset();return
        if not math.isfinite(probability) or not 0<=probability<=1:raise ValueError('Invalid raw probability')
        self.n+=1;delta=probability-self.mean;self.mean+=delta/self.n
        self.m2+=delta*(probability-self.mean);self.minimum=min(self.minimum,probability)
