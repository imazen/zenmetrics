"""Canonical matched timed validation; no metric import or execution at import."""
import math
import struct
import threading

BOUNDARY='paired-nr-timed-validation-v1'


class Observer:
    def __init__(self):
        self.local=threading.local()

    def record(self, values):
        if len(values)!=2 or [p[0] for p in values]!=['brisque','brisque_ref']:
            raise ValueError('paired NR column/order mismatch')
        distorted,reference=values[0][1],values[1][1]
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (distorted,reference)):
            raise ValueError('nonfinite paired NR output')
        bits=tuple(struct.pack('>d',v) for v in (distorted,reference))
        state=self.local
        if not hasattr(state,'first'):
            state.first=bits;state.count=0;state.changed=False
        state.changed |= state.first!=bits
        state.last=(distorted,reference);state.count+=1
        return distorted,reference

    def result(self):
        state=self.local
        if not hasattr(state,'first'):raise ValueError('missing paired NR result')
        if state.changed:raise ValueError('paired NR result changed across calls')
        return (*state.last,state.count)


def timed_pair(score_pair, observer, clock):
    start=clock()
    scores=observer.record(score_pair())
    elapsed=clock()-start
    return elapsed,scores
