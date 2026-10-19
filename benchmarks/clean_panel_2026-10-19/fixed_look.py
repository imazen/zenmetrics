"""Disabled production controller. Only in-memory synthetic runner/clock supported."""
import copy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics

POLICY_PATH=Path(__file__).with_name('fixed_look_policy.json')
MASK=(1<<64)-1


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def policy():
    return json.loads(POLICY_PATH.read_text())


class Random64:
    def __init__(self,seed):self.state=seed
    def next(self):
        x=self.state;x^=x>>12;x^=(x<<25)&MASK;x^=x>>27;self.state=x&MASK
        return (self.state*2685821657736338717)&MASK
    def below(self,n):
        limit=(1<<64)-((1<<64)%n)
        while True:
            value=self.next()
            if value<limit:return value%n
    def shuffle(self,values):
        values=list(values)
        for i in range(len(values)-1,0,-1):
            j=self.below(i+1);values[i],values[j]=values[j],values[i]
        return values


def schedule(p):
    rng=Random64(p['seed']);out=[]
    for _ in range(p['slots_per_order']):
        out.extend(['AB','BA'] if rng.next()&1==0 else ['BA','AB'])
    return out


def rank_interval(values):
    if len(values)!=120 or any(type(v) not in (float,int) or not math.isfinite(v) or v<=0 for v in values):
        raise ValueError('exactly 120 positive finite block ratios required')
    n=len(values);tail=0;k=0
    for candidate in range(1,n//2+1):
        tail+=math.comb(n,candidate-1)
        if Fraction(2*tail,2**n)<=Fraction(1,20):k=candidate
        else:break
    ordered=sorted(values);estimate=statistics.median(ordered)
    lo,hi=ordered[k-1],ordered[n-k]
    return dict(n=n,k=k,upper_rank=n+1-k,estimate=estimate,lo=lo,hi=hi,
                unconditional_model_coverage=str(1-Fraction(2*sum(math.comb(n,j) for j in range(k)),2**n)),
                conditional_selected_coverage='not established')


def interval_decision(lo,estimate,hi):
    if not (0<lo<=estimate<=hi) or not all(math.isfinite(v) for v in (lo,estimate,hi)):
        raise ValueError('invalid interval')
    precision=max((estimate-lo)/estimate,(hi-estimate)/estimate)<=.03
    crossing=lo<=1.08<=hi
    return dict(precision_pass=precision,threshold_crossing=crossing,
                conclusion=('inconclusive' if not precision or crossing else
                            'above-1.08' if lo>1.08 else 'below-1.08'))


def lag_stat(values):
    # Pearson correlation of adjacent log ratios, absolute, zero variance -> 0.
    a,b=values[:-1],values[1:];ma,mb=statistics.mean(a),statistics.mean(b)
    numerator=sum((x-ma)*(y-mb) for x,y in zip(a,b))
    denom=math.sqrt(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))
    return abs(numerator/denom) if denom else 0.


def diagnostic_stats(ratios,orders,permutations=99999):
    if len(ratios)!=120 or len(orders)!=120 or orders.count('AB')!=60 or orders.count('BA')!=60:
        raise ValueError('balanced completed cohort required')
    if any(not math.isfinite(v) or v<=0 for v in ratios):raise ValueError('invalid diagnostic ratio')
    logs=list(map(math.log,ratios));a=[i for i,o in enumerate(orders) if o=='AB'];b=[i for i,o in enumerate(orders) if o=='BA']
    order_stat=lambda vals:abs(statistics.median(vals[i] for i in a)-statistics.median(vals[i] for i in b))
    observed_order,observed_lag=order_stat(logs),lag_stat(logs)
    order_rng,lag_rng=Random64(20261020),Random64(20261021)
    counts=[0,0]
    # If every value equals, every permutation ties. This is an exact shortcut.
    if len(set(logs))==1:counts=[permutations,permutations]
    else:
        for _ in range(permutations):
            counts[0]+=order_stat(order_rng.shuffle(logs))>=observed_order
            counts[1]+=lag_stat(lag_rng.shuffle(logs))>=observed_lag
    probabilities=[(c+1)/(permutations+1) for c in counts]
    return dict(order_abs_median_log_difference=observed_order,lag1_abs_correlation=observed_lag,
        p_order=probabilities[0],p_lag=probabilities[1],permutations=permutations,tie_rule='>= observed',
        diagnostic_refusal=any(p<=.025 for p in probabilities),
        order_medians={o:statistics.median(v for v,label in zip(ratios,orders) if label==o) for o in ('AB','BA')},
        time_bin_medians=[statistics.median(ratios[i:i+20]) for i in range(0,120,20)],
        iid_established=False,selection_model_qualified=False)


class SyntheticClock:
    def __init__(self):self.ns=0
    def advance(self,ns):
        if type(ns) is not int or ns<0:raise ValueError('invalid synthetic duration')
        self.ns+=ns


class SyntheticRunner:
    """No callbacks, subprocesses, files or hardware acquisition path."""
    def __init__(self,fixtures,stop_at_ns=None):
        self.fixtures=copy.deepcopy(fixtures);self.index=0;self.stop_at_ns=stop_at_ns
    def stopped(self,clock):return self.stop_at_ns is not None and clock.ns>=self.stop_at_ns
    def admission_wait(self,clock):
        if self.index<len(self.fixtures):clock.advance(self.fixtures[self.index].get('admission_wait_ns',0))
    def run(self,slot,order,clock):
        if self.index>=len(self.fixtures):raise ValueError('synthetic fixture exhausted')
        row=self.fixtures[self.index];self.index+=1
        clock.advance(row.get('duration_ns',1))
        return row
    def cleanup(self,row,clock):
        clock.advance(row.get('cleanup_ns',0))
        return row.get('cleanup_complete',True) is True


class StudyBook:
    """Append-only in-memory full-study ledger; no repeated study-ID admission."""
    def __init__(self):self.records=[];self.ids=set()
    def reserve(self,study_id):
        if not isinstance(study_id,str) or not study_id or study_id in self.ids:raise ValueError('duplicate/missing study ID')
        self.ids.add(study_id)
    def append(self,report):self.records.append(copy.deepcopy(report))
    def write_full_report(self,path):
        # Every reserved/completed/incomplete study is retained. No overwrite or
        # selected-qualified-subset export. This never recomputes an interval.
        with Path(path).open('x') as stream:
            json.dump(dict(synthetic_only=True,study_ids=sorted(self.ids),studies=self.records),stream,indent=2)


class SyntheticController:
    def __init__(self,manifest,runner,clock,book,*,enable_synthetic=False):
        if not enable_synthetic:raise ValueError('disabled by default; synthetic opt-in required')
        if type(runner) is not SyntheticRunner or type(clock) is not SyntheticClock or type(book) is not StudyBook:
            raise ValueError('only closed synthetic runner/clock/book supported; no live callbacks')
        if manifest!=policy():raise ValueError('frozen policy mismatch')
        self.p=copy.deepcopy(manifest);self.runner=runner;self.clock=clock;self.book=book;self.used=False

    def run(self,study_id):
        if self.used:raise ValueError('single fixed look; no retry/repeated finalization')
        self.used=True;self.book.reserve(study_id)
        report=dict(study_id=study_id,policy=copy.deepcopy(self.p),policy_sha256=identity(self.p),synthetic_only=True,
            production_enabled=False,qualified=False,status='incomplete',attempts=[],not_attempted=[],events=[],accepted={'AB':0,'BA':0},resource_rejected={'AB':0,'BA':0})
        rows=[];seen=set();start=self.clock.ns;halt=None;orders=schedule(self.p)
        report['schedule']=orders;report['schedule_sha256']=identity(orders)
        for slot,order in enumerate(orders):
            if halt or len(rows)==120:
                report['not_attempted'].append(dict(slot=slot,order=order,reason=halt or 'target-complete'));continue
            if self.runner.stopped(self.clock):halt='STOP-before-launch'
            elif self.clock.ns-start>=self.p['deadline_ns']:halt='deadline-before-launch'
            if halt:
                report['not_attempted'].append(dict(slot=slot,order=order,reason=halt));continue
            if report['accepted'][order]==60:
                report['not_attempted'].append(dict(slot=slot,order=order,reason='order-quota-complete'));continue
            try:self.runner.admission_wait(self.clock)
            except Exception as exc:halt='invalid-admission-wait: '+str(exc)
            if self.runner.stopped(self.clock):halt='STOP-during-admission'
            elif self.clock.ns-start>=self.p['deadline_ns']:halt='deadline-during-admission'
            if halt:
                report['not_attempted'].append(dict(slot=slot,order=order,reason=halt));continue
            # Synthetic proof only. A live implementation must use OS lock receipts.
            token=identity(dict(study_id=study_id,slot=slot,policy=report['policy_sha256']))
            report['events'].append(dict(kind='synthetic-locks-held',slot=slot,owner=study_id,token=token,lock_order=['primary','secondary']))
            report['events'].append(dict(kind='synthetic-launch',slot=slot,order=order,owner=study_id,token=token))
            row={};error=None
            try:row=self.runner.run(slot,order,self.clock)
            except Exception as exc:error=str(exc)
            try:clean=self.runner.cleanup(row,self.clock)
            except Exception as exc:clean=False;error=str(exc)
            report['events'].append(dict(kind='synthetic-cleanup',slot=slot,complete=clean,token=token))
            if clean:report['events'].append(dict(kind='synthetic-locks-released',slot=slot,token=token))
            attempt=dict(slot=slot,order=order,raw=copy.deepcopy(row),finished_ns=self.clock.ns,cleanup_complete=clean,status='rejected')
            report['attempts'].append(attempt)
            if not clean:halt='containment-unknown';attempt['reason']=halt;continue
            if error:halt='runner-failure';attempt['reason']=error;continue
            if self.runner.stopped(self.clock):halt='STOP-inflight';attempt['reason']=halt;continue
            if self.clock.ns-start>=self.p['deadline_ns']:halt='deadline-inflight';attempt['reason']=halt;continue
            block=row.get('block_id')
            if not isinstance(block,str) or not block or block in seen:halt='duplicate/missing-block';attempt['reason']=halt;continue
            seen.add(block)
            if row.get('order',order)!=order:halt='wrong-order';attempt['reason']=halt;continue
            if row.get('status')=='resource-rejected':
                report['resource_rejected'][order]+=1;attempt['reason']='resource-rejected'
                if report['resource_rejected'][order]>12:halt='rejection-cap'
                continue
            if row.get('status')!='accepted':halt='fatal-attempt';attempt['reason']=halt;continue
            try:
                medians={}
                for arm in ('ours','canonical'):
                    v=row['times'][arm]
                    if not isinstance(v,list) or len(v)!=10 or any(type(t) not in (float,int) or not math.isfinite(t) or t<=0 for t in v):raise ValueError('nonpositive/nonfinite/missing timing')
                    medians[arm]=statistics.median(v)
                ratio=medians['ours']/medians['canonical']
                if not math.isfinite(ratio) or ratio<=0:raise ValueError('invalid block ratio')
            except (ValueError,KeyError,TypeError,OverflowError) as exc:halt='invalid-timing';attempt['reason']=str(exc);continue
            attempt['status']='accepted';report['accepted'][order]+=1
            rows.append((ratio,order,slot))
        report['elapsed_ns']=self.clock.ns-start
        if len(rows)==120 and not halt:
            values=[r[0] for r in rows];labels=[r[1] for r in rows]
            interval=rank_interval(values);decision=interval_decision(interval['lo'],interval['estimate'],interval['hi'])
            diagnostics=diagnostic_stats(values,labels,self.p['diagnostic_permutations'])
            report.update(status='synthetic-complete-unqualified',fixed_look=dict(interval=interval,decision=decision,diagnostics=diagnostics),
                reason='model and implementation require independent review; no live admission')
        else:report['reason']=halt or 'schedule-exhausted'
        self.book.append(report)
        return report


if __name__=='__main__':
    raise SystemExit('Production disabled. Import only for synthetic fixtures; no live runner exists.')
