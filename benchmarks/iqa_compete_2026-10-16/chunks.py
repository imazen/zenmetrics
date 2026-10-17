"""Aggregate actual zenbench rounds across independently bounded lock holds."""
import copy,statistics,struct
MIN_ROUNDS=10

def combine(pieces):
    if not pieces:raise ValueError('no chunks')
    identity=['host','os','metric_name','mode','size_name','threads','device','pair_sha256','impl_binary_sha','engine','model_dir','scope_seconds','chunk_budget_ms','sample_target_ns','ort_shared_library_sha256','ort_disable_telemetry']
    for row in pieces:
        if any(row.get(k)!=pieces[0].get(k) for k in identity):raise ValueError('chunk identity mismatch')
        if len(row['round_means_ns'])!=row['n']:raise ValueError('chunk sample count mismatch')
    means=[v for row in pieces for v in row['round_means_ns']]
    if not means:raise ValueError('empty chunks')
    values=sorted(means)
    def percentile(p):
        i=p*(len(values)-1);lo=int(i);hi=min(lo+1,len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(i-lo)
    out=copy.deepcopy(pieces[-1]);scores={struct.pack('>d',row['score']) for row in pieces}
    out.update(n=len(means),round_means_ns=means,median_ns=statistics.median(means),mean_ns=statistics.mean(means),p10_ns=percentile(.1),p90_ns=percentile(.9),load_before=pieces[0]['load_before'],load_after=pieces[-1]['load_after'],status='partial' if len(means)<MIN_ROUNDS else ('ok' if len(scores)==1 and all(p.get('map_score_matches_scalar_bits',True) for p in pieces) else 'variant_mismatch'),score_consistency_issue=len(scores)!=1)
    if len(means)>=MIN_ROUNDS:out.pop('partial_reason',None)
    out['zenbench_samples']=[sample for row in pieces for sample in row['zenbench_samples']]
    out['gpu_call_bit_variation_count']=sum(p.get('gpu_call_bit_variation_count',0) for p in pieces)
    if out['gpu_call_bit_variation_count'] and out['status']!='partial':out['status']='variant_mismatch'
    monitored=[p.get('contention') for p in pieces]
    if all(monitored):
        out['contention']=dict(monitored[-1],max_competing_mean_cores=max(p['max_competing_mean_cores'] for p in monitored),max_competing_interval_cores=max(p['max_competing_interval_cores'] for p in monitored),contaminated=any(p['contaminated'] for p in monitored))
        out['contention_peak_cores']=out['contention']['max_competing_interval_cores']
        out['timing_quality']='contaminated' if out['contention']['contaminated'] else 'clean'
    else:out['timing_quality']='legacy_unmonitored'
    out['chunks']=[{k:row.get(k) for k in ['attempt','n','score','log_path','log_sha256','load_before','load_after','resource']} for row in pieces]
    out['zenbench_chunk_samples']=[dict(arm_index=row['zenbench_arm_index'],samples=row['zenbench_samples']) for row in pieces]
    receipts=[row['resource'] for row in pieces if row.get('resource')]
    if len(receipts)==len(pieces):
        out['resource']=dict(exit_code=0,seconds=sum(r['seconds'] for r in receipts),peak_rss_gib=max(r['peak_rss_gib'] for r in receipts),min_available_mib=min(r['min_available_mib'] for r in receipts),peak_load=max(r['peak_load'] for r in receipts),chunks=len(pieces),note='Seconds sum bounded holds; peak RSS is the largest observed per-chunk OS peak, not an aggregate concurrent-memory claim')
    out['notes']=out.get('notes','')+f'; {len(pieces)} independent warmed chunks; actual zenbench round means aggregated only after at least ten rounds'
    return out
