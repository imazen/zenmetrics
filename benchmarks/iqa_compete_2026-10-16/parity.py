"""Existing competition parity gates, shared by live and saved-row checks."""
from references import same_pair_reference

def qualify_rows(results,references,inventory,scales,python_package=None):
    mismatch=False
    for value in results:
        if value['status'] in ['skipped:nonfinite_score','skipped:gpu_dispatch_failure']:mismatch=True;continue
        if value['status'] not in ['ok','variant_mismatch']:continue
        if value.get('gpu_call_bit_variation_count',0)>0 or value.get('score_consistency_issue',False) or value.get('map_score_matches_scalar_bits') is False:
            value['status']='variant_mismatch';mismatch=True
        if value.get('score') is None:
            if value.get('n',0)>0:
                value.update(status='variant_mismatch',score_nonfinite=True);mismatch=True
            continue
        name=value['metric_name']
        ref=same_pair_reference(references,value)
        if ref is None:
            value.update(ref_score=None,abs_delta=None,relative_delta=None,comparison_status='unavailable:no qualified same-pair CPU reference',notes=value.get('notes','')+'; Native timing retained without a CPU reference')
            if python_package:value['status']='skipped:missing_same-pair_Rust_reference'
            continue
        reference=ref['score'];delta=abs(value['score']-reference)
        if name.endswith('-gpu'):
            bounds={'ssim2-gpu':max(.1,abs(reference)*.005),'dssim-gpu':abs(reference)*.01,
                    'butteraugli-gpu':abs(reference)*.002,'iwssim-gpu':.001,'cvvdp-gpu':.001}
            bound=bounds.get(name,0.)
            source='GPU crate parity_lock/opaque parity gates; CVVDP/IW-SSIM 1e-3 absolute'
        elif python_package and name not in {'lpips-alex','lpips-vgg','lpips-squeeze','dists','stlpips-alex','stlpips-vgg','deepdc'}:
            bound={'psnr':1e-10,'ssimc':7.5e-7,'ssimc-pyiqa-f32':7.5e-7,'ms-ssim-pyiqa-f32':1.9e-6,'gmsd':1e-14,'fsim':5e-5,'fsimc':5e-5,'vsi':1e-4,'haarpsi':5e-5,'flip-v17':1e-6,'cvvdp':1e-3,'vif':5e-13,'iwssim-piq':1e-4,'mdsi':0.,'ms-gmsd':0.,'ms-gmsdc':0.,'brisque':15.138428e-3}.get(name,0.)
            source='docs/METRIC_PROVENANCE.md; exact comparison for variants without a documented cross-package bound'
        else:
            column=next(m['columns'][0] for m in inventory if m['metric_name']==name)
            bound=scales[column]*1e-3
            source='zenmetrics-nn/models/jnd_scale.tsv; 1e-3 JND unchanged parity gate'
        value.update(comparison_status='same-pair CPU reference available',
                     ref_score=reference,abs_delta=delta,relative_delta=delta/abs(reference) if reference else None,
                     parity_abs_bound=bound,parity_source=source,reference_binary_sha=ref['impl_binary_sha'],
                     reference_pair_sha256=ref['pair_sha256'],reference_threads=ref['threads'])
        value.update(reference_kind=ref.get('reference_kind','qualified_cpu_timing'),
                     reference_source_status=ref.get('reference_source_status',ref.get('status')),
                     reference_log_sha256=ref.get('log_sha256'))
        if delta>bound or value.get('gpu_call_bit_variation_count',0)>0 or value.get('score_consistency_issue',False) or value.get('map_score_matches_scalar_bits') is False:value['status']='variant_mismatch';mismatch=True
    return mismatch

def reproduced_residual(row):
    """A previous rerun must reproduce this residual against the current reference."""
    reference=row.get('ref_score');bound=row.get('parity_abs_bound')
    if reference is None or bound is None:return False
    if row.get('abs_delta',0)<=bound:return False
    for first in row.get('mismatch_first_attempt',[]):
        if first.get('mode')==row['mode'] and first.get('score') is not None:
            if abs(first['score']-reference)>bound:return True
    return False
