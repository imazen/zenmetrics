"""Settled variant definitions; independent of branch implementation qualification."""
import re


def fixed_factor_contract(metric):
    fsim_vsi=re.fullmatch(r'(fsimc?|vsi)(?:-y)?-(?:k(\d+)|(nodownscale))',metric)
    ssim=re.fullmatch(r'(?:ssim|ssimc|ms-ssim)-pyiqa(?:-f32)?-k(\d+)',metric)
    if not fsim_vsi and not ssim:
        return None
    if fsim_vsi:
        size=int(fsim_vsi[2]) if fsim_vsi[2] else 1
        definition='Replace only kernel_size = max(1, round(min(H,W)/256)) with kernel_size = '+str(size)+'.'
    else:
        size=int(ssim[1])
        definition='Set downsample=True and kernel_size='+str(size)+'.'
    result=dict(kernel_size=size,definition=definition,other_pipeline_steps='unchanged',
                qualification='Settled definition, 2026-10-19; branch and implementation verification tracked separately.')
    if ssim:
        result['downsample']=True
    return result


def description(metric):
    contract=fixed_factor_contract(metric)
    return (contract['definition']+' All other pipeline steps remain unchanged. '+contract['qualification']) if contract else ''
