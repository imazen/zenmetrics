#!/usr/bin/env python3
"""Time-boxed AIC WD convention probe, separate from the shipped Cool-Chic port.

A native-port control is supplied by wd_score. The previous 24-variant grid
is not repeated. New input, tap and aggregation conventions are selected on
three pairs, then the winning convention is evaluated on all 61 pairs.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time
from datetime import datetime, timezone

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision

MODES = ('sum', 'mean', 'sqrt_sum', 'sqrt_mean', 'mean_sqrt', 'sum_sqrt', 'channel_sum_mean')
TAPS = (3, 8, 15, 22)
CONFIGS = [
    dict(name='pre_relu_raw', norm='raw', taps=(2,7,14,21)),
    dict(name='pre_relu_imnet', norm='imnet', taps=(2,7,14,21)),
    dict(name='post_pool_raw', norm='raw', taps=(4,9,16,23)),
    dict(name='post_pool_imnet', norm='imnet', taps=(4,9,16,23)),
    dict(name='pm1', norm='pm1', taps=TAPS),
    dict(name='u8_range', norm='u8', taps=TAPS),
    dict(name='linear_rgb', norm='linear', taps=TAPS),
    dict(name='bgr_raw', norm='bgr', taps=TAPS),
    dict(name='luma601', norm='luma', taps=TAPS),
    dict(name='short256_raw', norm='raw', taps=TAPS, short=256),
    dict(name='short256_imnet', norm='imnet', taps=TAPS, short=256),
    dict(name='short512_raw', norm='raw', taps=TAPS, short=512),
    dict(name='short512_imnet', norm='imnet', taps=TAPS, short=512),
    dict(name='pre_relu_rescaled_imnet', norm='imnet', taps=(2,7,14,21), rescale=True),
    dict(name='post_pool_rescaled_raw', norm='raw', taps=(4,9,16,23), rescale=True),
]


def load(path):
    return torchvision.transforms.functional.to_tensor(Image.open(path).convert('RGB')).unsqueeze(0).cuda()


def transform(x, config):
    if config.get('short'):
        h, w = x.shape[-2:]
        scale = config['short'] / min(h, w)
        x = F.interpolate(x, size=(int(h*scale), int(w*scale)), mode='bilinear', align_corners=False, antialias=True)
    norm = config['norm']
    if norm == 'imnet':
        x = (x-x.new_tensor([.485,.456,.406]).view(1,3,1,1))/x.new_tensor([.229,.224,.225]).view(1,3,1,1)
    elif norm == 'pm1': x = x*2-1
    elif norm == 'u8': x = x*255
    elif norm == 'linear': x = torch.where(x <= .04045, x/12.92, ((x+.055)/1.055)**2.4)
    elif norm == 'bgr': x = x[:,[2,1,0]]
    elif norm == 'luma': x = (x*x.new_tensor([.299,.587,.114]).view(1,3,1,1)).sum(1,keepdim=True).repeat(1,3,1,1)
    return x


def features(backbone, x, taps):
    result = []
    for i, layer in enumerate(backbone):
        x = layer(x)
        if i in taps:
            value = x.clone() if isinstance(layer, torch.nn.Conv2d) else x
            result.append(value.reshape(-1,1,*x.shape[-2:]))
        if i == max(taps): break
    assert len(result) == len(taps)
    return result


def components(loss, a, b, shape, rescale):
    result = []
    for fa, fb in zip(a, b):
        n, _, h, w = fa.shape
        ratio = (math.log2(shape[0]/h)+math.log2(shape[1]/w))/2 if rescale else 0
        values = np.zeros(5, dtype=np.float64)
        for start in range(0,n,4):
            ca, cb = fa[start:start+4], fb[start:start+4]
            # Reuse the unmodified upstream moment tensors across sigma values.
            # The forward computation is unchanged; the cache is scoped to a
            # single channel chunk and is discarded before the next one.
            original = loss.compute_multiscale_stats
            cache = {}
            def moments(x, levels):
                key = id(x)
                if key not in cache: cache[key] = original(x,levels)
                return cache[key]
            loss.compute_multiscale_stats = moments
            try:
                for k in range(1,6):
                    sigma = torch.full((1,1,h,w), max(k-ratio,0), device=ca.device, dtype=ca.dtype)
                    values[k-1] += float(loss.wasserstein_distortion(ca,cb,sigma,5))*len(ca)/n
            finally:
                loss.compute_multiscale_stats = original
            del cache
        result.append(values.astype(np.float32))
    return np.asarray(result), [len(f) for f in a]


def aggregate(values, channels, mode):
    if mode == 'sum':
        total = np.zeros(5, np.float32)
        for row in values: total += row
        return total.astype(float)
    if mode == 'mean': return aggregate(values,channels,'sum')/len(values)
    if mode == 'sqrt_sum': return np.sqrt(aggregate(values,channels,'sum'))
    if mode == 'sqrt_mean': return np.sqrt(aggregate(values,channels,'sum')/len(values))
    if mode == 'mean_sqrt': return np.sqrt(values.astype(float)).mean(axis=0)
    if mode == 'sum_sqrt': return np.sqrt(values.astype(float)).sum(axis=0)
    if mode == 'channel_sum_mean': return (values.astype(float)*np.asarray(channels)[:,None]).mean(axis=0)
    raise ValueError(mode)


def residual(predicted, published, mappings):
    raw, jnd = [], []
    for k,(a,b) in enumerate(zip(predicted,published),1):
        m = mappings[f'WD_s{k}']
        assert m['b'] == 0 and m['sign'] == 1
        raw.append(abs(a-b))
        # WD's supplied site mapping has b=0, sign=+1: a*max(0,x)^c.
        jnd.append(abs(m['a']*max(0,a)**m['c']-m['a']*max(0,b)**m['c']))
    return raw,jnd


def main():
    ap = argparse.ArgumentParser()
    for arg in ('reference','site','images','config','native_scores','previous_log','out'):
        ap.add_argument('--'+arg.replace('_','-'),required=True)
    ap.add_argument('--budget-seconds',type=float,required=True)
    args = ap.parse_args()
    start = time.monotonic(); deadline=start+args.budget_seconds
    start_utc=datetime.now(timezone.utc).isoformat()
    search_deadline=start+args.budget_seconds/2
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    spec=importlib.util.spec_from_file_location('coolchic_wd',args.reference)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    loss=module.WassersteinLoss();loss.to_device('cuda')
    backbone=torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features.eval().cuda()
    rows=list(csv.DictReader(open(args.site)));assert len(rows)==61
    mappings=json.load(open(args.config))['mappings']
    native={Path(r['dist_path']).name:np.array([float(r[f'wd_s{k}']) for k in range(1,6)]) for r in csv.DictReader(open(args.native_scores),delimiter='\t')}
    assert len(native)==61
    panel=rows[:3]
    selection=[]
    stream=(out/'selection.tsv').open('w')
    writer=csv.writer(stream,delimiter='\t');writer.writerow(['convention','aggregation','image','column','predicted','published','absolute_error','site_jnd_error'])
    def record(config,mode,predictions):
        errors=[]
        for row,predicted in zip(panel,predictions):
            pub=[float(row[f'WD_s{k}']) for k in range(1,6)]
            raw,jnd=residual(predicted,pub,mappings);errors.extend(jnd)
            for k in range(5):writer.writerow([config['name'],mode,row['distorted'],f'WD_s{k+1}',f'{predicted[k]:.17e}',f'{pub[k]:.17e}',f'{raw[k]:.17e}',f'{jnd[k]:.17e}'])
        stream.flush()
        selection.append(dict(config=config,aggregation=mode,max_jnd=max(errors),median_jnd=statistics.median(errors)))
    def from_sum(values,mode):
        if mode=='sum':return values
        if mode=='mean':return values/4
        if mode=='sqrt_sum':return np.sqrt(values)
        if mode=='sqrt_mean':return np.sqrt(values/4)
        raise ValueError(mode)
    for mode in ('sum','mean','sqrt_sum','sqrt_mean'):
        record(dict(name='native_raw',norm='raw',taps=TAPS),mode,[from_sum(native[r['distorted']],mode) for r in panel])
    # Retain the previously measured ImageNet-native control; do not rerun its
    # old convention grid. Its archived values have six printed decimals.
    old={};name=None
    for line in Path(args.previous_log).read_text().splitlines():
        if line.startswith('S') and '.png ' in line:name=line.split()[0]
        elif line.strip().startswith('imnet '):old[name]=np.array([float(x) for x in line.split()[1:]])
    assert all(r['distorted'] in old for r in panel)
    for mode in ('sum','mean','sqrt_sum','sqrt_mean'):
        record(dict(name='historical_imnet_native',norm='imnet',taps=TAPS),mode,[from_sum(old[r['distorted']],mode) for r in panel])
    for config in CONFIGS:
        if time.monotonic()>search_deadline:break
        print('candidate',config['name'],flush=True)
        predictions={mode:[] for mode in MODES}
        for row in panel:
            r=transform(load(Path(args.images)/row['source']),config)
            d=transform(load(Path(args.images)/row['distorted']),config)
            a=features(backbone,r,config['taps']);b=features(backbone,d,config['taps'])
            values,channels=components(loss,a,b,r.shape[-2:],config.get('rescale',False))
            for mode in MODES:predictions[mode].append(aggregate(values,channels,mode))
            del r,d,a,b;torch.cuda.empty_cache()
            print('  panel',row['distorted'],flush=True)
        for mode in MODES:record(config,mode,predictions[mode])
        best=min(selection,key=lambda x:(x['max_jnd'],x['median_jnd']))
        (out/'selection_summary.json').write_text(json.dumps(dict(best=best,candidates=selection),indent=2)+'\n')
        print('best',best,flush=True)
    stream.close()
    best=min(selection,key=lambda x:(x['max_jnd'],x['median_jnd']))
    config,mode=best['config'],best['aggregation']
    checked=0; mismatches=0
    with (out/'best_full61.tsv').open('w') as stream:
        writer=csv.writer(stream,delimiter='\t');writer.writerow(['image','column','predicted','published','absolute_error','site_jnd_error'])
        for row in rows:
            if time.monotonic()>deadline-10:break
            if config['name']=='native_raw':predicted=from_sum(native[row['distorted']],mode)
            else:
                def evaluate():
                    r=transform(load(Path(args.images)/row['source']),config)
                    d=transform(load(Path(args.images)/row['distorted']),config)
                    a=features(backbone,r,config['taps']);b=features(backbone,d,config['taps'])
                    v,c=components(loss,a,b,r.shape[-2:],config.get('rescale',False))
                    value=aggregate(v,c,mode)
                    del r,d,a,b;torch.cuda.empty_cache()
                    return value
                predicted=evaluate();retry=evaluate()
                if not np.array_equal(predicted,retry):
                    mismatches+=1; print('repeat mismatch',row['distorted'],predicted.tolist(),retry.tolist(),flush=True)
                    third=evaluate()
                    assert np.array_equal(retry,third),'winner did not reproduce'
                    predicted=retry
            published=[float(row[f'WD_s{k}']) for k in range(1,6)]
            raw,jnd=residual(predicted,published,mappings)
            for k in range(5):writer.writerow([row['distorted'],f'WD_s{k+1}',f'{predicted[k]:.17e}',f'{published[k]:.17e}',f'{raw[k]:.17e}',f'{jnd[k]:.17e}'])
            stream.flush();checked+=1;print('winner',checked,row['distorted'],flush=True)
    metadata=dict(torch=torch.__version__,torchvision=torchvision.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),start_utc=start_utc,end_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,
        budget_seconds=args.budget_seconds,best=best,checked_pairs=checked,repeat_mismatches=mismatches,
        novel_conventions_completed=len(selection)-2,selection_pairs=[r['distorted'] for r in panel],
        jnd_definition='abs(a*predicted**c - a*published**c); WD b=0, sign=+1 from supplied site config',
        inputs={k:hashlib.sha256(Path(getattr(args,k)).read_bytes()).hexdigest() for k in ['reference','site','config','native_scores','previous_log']})
    (out/'receipt.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('PROBE_RESULT',json.dumps(metadata),flush=True)

if __name__=='__main__':main()
