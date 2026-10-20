#!/usr/bin/env python3
"""Resident RGB callers for explicitly supplied original author networks.

Source modules and checkpoints stay unchanged. Existing export helpers provide
named input conventions and the previously checked MANIQA CPU compatibility
forward; the actual metric operators come from the author modules.
"""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import typing


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes} != set(names):raise ValueError('missing explicit source helper')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def setup_prefix(path, end_name, argv):
    tree = ast.parse(path.read_text())
    end = next(node.lineno for node in tree.body if isinstance(node,ast.FunctionDef) and node.name == end_name)
    nodes = [node for node in tree.body if node.lineno < end]
    old = sys.argv
    namespace = dict(__name__='canonical_original_setup',__file__=str(path))
    try:
        sys.argv = [str(path), *map(str,argv)]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    finally:sys.argv = old
    return namespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path)
    p.add_argument('metric',choices=['ckdn','afine','dreamsim','arniqa','maniqa','liqe','hyperiqa','paq2piq','wd'])
    p.add_argument('width',type=int);p.add_argument('height',type=int)
    p.add_argument('reference',type=Path);p.add_argument('distorted',type=Path)
    p.add_argument('threads',type=int);p.add_argument('samples',type=int)
    p.add_argument('--setup-only',action='store_true',help='Validate source/checkpoint construction without acquiring scores')
    a = p.parse_args();cfg=json.loads(a.config.read_text())
    os.environ['OPENCV_IPP']='disabled'
    import numpy as np
    import torch
    import torchvision
    import torchvision.transforms.functional as TF
    from PIL import Image
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
    r,d=[np.fromfile(path,np.uint8).reshape(a.height,a.width,3) for path in [a.reference,a.distorted]]
    root=Path(cfg['assets']);sources=Path(cfg['sources']);exports=Path(cfg['exports'])
    def tensor(rgb):return TF.to_tensor(Image.fromarray(rgb))
    nr=a.metric in ['arniqa','maniqa','liqe','hyperiqa','paq2piq']
    if a.metric=='ckdn':
        namespace=setup_prefix(exports/'ckdn/export.py','score',
            ['unused',Path.cwd(),root/'CKDN',root/'ckdn.bin'])
        functions(exports/'ckdn/export.py',['prep'],namespace)
        model=namespace['Full']().eval()
        def score():
            x,y=[torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0).float() for rgb in [r,d]]
            return float(model(namespace['prep'](x,True),namespace['prep'](y,False)).item()),None
    elif a.metric=='afine':
        namespace=setup_prefix(exports/'afine/export.py','pe_torch',
            ['unused',Path.cwd(),root/'AFINE',root/'afine.pth',root/'ViT-B-32.pt'])
        def score():
            x,y=[namespace['prep'](tensor(rgb)) for rgb in [r,d]]
            return float(namespace['score'](x,y).item()),None
    elif a.metric=='dreamsim':
        sys.path.insert(0,str(root/'DreamSim'))
        from dreamsim import dreamsim
        model,prep=dreamsim(pretrained=True,device='cpu',cache_dir=str(root/'dreamsim-weights'),dreamsim_type='ensemble')
        model.eval()
        def score():return float(model(prep(Image.fromarray(r)),prep(Image.fromarray(d))).item()),None
    elif a.metric=='arniqa':
        sys.path.insert(0,str(sources/'ARNIQA'))
        module=load(sources/'ARNIQA/hubconf.py','original_arniqa')
        original_encoder=module.ResNet
        def encoder(*args,**kwargs):
            kwargs['pretrained']=False
            return original_encoder(*args,**kwargs)
        module.ResNet=encoder
        original=torch.hub.load_state_dict_from_url
        def local_weight(url,**kwargs):
            return torch.load(root/'arniqa-weights'/url.rsplit('/',1)[1],map_location='cpu',weights_only=False)
        torch.hub.load_state_dict_from_url=local_weight
        try:model=module.ARNIQA('kadid10k').eval()
        finally:torch.hub.load_state_dict_from_url=original
        namespace=functions(sources/'ARNIQA/utils/utils_data.py',['center_corners_crop'],
            dict(TF=TF,PILImage=Image.Image,List=typing.List))
        def single(rgb):
            image=Image.fromarray(rgb);half=TF.resize(image,[a.height//2,a.width//2],TF.InterpolationMode.BILINEAR)
            x,y=[torch.stack([TF.normalize(TF.to_tensor(crop),[.485,.456,.406],[.229,.224,.225])
                for crop in namespace['center_corners_crop'](view,224)]) for view in [image,half]]
            return float(model(x,y,return_embedding=False,scale_score=True).mean().item())
    elif a.metric=='maniqa':
        import cv2
        import timm
        cv2.setNumThreads(a.threads);cv2.ipp.setUseIPP(False)
        sys.path.insert(0,str(sources/'MANIQA'))
        module=load(sources/'MANIQA/models/maniqa.py','original_maniqa')
        original=timm.create_model
        def no_pretraining(*args,**kwargs):
            kwargs['pretrained']=False
            return original(*args,**kwargs)
        timm.create_model=no_pretraining
        try:model=module.MANIQA(embed_dim=768,num_outputs=1,dim_mlp=768,patch_size=8,img_size=224,
            window_size=4,depths=[2,2],num_heads=[4,4],num_tab=2,scale=.8)
        finally:timm.create_model=original
        missing,_=model.load_state_dict(torch.load(root/'ckpt_koniq10k.pt',map_location='cpu',weights_only=True),strict=False)
        if missing:raise ValueError('unfilled MANIQA parameters: '+str(missing))
        model.eval()
        namespace=functions(exports/'maniqa/export.py',['upstream_forward'],dict(torch=torch))
        def single(rgb):
            image=cv2.resize(rgb,(224,224),interpolation=cv2.INTER_CUBIC)
            x=torch.from_numpy((image.astype(np.float32)/255-.5)/.5).permute(2,0,1).unsqueeze(0)
            return float(namespace['upstream_forward'](model,x).item())
    elif a.metric=='liqe':
        import clip
        sys.path.insert(0,str(sources/'LIQE'))
        original=clip.load
        def local_clip(name,*args,**kwargs):return original(str(root/'ViT-B-32.pt'),*args,**kwargs)
        clip.load=local_clip
        try:model=load(sources/'LIQE/LIQE.py','original_liqe').LIQE(str(root/'LIQE.pt'),'cpu').eval()
        finally:clip.load=original
        def single(rgb):return float(model(tensor(rgb).unsqueeze(0))[0].item())
    elif a.metric=='hyperiqa':
        sys.path.insert(0,str(sources/'hyperIQA'))
        import models
        namespace=functions(exports/'hyperiqa/oracle.py',['build','score'],
            dict(torch=torch,models=models,CKPT=str(root/'hyperiqa-weights')))
        model=namespace['build']()
        def single(rgb):
            image=TF.resize(Image.fromarray(rgb),[512,384],TF.InterpolationMode.BILINEAR)
            x=TF.normalize(TF.to_tensor(image),[.485,.456,.406],[.229,.224,.225])
            batch=torch.stack([x[:,y:y+224,z:z+224] for y in [0,72,144,216,288] for z in [0,40,80,120,160]])
            return namespace['score'](model,batch)
    elif a.metric=='paq2piq':
        sys.path.insert(0,str(sources/'paq2piq'))
        from paq2piq.model import RoIPoolModel
        model=RoIPoolModel().eval()
        state=torch.load(root/'paq2piq-weights/RoIPoolModel-fit.10.bs.120.pth',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model'])
        def single(rgb):
            model.input_block_rois(blk_size=[20,20],img_size=[a.height,a.width],include_image=True,device='cpu')
            return float(model(tensor(rgb).unsqueeze(0))[0,0].item())
    elif a.metric=='wd':
        driver=load(exports/'wd/reference.py','original_wd_driver')
        model=driver.reference(Path(cfg['wd_reference']))
        def score():
            x,y=[model.wd_net(tensor(rgb).unsqueeze(0)) for rgb in [r,d]]
            return driver.scores(model,x,y)[0],None
    if nr:
        def score():return single(d),single(r)
    if a.setup_only:
        print(json.dumps(dict(metric=a.metric,setup='ready',torch=torch.__version__,threads=a.threads)))
        return
    with torch.inference_mode():
        for i in range(a.samples+1):
            start=time.perf_counter_ns();value,reference=score();elapsed=time.perf_counter_ns()-start
            print(('COLD' if i==0 else 'SAMPLE')+f' {elapsed} {value:.17g}',flush=True)
            if reference is not None:print(f'PAIR_REFERENCE {reference:.17g}',flush=True)


if __name__=='__main__':main()
