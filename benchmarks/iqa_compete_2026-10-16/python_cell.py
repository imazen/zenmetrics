#!/usr/bin/env python3
"""One resident numpy/torch metric cell. Import/model/tensor setup is untimed."""
import argparse,importlib.metadata,json,os,pathlib,statistics,subprocess,sys,time

# Owner constraint: fail before any direct or transitive PyIQA import.
class _NoPyIQA:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.partition('.')[0].lower()=='pyiqa':
   raise ModuleNotFoundError('PyIQA is excluded from this benchmark')
if any(name.partition('.')[0].lower()=='pyiqa' for name in sys.modules):
 raise RuntimeError('Excluded PyIQA module was already loaded at process startup')
sys.meta_path.insert(0,_NoPyIQA())

P=pathlib.Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('config',type=P);p.add_argument('--mode',choices=['score','score+map']);a=p.parse_args();c=json.loads(a.config.read_text())
if a.mode is None:
 for selected in ['score','score+map']:
  result=subprocess.run([sys.executable,str(P(__file__).absolute()),str(a.config.absolute()),'--mode',selected])
  if result.returncode:raise SystemExit(result.returncode)
 raise SystemExit
c['mode']=a.mode
before=list(os.getloadavg()) if hasattr(os,'getloadavg') else []
if before and before[0]>4:print('IQA_BUSY '+json.dumps(before),flush=True);raise SystemExit(75)
# Package choices are explicit; PyIQA is excluded.
package=c['package'];metric=c['metric_name'];mode=c['mode'];device=c.get('device','cpu');threads=c['threads']
import numpy as np
r=np.fromfile(c['reference'],dtype=np.uint8).reshape(c['height'],c['width'],3);d=np.fromfile(c['distorted'],dtype=np.uint8).reshape(r.shape)
version=importlib.metadata.version(package)
row=dict(c,impl=package,impl_version=version,venv=str(P(os.sys.executable).parent.parent),package_versions={'numpy':np.__version__},pyiqa_import_blocked=True,n=0)
sync=lambda:None
if package=='flip-evaluator':
 import flip_evaluator
 x=r.astype(np.float32)/255.;y=d.astype(np.float32)/255.
 fn=lambda:flip_evaluator.evaluate(x,y,'LDR',inputsRGB=True,applyMagma=False,computeMeanError=True)
 score=lambda value:float(value[1])
 row['native_api_note']='flip-evaluator returns its raw error map in both score and score+map modes; both are the same public compute call'
elif package=='scikit-image':
 from skimage.metrics import structural_similarity,peak_signal_noise_ratio
 if metric=='psnr':
  if mode=='score+map':row.update(status='unsupported',notes='PSNR API has no native map');print('IQA_ROW '+json.dumps(row));raise SystemExit
  fn=lambda:peak_signal_noise_ratio(r,d,data_range=255)
 else:fn=lambda:structural_similarity(r,d,data_range=255,channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,full=mode=='score+map')
 score=lambda value:float(value[0] if isinstance(value,tuple) else value)
else:
 import torch
 torch.set_num_threads(threads);torch.set_num_interop_threads(1)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 row['package_versions'].update(torch=torch.__version__,torchvision=importlib.metadata.version('torchvision'))
 row.update(torch_threads_observed=torch.get_num_threads(),tf32_matmul=False,tf32_cudnn=False)
 x=torch.from_numpy(r.copy()).permute(2,0,1).unsqueeze(0).contiguous().to(device=device,dtype=torch.float32)/255
 y=torch.from_numpy(d.copy()).permute(2,0,1).unsqueeze(0).contiguous().to(device=device,dtype=torch.float32)/255
 row.update(tensor_dtype=str(x.dtype),tensor_layout='contiguous NCHW',tensor_strides=list(x.stride()))
 if device=='cuda':sync=torch.cuda.synchronize
 maps=(package=='torchmetrics' and metric=='ssimc-pyiqa-f32') or package=='cvvdp'
 if mode=='score+map' and not maps:
  row.update(status='unsupported',notes='No qualified native-map output on this public metric entry point');print('IQA_ROW '+json.dumps(row));raise SystemExit
 if package=='piq':
  import piq
  if metric=='lpips-vgg':
   model=piq.LPIPS(replace_pooling=False,distance='mse',reduction='mean').to(device).eval();fn=lambda:model(x,y)
   row['native_api_note']='PIQ public LPIPS VGG16, default ImageNet normalization of resident RGB [0,1]; model construction excluded'
  elif metric=='dists':
   model=piq.DISTS(reduction='mean').to(device).eval();fn=lambda:model(x,y)
   row['native_api_note']='PIQ public DISTS distance (1-DISTS), default ImageNet normalization; model construction excluded'
  elif metric=='ssimc-pyiqa-f32':fn=lambda:piq.ssim(x,y,data_range=1.,downsample=False)
  elif metric=='ms-ssim-pyiqa-f32':
   # Public function accepts gray tensors. Wrapper luma conversion remains
   # inside timing, using the named unrounded 0.299/0.587/0.114 convention.
   weights=torch.tensor([.299,.587,.114],device=device).view(1,3,1,1)
   fn=lambda:piq.multi_scale_ssim((x*weights).sum(1,keepdim=True),(y*weights).sum(1,keepdim=True),data_range=1.)
  elif metric in ['fsim','fsimc']:
   fn=lambda:piq.fsim(x,y,data_range=1.,chromatic=metric=='fsimc')
   row['reference_declared_columns']=c.get('declared_columns');row['declared_columns']=[metric]
   row['native_api_note']='PIQ returns the requested scalar; the Rust fsim call also computes/returns FSIMc. This API workload difference is retained, not hidden by reimplementing either library.'
  elif metric=='iwssim-piq':fn=lambda:piq.information_weighted_ssim(x,y,data_range=1.)
  elif metric=='vif':fn=lambda:piq.vif_p(x,y,data_range=1.)
  elif metric=='ms-gmsd':fn=lambda:piq.multi_scale_gmsd(x,y,data_range=1.)
  elif metric=='brisque':
   import importlib
   module=importlib.import_module('piq.brisque');original_loader=module.load_url;cached={}
   def load_parameters(url,**kwargs):
    key=(url,str(kwargs.get('map_location')))
    if key not in cached:cached[key]=tuple(v.clone() for v in original_loader(url,**kwargs))
    # _score_svr transposes its loaded support-vector tensor in place.
    # Preserve the original owned-value semantics by cloning cached parameters.
    return tuple(v.clone() for v in cached[key])
   fn=lambda:torch.stack([piq.brisque(y,data_range=1.),piq.brisque(x,data_range=1.)])
   torch.set_grad_enabled(False);original_score=fn().detach().clone()
   module.load_url=load_parameters
   cached_score=fn()
   assert torch.equal(original_score.view(torch.int32),cached_score.view(torch.int32)), 'BRISQUE preload changed bits'
   row.update(model_preload_adapter='cached loader returns owned parameter clones; no SVM arithmetic changed',model_preload_bit_identity=True)

  else:fn=lambda:getattr(piq,metric)(x,y,data_range=1.)
 elif package=='torchmetrics':
  from torchmetrics.functional.image import structural_similarity_index_measure,multiscale_structural_similarity_index_measure,peak_signal_noise_ratio
  if metric=='psnr':fn=lambda:peak_signal_noise_ratio(x,y,data_range=1.)
  elif metric=='ssimc-pyiqa-f32':fn=lambda:structural_similarity_index_measure(x,y,data_range=1.,return_full_image=mode=='score+map')
  else:
   weights=torch.tensor([.299,.587,.114],device=device).view(1,3,1,1)
   fn=lambda:multiscale_structural_similarity_index_measure((x*weights).sum(1,keepdim=True),(y*weights).sum(1,keepdim=True),data_range=1.)
 elif package=='pytorch-msssim':
  import pytorch_msssim
  if metric=='ssimc-pyiqa-f32':fn=lambda:pytorch_msssim.ssim(x,y,data_range=1.,size_average=True,nonnegative_ssim=True)
  else:
   weights=torch.tensor([.299,.587,.114],device=device).view(1,3,1,1)
   fn=lambda:pytorch_msssim.ms_ssim((x*weights).sum(1,keepdim=True),(y*weights).sum(1,keepdim=True),data_range=1.,size_average=True)
 elif package=='lpips':
  import lpips
  model=lpips.LPIPS(net=metric.split('-')[-1],verbose=False).to(device).eval();x=x*2-1;y=y*2-1
  fn=lambda:model(x,y)
 elif package=='stlpips-pytorch':
  import types,sys
  debug=types.ModuleType('IPython');debug.embed=lambda *args,**kwargs:None;sys.modules.setdefault('IPython',debug)
  from stlpips_pytorch import stlpips
  model=stlpips.LPIPS(net=metric.split('-')[-1],variant='shift_tolerant',verbose=False).to(device).eval();x=x*2-1;y=y*2-1
  fn=lambda:model(x,y)
 elif package=='DeepDC-PyTorch':
  import types,sys
  sys.modules.setdefault('imp',types.ModuleType('imp'))
  from DeepDC_PyTorch import DeepDC
  model=DeepDC().to(device).eval()
  fn=lambda:model(x,y)
  row['native_api_note']='Official README DeepDC() call on resident RGB [0,1], default model configuration and no resize. Python-3.12 unused imp import receives an empty compatibility module.'
 elif package=='cvvdp':
  import pycvvdp
  model=pycvvdp.cvvdp(display_name='standard_fhd',heatmap='raw' if mode=='score+map' else None,device=torch.device(device))
  fn=lambda:model.predict(y,x,dim_order='BCHW')
  row['native_api_note']='Official standard_fhd; image prediction from resident tensors; raw native heatmap when requested'
 elif package=='DISTS-pytorch':
  import DISTS_pytorch
  model=DISTS_pytorch.DISTS().to(device).eval();fn=lambda:model(x,y,batch_average=True)
 else:raise ValueError('unqualified package '+package)
 score=lambda value:float((value[0] if isinstance(value,tuple) else value).flatten()[0].item())
 torch.set_grad_enabled(False)
sync();warm=fn();sync();row['warmup_score']=score(warm);del warm
samples=[];scores=[];before=list(os.getloadavg()) if hasattr(os,'getloadavg') else []
for _ in range(10):
 sync();start=time.perf_counter_ns();value=fn();sync();observed=score(value);del value;elapsed=time.perf_counter_ns()-start
 samples.append(elapsed);scores.append(observed)
 print('IQA_SAMPLE '+json.dumps({'ns':elapsed,'score':observed}),flush=True)
 if len(samples)>=3 and sum(samples)>=3_000_000_000:break
# Scalar extraction and output destruction are included, as on the Rust
# adapters returning f64 after releasing their native outputs.
sorted_samples=sorted(samples)
def percentile(p):
 i=p*(len(samples)-1);lo=int(i);hi=min(lo+1,len(samples)-1);return sorted_samples[lo]+(sorted_samples[hi]-sorted_samples[lo])*(i-lo)
row.update(status='ok',n=len(samples),timed_seconds=sum(samples)/1e9,median_ns=statistics.median(samples),mean_ns=statistics.mean(samples),p10_ns=percentile(.1),p90_ns=percentile(.9),score=scores[-1],sample_scores=scores,samples_ns=samples,load_before=before,load_after=list(os.getloadavg()) if hasattr(os,'getloadavg') else [],notes='Resident native ndarray/tensors. Imports, model load, normalization/upload excluded. Callable including its internal color conversion + device synchronization timed; scalar extraction/output destruction included. One warmup; ten calls or at least three calls totaling three seconds.')
print('IQA_ROW '+json.dumps(row,allow_nan=False),flush=True)
