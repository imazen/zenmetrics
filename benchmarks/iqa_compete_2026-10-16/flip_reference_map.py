import argparse,pathlib
p=argparse.ArgumentParser();p.add_argument('width',type=int);p.add_argument('height',type=int);p.add_argument('reference',type=pathlib.Path);p.add_argument('distorted',type=pathlib.Path);p.add_argument('threads',type=int);p.add_argument('output',type=pathlib.Path);a=p.parse_args()
import numpy as np
import flip_evaluator
r,d=[np.fromfile(path,np.uint8).reshape(a.height,a.width,3) for path in [a.reference,a.distorted]]
x,y=r.astype(np.float32)/255,d.astype(np.float32)/255
error,_,_=flip_evaluator.evaluate(x,y,'LDR',inputsRGB=True,applyMagma=False,computeMeanError=False)
assert error.size==a.width*a.height
with a.output.open('xb') as out:out.write(error.astype('<f4').tobytes())
# Float32 row-major accumulation is the distinct FLIP v1.7 scalar convention.
s=np.float32(0)
for value in error.flat:s=np.float32(s+value)
print('FLIP',format(float(error.mean(dtype=np.float64)),'.17e'),'V17',format(float(s/np.float32(error.size)),'.17e'),'MAP',a.width,a.height,flush=True)
