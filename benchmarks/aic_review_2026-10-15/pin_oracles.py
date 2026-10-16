"""Generate the exact Rust unit-test fixtures and score original-author oracles.
Uses the Round 4 pinned sources, never pyiqa. Run with the Round 4 oracle venv.
Only printf precision differs from the original Butteraugli CLI.
"""
import json,pathlib,subprocess,os
import cv2,numpy as np,pywt
from PIL import Image
home=pathlib.Path.home();s=home/'tmp/aic-variants-scratch';old=s/'round4';out=pathlib.Path(os.environ.get('AIC_PIN_RESULTS_DIR',str(s/'round5')));out.mkdir(exist_ok=True)
butter=old/'oracles/butteraugli-71b18b636b9c7d1ae0c1d3730b85b3c127eb4511/butteraugli'
source=(butter/'butteraugli_main.cc').read_text().replace('printf("%lf\\n", diff_value);','printf("%.17g\\n", diff_value);')
(out/'butteraugli_main_precise.cc').write_text(source)
subprocess.run(['g++','-O2','-std=c++11','-I'+str(butter.parent),str(out/'butteraugli_main_precise.cc'),str(butter/'butteraugli.cc'),'-lpng','-ljpeg','-o',str(out/'butter-oracle')],check=True)
# SSIMULACRA also prints eight decimal places by default; print full precision.
source=(old/'ssimulacra-build.cpp').read_text().replace('%.8f','%.17g')
(out/'ssimulacra_precise.cpp').write_text(source)
subprocess.run(['g++','-O2','-std=c++11','-I/usr/include/opencv4',str(out/'ssimulacra_precise.cpp'),'-lopencv_core','-lopencv_imgproc','-lopencv_imgcodecs','-o',str(out/'ssim-oracle')],check=True)
nb=json.load(open(old/'WS_HV.ipynb'));function=next(''.join(c['source']) for c in nb['cells'] if ''.join(c.get('source',[])).startswith('def IQA_WS_HV'));ns={'np':np,'cv2':cv2,'pywt':pywt};exec(function,ns)
results=[]
for w,h in [(64,64),(37,41)]:
 r=np.array([((x*13+y*17+c*71+(x*y)%29)%256) for y in range(h) for x in range(w) for c in range(3)],dtype=np.uint8).reshape(h,w,3)
 d=r.copy().reshape(-1);d[::17]^=63;d=d.reshape(h,w,3)
 Image.fromarray(r).save(out/'pin-r.png');Image.fromarray(d).save(out/'pin-d.png');r.tofile(out/f'r-{w}.rgb');d.tofile(out/f'd-{w}.rgb')
 row=dict(width=w,height=h,butter=float(subprocess.check_output([str(out/'butter-oracle'),str(out/'pin-r.png'),str(out/'pin-d.png')])),ssim1=float(subprocess.check_output([str(out/'ssim-oracle'),str(out/'pin-r.png'),str(out/'pin-d.png')])),ws=float(ns['IQA_WS_HV'](r,d)))
 results.append(row)
# Non-zero banding controls in 10-bit limited-range luma.
w=h=256
binary=old/'oracles/vmaf-3.2.1/libvmaf/build-aic/tools/vmaf'
for block,step in [(8,1),(16,4)]:
 y=np.array([256+(x//block)*step for yy in range(h) for x in range(w)],dtype='<u2')
 with (out/'banded.yuv').open('wb') as f:f.write(y.tobytes());f.write(np.full(w*h//2,512,dtype='<u2').tobytes())
 subprocess.run([str(binary),'-r',str(out/'banded.yuv'),'-d',str(out/'banded.yuv'),'-w',str(w),'-h',str(h),'-p','420','-b','10','--feature','cambi','--no_prediction','--json','-o',str(out/'cambi-pin.json')],check=True)
 results.append(dict(block=block,step=step,cambi=json.load(open(out/'cambi-pin.json'))['frames'][0]['metrics']['cambi']))
with (out/'oracle-pins-banded.json').open('x') as f:json.dump(results,f,indent=2)
print(json.dumps(results,indent=2))
