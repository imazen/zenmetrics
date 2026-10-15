"""Coarse-only raised-cosine pyramid; independent check of grid and pooling.
Based on the Simoncelli frequency construction and MIT pyrtools 1.0.11.
"""
import numpy as np
from scipy.ndimage import uniform_filter

def coarse(a, grid='pyrtools',analytic=False):
    h,w=a.shape
    if grid=='fftfreq': xx=np.fft.fftshift(np.fft.fftfreq(w))*2; yy=np.fft.fftshift(np.fft.fftfreq(h))*2
    elif grid=='inclusive':xx=np.linspace(-1,1,w); yy=np.linspace(-1,1,h)
    else:xx=np.linspace(-1,1,w+1)[:-1]; yy=np.linspace(-1,1,h+1)[:-1]
    x,y=np.meshgrid(xx,yy); angle=np.arctan2(y,x);rad=np.sqrt(x*x+y*y)
    cy,cx=(np.ceil((np.array([h,w])+.5)/2).astype(int)-1);rad[cy,cx]=rad[cy,cx-1];rad=np.log2(rad)
    hi=np.abs(np.cos(np.pi*np.arange(-257,2)/512));hi[0]=hi[1];hi[-1]=hi[-2];lo=np.sqrt(1-hi**2)
    def radial(r,shift,low=False):
        if analytic:
            high=np.cos(np.pi/2*np.clip(r+shift,-1,0))
            return np.sqrt(np.maximum(0,1-high**2)) if low else high
        return np.interp(r,np.arange(-257,2)/256-shift,lo if low else hi)
    a=np.fft.fftshift(np.fft.fft2(a))*radial(rad,0,True)
    for level in range(3):
        dims=np.array(a.shape); ctr=np.ceil((dims+.5)/2).astype(int);ld=np.ceil((dims-.5)/2).astype(int);lc=np.ceil((ld+.5)/2).astype(int);s=ctr-lc
        sl=(slice(s[0],s[0]+ld[0]),slice(s[1],s[1]+ld[1]));a=a[sl];rad=rad[sl];angle=angle[sl]
        a=a*radial(rad,level+1,True)
    return a,angle,radial(rad,4)

def bands(c,analytic=False):
    a,angle,rad=c;order=7
    import math
    const=2**14*math.factorial(7)**2/(8*math.factorial(14))
    xc=np.pi*np.arange(-2049,1026)/1024
    alpha=(np.pi+xc)%(2*np.pi)-np.pi
    yc=2*np.sqrt(const)*np.cos(xc)**7*(np.abs(alpha)<np.pi/2)
    for ori in range(8):
        if analytic:
            alpha=(angle-np.pi*ori/8+np.pi)%(2*np.pi)-np.pi
            ang=2*np.sqrt(const)*np.cos(alpha)**7*(np.abs(alpha)<np.pi/2)
        else:ang=np.interp(angle,xc+np.pi*ori/8,yc)
        yield np.fft.ifft2(np.fft.ifftshift(1j*a*rad*ang))

def similarities(a,b):
    values={}
    for x,y in zip(a,b):
        c=x*np.conj(y);corr=uniform_filter(c.real,7,mode='constant')+1j*uniform_filter(c.imag,7,mode='constant');en=uniform_filter(np.abs(x)**2+np.abs(y)**2,7,mode='constant')
        q=np.divide(2*np.abs(corr),en,out=np.ones_like(en),where=en>0)[3:-3,3:-3]
        h,w=q.shape;yy=np.arange(h)-(h-1)/2;xx=np.arange(w)-(w-1)/2
        for name,sigma in [('full',x.shape[0]/4),('crop',h/4)]:
            weight=np.exp(-(yy[:,None]**2+xx[None,:]**2)/(2*sigma*sigma))
            values.setdefault(name,[]).append(float((q*weight).sum()/weight.sum()))
    return {k:float(np.mean(v)) for k,v in values.items()}

if __name__=='__main__':
    import csv,json,pathlib,sys
    from PIL import Image
    home=pathlib.Path.home();pub={r['distorted']:r for r in csv.DictReader(open(home/'tmp/aic-site/site_fullres_subset61.csv'))};rows=[]
    for pair in csv.DictReader(open(home/'tmp/aic-audit/pairs.tsv'),delimiter='\t'):
        if len(sys.argv)>2 and pair['distorted'] not in sys.argv[2:]:continue
        row={'key':pair['distorted'],'published':float(pub[pair['distorted']]['CW-SSIM'])}
        r=np.asarray(Image.open(pair['ref_path']).convert('RGB'),dtype=np.float64);d=np.asarray(Image.open(pair['dist_path']).convert('RGB'),dtype=np.float64)
        r=np.round(r@np.array([.299,.587,.114]));d=np.round(d@np.array([.299,.587,.114]))
        for grid in ['pyrtools','fftfreq','inclusive']:
            for analytic in [False,True]:
                a=coarse(r,grid,analytic);b=coarse(d,grid,analytic)
                for name,value in similarities(bands(a,analytic),bands(b,analytic)).items():row[f'{grid}_{analytic}_{name}']=value
        rows.append(row);print(json.dumps(row),flush=True)
    pathlib.Path(sys.argv[1]).open('x').write(json.dumps(rows,indent=2)+'\n')
    for k in rows[0]:
        if k in ['key','published']:continue
        delta=[abs(r[k]-r['published']) for r in rows];print('SUMMARY',k,np.median(delta),max(delta),flush=True)
