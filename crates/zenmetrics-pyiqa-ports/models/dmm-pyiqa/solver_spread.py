"""How much of DMM's score depends on the SVD solver, at identical taps.

usage: solver_spread.py PAIRS_TSV N

For the first N pairs: pyiqa 0.1.16 `dmm` taps (its own prepare_image_adt and VGG16-L2pool), then
pyiqa's head computed with each SVD: torch.svd (pyiqa's own call, MKL on CPU), scipy gesdd f32,
scipy gesvd f32, torch.svd in f64. Prints each score and the max spread per pair.
"""
import csv, sys
import numpy as np, torch, scipy.linalg as sl, pyiqa

torch.set_grad_enabled(False)
net = pyiqa.create_metric('dmm', device='cpu', as_loss=False).net.eval()
rows = list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))[: int(sys.argv[2])]


def svd_torch(m):
    u, s, v = torch.svd(m)
    return u, s


def svd_f64(m):
    u, s, v = torch.svd(m.double())
    return u.float(), s.float()


def scipy_svd(driver):
    def f(m):
        a = m.numpy()
        us, ss = [], []
        for i in range(a.shape[0]):
            u, s, _ = sl.svd(a[i], lapack_driver=driver, check_finite=False)
            us.append(u); ss.append(s)
        return torch.from_numpy(np.stack(us)), torch.from_numpy(np.stack(ss))
    return f


def head(df, rf, svd):
    dist = 0.0
    for key in rf:
        td, tr = df[key], rf[key]
        a, b, c, d = td.size(); k = net.patchsize
        dp = net.unfold(td).view(a, b, k, k, -1).transpose(2, 4).contiguous()
        rp = net.unfold(tr).view(a, b, k, k, -1).transpose(2, 4).contiguous()
        pt = dp.shape[2]
        dp = dp.view(a * b * pt, k, k); rp = rp.view(a * b * pt, k, k)
        u_r, s_r = svd(rp); u_d, s_d = svd(dp)
        s_r = s_r.view(a * b, pt, k); s_d = s_d.view(a * b, pt, k)
        diff_s = (s_r - s_d) ** 2
        u_rd = (torch.abs(u_r.view(a, b, pt, k, k) * u_d.view(a, b, pt, k, k))).sum(-1)
        wt = u_rd.std(-1) / (u_rd.mean(-1) + 1e-9)
        diff_s = ((diff_s.view(a, b, pt, k).sum(-1)) * wt).mean(-1).mean(-1)
        xm = td.mean([2, 3], keepdim=True); ym = tr.mean([2, 3], keepdim=True)
        s1 = (2 * xm * ym + 1e-6) / (xm ** 2 + ym ** 2 + 1e-6)
        dist = dist + diff_s * torch.exp(-2.0 * s1.squeeze(-1).squeeze(-1).mean(1))
    return float(dist)


solvers = {'torch(mkl)': svd_torch, 'gesdd_f32': scipy_svd('gesdd'), 'gesvd_f32': scipy_svd('gesvd'), 'torch_f64': svd_f64}
spreads = []
for r in rows:
    ld = net.prepare_image_adt(pyiqa.utils.img_util.imread2tensor(r['dist_path'], rgb=True).unsqueeze(0))
    lr = net.prepare_image_adt(pyiqa.utils.img_util.imread2tensor(r['ref_path'], rgb=True).unsqueeze(0))
    df, rf = net.features_extractor(ld), net.features_extractor(lr)
    v = {n: head(df, rf, f) for n, f in solvers.items()}
    sp = max(v.values()) - min(v.values()); spreads.append(sp)
    print(r['distorted'], ' '.join(f'{n}={x:.9f}' for n, x in v.items()), f'spread={sp:.3e}', flush=True)
print('max spread', max(spreads), 'median', float(np.median(spreads)))
