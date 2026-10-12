#!/usr/bin/env python3
"""Independent NumPy least-squares oracle on synthetic irregular size panels."""
import json
import numpy as np
from analyze_benchmark_study import linear_fit

panels=[[(64**2,0.001),(256**2,0.012),(512**2,0.031),(1024**2,0.21)],
        [(65536,1001),(196608,3104),(589824,9109),(1769472,28570)],
        [(1000000000+i*1000,1.0+i*0.03) for i in range(5)]]
for panel in panels:
    xs,ys=np.array(panel,dtype=np.float64).T
    center=xs.mean()
    matrix=np.column_stack((np.ones(len(xs)),xs-center))
    coefficients=np.linalg.lstsq(matrix,ys,rcond=None)[0]
    intercept=coefficients[0]-coefficients[1]*center
    fit=linear_fit(panel)
    np.testing.assert_allclose([fit['intercept'],fit['slope']], [intercept,coefficients[1]],rtol=1e-10,atol=1e-9)
    residuals=ys-matrix@coefficients
    np.testing.assert_allclose(fit['residuals'],residuals,rtol=1e-9,atol=1e-9)
print(json.dumps(dict(panels=len(panels),oracle='NumPy centered least-squares',
                     evidence='synthetic numerical controls; no inferential fit qualification')))
