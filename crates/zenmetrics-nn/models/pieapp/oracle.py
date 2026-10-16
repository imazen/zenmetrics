#!/usr/bin/env python3
"""Run the official public compute_score as an opaque patch oracle.

Only public patch errors and weights are consumed. No model source or internal
features are inspected. Dense sampling follows the release README: valid
64-square patches at stride six, in row-major order, without an extra edge.
"""
import argparse
import csv
import math
from pathlib import Path
import sys
import cv2
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    for name in ('upstream', 'checkpoint', 'pairs', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    a = p.parse_args()
    sys.path.insert(0, a.upstream)
    from model.PieAPPv0pt1_PT import PieAPP
    state = torch.load(a.checkpoint, map_location='cpu')
    models = {}
    print('runtime', torch.__version__, 'device', a.device, flush=True)
    with open(a.pairs) as f:
        pairs = list(csv.DictReader(f, delimiter='\t'))
    with open(a.out, 'x') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerow(['ref_path', 'dist_path', 'pieapp'])
        for row in pairs:
            r = cv2.imread(row['ref_path'])
            d = cv2.imread(row['dist_path'])
            if r is None or d is None or r.shape != d.shape:
                raise ValueError('missing image or incompatible geometry')
            h, w = r.shape[:2]
            if min(h, w) < 64:
                raise ValueError('PieAPP requires both dimensions >= 64')
            positions = [(y, x) for y in range(0, h-63, 6) for x in range(0, w-63, 6)]
            numerator = denominator = 0.0
            for start in range(0, len(positions), 32):
                points = positions[start:start+32]
                n = len(points)
                if n not in models:
                    model = PieAPP(1, n)
                    model.load_state_dict(state)
                    model.eval()
                    if a.device == 'cuda':
                        model.cuda()
                    models[n] = model
                inputs = [torch.from_numpy(np.stack([image[y:y+64, x:x+64].transpose(2, 0, 1) for y, x in points]).astype(np.float32)) for image in (d, r)]
                if a.device == 'cuda':
                    inputs = [value.cuda() for value in inputs]
                def run():
                    outputs = models[n].compute_score(*inputs)
                    return np.stack([outputs[i].detach().cpu().numpy().reshape(-1) for i in (1, 2)], axis=1)
                got = run()
                if a.device == 'cuda':
                    repeated = run()
                    if not np.array_equal(got, repeated):
                        print('GPU repeat mismatch', row, start, got.tolist(), repeated.tolist(), flush=True)
                        third = run()
                        if not np.array_equal(repeated, third):
                            raise ValueError('unreproduced GPU oracle')
                        got = repeated
                for error, weight in got:
                    if not math.isfinite(float(error)) or not math.isfinite(float(weight)) or weight <= 0:
                        raise ValueError('invalid public patch output')
                    numerator += float(error) * float(weight)
                    denominator += float(weight)
            value = numerator / denominator
            writer.writerow([row['ref_path'], row['dist_path'], format(value, '.17e')])
            f.flush()
            print(row['dist_path'], len(positions), format(value, '.17e'), flush=True)


if __name__ == '__main__':
    main()
