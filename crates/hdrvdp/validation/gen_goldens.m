% Generate HDR-VDP-2.2.2 goldens for validating the Rust port.
% Images written as raw f64 row-major binaries; scalars to goldens.tsv.
%
% PREREQUISITES (see docs/VALIDATION.md for the full recipe):
%   1. Download hdrvdp-2.2.2.zip from SourceForge and unzip.
%   2. Octave + `pkg install image` (upstream calls padarray).
%   3. Apply the local patches documented in docs/VALIDATION.md —
%      including the `is_mex` fix in reconSpyr.m / reconSpyrLevs.m,
%      without which upstream silently drops 13 of 14 pyramid bands
%      from the visibility map under Octave.
%   4. Set SRCDIR below to the patched source tree.
pkg load image
SRCDIR = getenv('HDRVDP_SRC');
if isempty(SRCDIR), SRCDIR = 'hdrvdp-2.2.2'; end
addpath(SRCDIR, fullfile(SRCDIR, 'matlabPyrTools_1.4_fixed'));
outdir = getenv('HDRVDP_GOLDENS');
if isempty(outdir), outdir = 'goldens'; end
mkdir(outdir);

function write_f64(path, img)
  fid = fopen(path, 'w');
  fwrite(fid, reshape(img.', [], 1), 'double');  % transpose -> row-major stream
  fclose(fid);
end

% 2-D Gaussian kernel (avoids the image package's fspecial).
function k = gauss_kernel(sz, sigma)
  g = exp(-0.5 * ((-(sz-1)/2:(sz-1)/2) / sigma).^2);
  k = g' * g;
  k = k / sum(k(:));
end

fid = fopen(fullfile(outdir, 'goldens.tsv'), 'w');
fprintf(fid, "case\tenc\tw\th\tppd\tq\tp_det\tc_max\tqmos_disabled\n");

rand('seed', 42);
ppds = [30, 60];

sizes = {[128, 128], [192, 160]};
for si = 1:numel(sizes)
  H = sizes{si}(1); W = sizes{si}(2);
  xx = repmat(linspace(0, 1, W), H, 1);
  yy = repmat(linspace(0, 1, H)', 1, W);
  % Textured luminance field, ~1..400 cd/m^2.
  ref = (0.5 + 0.5 * sin(xx * 40) .* cos(yy * 31)) * 150 ...
      + (0.5 + 0.5 * sin(xx * 7 + yy * 3)) * 100 + rand(H, W) * 60 + 1;

  k = gauss_kernel(7, 1.2);
  cases = {
    'identical', ref;
    'noise',     ref + randn(H, W) * 4;
    'blur',      conv2(ref, k, 'same');
    'dark',      ref * 0.55;
    'bright',    ref * 1.5 + 8;
    'contrast',  (ref - mean(ref(:))) * 0.6 + mean(ref(:));
  };

  for pi = 1:numel(ppds)
    ppd = ppds(pi);
    write_f64(fullfile(outdir, sprintf('ref_%dx%d.f64', H, W)), ref);
    for ci = 1:size(cases, 1)
      name = cases{ci, 1}; tst = cases{ci, 2};
      res = hdrvdp(tst, ref, 'luminance', ppd);
      % The Q_MOS logistic upstream disabled in 2.2.1, computed here only
      % for comparison: 100/(1+exp(q1*(Q_raw+q2))) on the RAW correlate.
      Q_raw = 100 - res.Q;
      qmos_disabled = 100 / (1 + exp(3.455 * (Q_raw + 0.8886)));
      write_f64(fullfile(outdir, sprintf('%s_%dx%d_p%d.f64', name, H, W, ppd)), tst);
      write_f64(fullfile(outdir, sprintf('%s_%dx%d_p%d_pmap.f64', name, H, W, ppd)), res.P_map);
      % Diagnostic: dump the D_bands pyramid + S_map for one case.
      if strcmp(name, 'noise') && ppd == 30 && H == 128
        fidb = fopen(fullfile(outdir, 'diag_dpyr.f64'), 'w');
        fwrite(fidb, res.dbg_pyr, 'double');
        fclose(fidb);
        fidb = fopen(fullfile(outdir, 'diag_pind.f64'), 'w');
        fwrite(fidb, res.dbg_pind, 'double');
        fclose(fidb);
        write_f64(fullfile(outdir, 'diag_smap.f64'), res.dbg_smap);
        fprintf('diag dumps: pyr=%d pind=%dx%d S_map=%dx%d\n', ...
          numel(res.dbg_pyr), size(res.dbg_pind), size(res.dbg_smap));
      end
      % Per-plane quality terms for offline pooling-formula comparison.
      fidp = fopen(fullfile(outdir, sprintf('%s_%dx%d_p%d_planes.tsv', name, H, W, ppd)), 'w');
      fprintf(fidp, "band\tori\tw_f\tmsre\n");
      for pi2 = 1:numel(res.qmsres)
        fprintf(fidp, "%d\t%d\t%.17g\t%.17g\n", res.qmsres(pi2).band, ...
          res.qmsres(pi2).ori, res.qmsres(pi2).w_f, res.qmsres(pi2).msre);
      end
      fclose(fidp);
      fprintf(fid, "%s\tluminance\t%d\t%d\t%d\t%.17g\t%.17g\t%.17g\t%.17g\n", ...
        name, W, H, ppd, res.Q, res.P_det, res.C_max, qmos_disabled);
      printf('%s %dx%d ppd=%d  Q=%.6f P_det=%.6f C_max=%.4f\n', ...
        name, H, W, ppd, res.Q, res.P_det, res.C_max);
      fflush(stdout);
    end
  end
end
fclose(fid);
disp('done');
