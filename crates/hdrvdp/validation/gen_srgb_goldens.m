% Generate HDR-VDP-2.2.2 'sRGB-display' goldens (Q only) for the Rust port's 8-bit sRGB route.
%
% Upstream documents 'sRGB-display' as the encoding "for standard (LDR) color images": an sRGB display
% with a 100 cd/m^2 peak and 1 cd/m^2 black (display_model_srgb: sRGB EOTF, then 99*linear + 1).
% Images are built with integer arithmetic only, so tests/golden_srgb_display.rs regenerates them
% bit-identically and only goldens_srgb.tsv is committed.
%
% PREREQUISITES: the official hdrvdp-2.2.2 release with the Octave ifft2 patch from docs/VALIDATION.md
% (the is_mex patch only affects P_map, which this generator does not record), plus `pkg install image`.
%   HDRVDP_SRC=/path/to/hdrvdp-2.2.2 HDRVDP_GOLDENS=out octave --no-gui validation/gen_srgb_goldens.m
pkg load image
SRCDIR = getenv('HDRVDP_SRC');
if isempty(SRCDIR), SRCDIR = 'hdrvdp-2.2.2'; end
addpath(SRCDIR, fullfile(SRCDIR, 'matlabPyrTools_1.4_fixed'));
outdir = getenv('HDRVDP_GOLDENS');
if isempty(outdir), outdir = 'goldens_srgb'; end
mkdir(outdir);

% Reference: H x W x 3 integer codes in 0..255 from integer-only arithmetic.
function img = reference_codes(H, W)
  img = zeros(H, W, 3);
  for y = 0:H-1
    for x = 0:W-1
      for c = 0:2
        img(y+1, x+1, c+1) = mod(31*x + 17*y + 59*c + floor(x*y/13) + floor((x+2*c)^2/29), 256);
      end
    end
  end
end

% MINSTD generator: every product stays below 2^53, so doubles are exact.
function [v, state] = minstd(state)
  state = mod(state * 48271, 2147483647);
  v = state;
end

function tst = distort(name, ref)
  [H, W, C] = size(ref);
  switch name
    case 'identical'
      tst = ref;
    case 'noise'
      tst = ref; state = 12345;
      for y = 1:H, for x = 1:W, for c = 1:C
        [v, state] = minstd(state);
        tst(y, x, c) = min(255, max(0, ref(y, x, c) + mod(v, 21) - 10));
      end, end, end
    case 'blur'
      tst = ref;
      for y = 1:H, for x = 1:W, for c = 1:C
        s = 0;
        for dy = -1:1, for dx = -1:1
          s = s + ref(min(H, max(1, y+dy)), min(W, max(1, x+dx)), c);
        end, end
        tst(y, x, c) = floor(s / 9);
      end, end, end
    case 'dark'
      tst = floor(ref * 55 / 100);
    case 'bright'
      tst = min(255, floor(ref * 130 / 100) + 8);
    case 'chroma'
      tst = ref;
      tst(:, :, 1) = min(255, ref(:, :, 1) + 20);
  end
end

fid = fopen(fullfile(outdir, 'goldens_srgb.tsv'), 'w');
fprintf(fid, "case\tenc\tw\th\tppd\tq\n");
sizes = {[128, 128], [96, 160]};
names = {'identical', 'noise', 'blur', 'dark', 'bright', 'chroma'};
for si = 1:numel(sizes)
  H = sizes{si}(1); W = sizes{si}(2);
  ref = reference_codes(H, W);
  for ni = 1:numel(names)
    tst = distort(names{ni}, ref);
    for ppd = [30, 60]
      res = hdrvdp(tst / 255, ref / 255, 'sRGB-display', ppd);
      fprintf(fid, "%s\tsrgb-display\t%d\t%d\t%d\t%.17g\n", names{ni}, W, H, ppd, res.Q);
      printf('%s %dx%d ppd=%d Q=%.6f\n', names{ni}, H, W, ppd, res.Q);
      fflush(stdout);
    end
  end
end
fclose(fid);
disp('done');
