% HaarPSI goldens — runs the authors' HaarPSI.m (MIT, rgcda/haarpsi) under
% GNU Octave on deterministic patterns shared with crates/haarpsi tests.
% Prints rows: name w h flag score

cases = {
  % kind_r kind_d w h  name
  1 2 64 64   "g1_vs_g2_64";
  1 2 65 63   "g1_vs_g2_65x63";
  1 4 64 64   "g1_vs_g4_64";
  3 4 96 80   "g3_vs_g4_96x80";
  5 5 64 64   "const_identical";
  1 1 64 64   "g1_identical";
  2 2 40 40   "g2_identical_40";
  4 4 64 64   "g4_identical";
  3 1 32 32   "g3_vs_g1_32";
  5 1 64 64   "const_vs_g1_64";
  2 1 37 41   "g2_vs_g1_37x41";
};

for i = 1:rows(cases)
  kr = cases{i,1}; kd = cases{i,2}; w = cases{i,3}; h = cases{i,4};
  nm = cases{i,5};
  r = gen(kr, w, h); d = gen(kd, w, h);
  s1 = HaarPSI(r, d);
  s0 = HaarPSI(r, d, 0);
  printf("%s %d %d sub=%.12f nosub=%.12f\n", nm, w, h, s1, s0);
endfor

% RGB (YIQ path) — three channel planes per image, packed as HxWx3
rgbcases = {
  1 2 64 64   "rgb_g1_g2_64";
  1 1 48 40   "rgb_identical_48x40";
  4 2 65 63   "rgb_g4_g2_65x63";
};
for i = 1:rows(rgbcases)
  kr = rgbcases{i,1}; kd = rgbcases{i,2}; w = rgbcases{i,3}; h = rgbcases{i,4};
  nm = rgbcases{i,5};
  r = zeros(h,w,3); d = zeros(h,w,3);
  % per-channel patterns: R=gen(k), G=gen(k+offset wrap), B=another
  r(:,:,1) = gen(kr, w, h);            r(:,:,2) = gen(mod(kr,5)+1, w, h);      r(:,:,3) = gen(mod(kr+1,5)+1, w, h);
  d(:,:,1) = gen(kd, w, h);            d(:,:,2) = gen(mod(kd,5)+1, w, h);      d(:,:,3) = gen(mod(kd+1,5)+1, w, h);
  s = HaarPSI(r, d);
  printf("%s %d %d rgb=%.12f\n", nm, w, h, s);
endfor

% Luma path: BT.601 unrounded float luma plane -> grayscale HaarPSI
lumacases = {
  1 2 64 64   "luma_g1_g2_64";
  4 4 48 40   "luma_identical_48x40";
};
for i = 1:rows(lumacases)
  kr = lumacases{i,1}; kd = lumacases{i,2}; w = lumacases{i,3}; h = lumacases{i,4};
  nm = lumacases{i,5};
  for c = [0 1]
    if c == 0, k = kr; else, k = kd; end
    R = double(gen(k, w, h)); G = double(gen(mod(k,5)+1, w, h)); B = double(gen(mod(k+1,5)+1, w, h));
    L = 0.299*R + 0.587*G + 0.114*B;
    if c == 0, Lr = L; else, Ld = L; end
  endfor
  s = HaarPSI(Lr, Ld);
  printf("%s %d %d luma=%.12f\n", nm, w, h, s);
endfor
