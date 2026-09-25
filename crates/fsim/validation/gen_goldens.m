% FSIM goldens — runs the authors' FR_FSIMc.m (research-licensed, obtain
% separately: Lin Zhang's FSIM package) under GNU Octave on deterministic
% patterns shared with crates/fsim tests. FR_FSIMc calls fspecial('average')
% — the local fspecial.m shim supplies it. Prints: name w h fsim fsimc

cases = {
  % kind_r kind_d w h  name
  1 2 64 64   "g1_vs_g2_64";
  1 2 65 63   "g1_vs_g2_65x63";
  1 4 64 64   "g1_vs_g4_64";
  3 4 96 80   "g3_vs_g4_96x80";
  1 1 64 64   "g1_identical";
  2 2 40 40   "g2_identical_40";
  4 4 64 64   "g4_identical";
  3 1 32 32   "g3_vs_g1_32";
  2 1 37 41   "g2_vs_g1_37x41";
  1 2 300 260 "g1_vs_g2_300x260";   % F=1 (round(260/256)=1)
  1 4 520 400 "g1_vs_g4_520x400";   % F=2 even-kernel path (round(400/256)=2)
};

for i = 1:rows(cases)
  kr = cases{i,1}; kd = cases{i,2}; w = cases{i,3}; h = cases{i,4};
  nm = cases{i,5};
  r = gen(kr, w, h); d = gen(kd, w, h);
  [fs, fc] = FR_FSIMc(r, d);
  printf("%s %d %d fsim=%.12f fsimc=%.12f\n", nm, w, h, fs, fc);
endfor

% Constant / degenerate pairs
[fs, fc] = FR_FSIMc(128*ones(64,64), 128*ones(64,64));
printf("const_identical 64 64 fsim=%.12f fsimc=%.12f\n", fs, fc);
[fs, fc] = FR_FSIMc(zeros(64,64), zeros(64,64));
printf("zero_identical 64 64 fsim=%.12f fsimc=%.12f\n", fs, fc);
[fs, fc] = FR_FSIMc(128*ones(64,64), gen(1,64,64));
printf("const_vs_g1 64 64 fsim=%.12f fsimc=%.12f\n", fs, fc);

% RGB (YIQ path) — channel assignment matching crates/fsim tests:
% R = gen(k), G = gen(mod(k,5)+1), B = gen(mod(k+1,5)+1)
rgbcases = {
  1 2 64 64   "rgb_g1_g2_64";
  4 2 65 63   "rgb_g4_g2_65x63";
  1 1 48 40   "rgb_identical_48x40";
};
for i = 1:rows(rgbcases)
  kr = rgbcases{i,1}; kd = rgbcases{i,2}; w = rgbcases{i,3}; h = rgbcases{i,4};
  nm = rgbcases{i,5};
  r = zeros(h,w,3); d = zeros(h,w,3);
  r(:,:,1) = gen(kr, w, h);        r(:,:,2) = gen(mod(kr,5)+1, w, h);   r(:,:,3) = gen(mod(kr+1,5)+1, w, h);
  d(:,:,1) = gen(kd, w, h);        d(:,:,2) = gen(mod(kd,5)+1, w, h);   d(:,:,3) = gen(mod(kd+1,5)+1, w, h);
  [fs, fc] = FR_FSIMc(r, d);
  printf("%s %d %d fsim=%.12f fsimc=%.12f\n", nm, w, h, fs, fc);
endfor

% Luma path: Y = 0.299R+0.587G+0.114B unrounded, then grayscale FSIM.
r = zeros(64,64,3); d = zeros(64,64,3);
r(:,:,1) = gen(1,64,64); r(:,:,2) = gen(2,64,64); r(:,:,3) = gen(3,64,64);
d(:,:,1) = gen(2,64,64); d(:,:,2) = gen(3,64,64); d(:,:,3) = gen(4,64,64);
lr = 0.299*r(:,:,1) + 0.587*r(:,:,2) + 0.114*r(:,:,3);
ld = 0.299*d(:,:,1) + 0.587*d(:,:,2) + 0.114*d(:,:,3);
[fs, fc] = FR_FSIMc(lr, ld);
printf("luma_g1_g2_64 64 64 fsim=%.12f fsimc=%.12f\n", fs, fc);
