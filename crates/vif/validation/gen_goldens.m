% VIFp goldens — drives the authors' pixel-domain reference
% `vifp_mscale.m` under GNU Octave (image package for fspecial).
% Prints `name w h vifp`. gen.m patterns are shared with the Rust
% tests.
pkg load image;

cases = {
  1 2 64 64    "g1_g2_64";
  1 2 65 63    "g1_g2_65x63";
  4 2 96 80    "g4_g2_96x80";
  1 2 256 256  "g1_g2_256";
  2 3 256 256  "g2_g3_256";
  1 4 300 260  "g1_g4_300x260";
  3 2 37 41    "g3_g2_37x41";
  1 2 24 24    "g1_g2_24";
  1 2 16 16    "g1_g2_16";
  1 2 48 40    "g1_g2_48x40";
  5 3 512 512  "g5_g3_512";
};
for i = 1:rows(cases)
  kr = cases{i,1}; kd = cases{i,2}; w = cases{i,3}; h = cases{i,4};
  nm = cases{i,5};
  r = gen(kr,w,h); d = gen(kd,w,h);
  s = vifp_mscale(r, d);
  printf("%s %d %d vifp=%.12f\n", nm, w, h, s);
endfor

% identical / constant degenerates
s = vifp_mscale(gen(1,256,256), gen(1,256,256));
printf("identical_256 256 256 vifp=%.12f\n", s);
s = vifp_mscale(128*ones(64,64), 128*ones(64,64));
printf("const_identical_64 64 64 vifp=%.12f\n", s);
s = vifp_mscale(zeros(64,64), zeros(64,64));
printf("zero_identical_64 64 64 vifp=%.12f\n", s);
% constant REFERENCE: sigma1_sq ≡ 0 → den = 0 → 0/0 NaN
s = vifp_mscale(128*ones(64,64), gen(1,64,64));
printf("constref_vs_g1_64 64 64 vifp=%.12f\n", s);
% constant DISTORTED: sigma2_sq ≡ 0 → num = 0 → 0
s = vifp_mscale(gen(1,64,64), 128*ones(64,64));
printf("g1_vs_constdist_64 64 64 vifp=%.12f\n", s);
