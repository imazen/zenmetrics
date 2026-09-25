% VSI goldens — runs the authors' VSI.m (research-licensed, obtain
% separately: cslinzhang.github.io/home/VSI/) under GNU Octave with the
% image package loaded (imresize + fspecial). Prints name w h vsi.
% Deterministic `gen.m` patterns are shared with crates/vsi tests, so
% inputs are bit-identical. Channel assignment matches the Rust tests:
% R = gen(k), G = gen(mod(k,5)+1), B = gen(mod(k+1,5)+1).
pkg load image;

function img = rgbplanes(k, w, h)
  img = zeros(h,w,3);
  img(:,:,1) = gen(k, w, h);
  img(:,:,2) = gen(mod(k,5)+1, w, h);
  img(:,:,3) = gen(mod(k+1,5)+1, w, h);
endfunction

cases = {
  % kind_r kind_d w h  name
  1 2 64 64   "rgb_g1_g2_64";
  1 2 65 63   "rgb_g1_g2_65x63";
  4 2 96 80   "rgb_g4_g2_96x80";
  2 3 40 40   "rgb_g2_g3_40";
  1 1 64 64   "rgb_identical_64";
  4 4 65 63   "rgb_identical_65x63";
  1 2 300 260 "rgb_g1_g2_300x260";
  1 4 520 400 "rgb_g1_g4_520x400";  % F=2 (min 400 -> round(400/256)=2)
  2 1 384 384 "rgb_g2_g1_384";      % F=2 threshold boundary
  3 2 37 41   "rgb_g3_g2_37x41";
};

for i = 1:rows(cases)
  kr = cases{i,1}; kd = cases{i,2}; w = cases{i,3}; h = cases{i,4};
  nm = cases{i,5};
  r = rgbplanes(kr, w, h); d = rgbplanes(kd, w, h);
  s = VSI(uint8(r), uint8(d));
  printf("%s %d %d vsi=%.12f\n", nm, w, h, s);
endfor

% Grayscale content (identical channels) — the reference still requires
% 3 channels; replicated gray exercises the same path.
r = zeros(64,64,3); d = zeros(64,64,3);
r(:,:,1) = gen(1,64,64); r(:,:,2) = r(:,:,1); r(:,:,3) = r(:,:,1);
d(:,:,1) = gen(2,64,64); d(:,:,2) = d(:,:,1); d(:,:,3) = d(:,:,1);
printf("grayrep_g1_g2_64 64 64 vsi=%.12f\n", VSI(uint8(r), uint8(d)));

% Constant / flat inputs — SDSP normalisation degenerates (maxA==minA
% etc.) producing NaN.
s = VSI(uint8(128*ones(64,64,3)), uint8(128*ones(64,64,3)));
printf("const_identical 64 64 vsi=%.12f\n", s);
s = VSI(zeros(64,64,3,'uint8'), zeros(64,64,3,'uint8'));
printf("zero_identical 64 64 vsi=%.12f\n", s);
s = VSI(uint8(128*ones(64,64,3)), uint8(rgbplanes(1,64,64)));
printf("const_vs_rgb_g1 64 64 vsi=%.12f\n", s);
