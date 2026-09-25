% MS-SSIM goldens — drives Wang's msssim.m + ssim_index_new.m under
% GNU Octave (image package for fspecial/imfilter). Prints
% name w h L msssim. gen.m patterns are shared with the Rust tests.
pkg load image;

K = [0.01 0.03];
win = fspecial('gaussian', 11, 1.5);
W5 = [0.0448 0.2856 0.3001 0.2363 0.1333];

function L = auto_level(w, h)
  m = min(w,h);
  L = 1;
  while m / (2^L) >= 11; L = L + 1; end
  L = min(L, 5);
endfunction

function s = run(im1, im2)
  persistent K2 win2 W52
  K2 = [0.01 0.03]; win2 = fspecial('gaussian', 11, 1.5);
  W52 = [0.0448 0.2856 0.3001 0.2363 0.1333];
  L = auto_level(columns(im1), rows(im1));
  s = msssim(im1, im2, K2, win2, L, W52(1:L), 'product');
endfunction

cases = {
  1 2 64 64    "g1_g2_64";
  1 2 65 63    "g1_g2_65x63";
  4 2 96 80    "g4_g2_96x80";
  1 2 176 176  "g1_g2_176";
  2 3 256 256  "g2_g3_256";
  1 4 300 260  "g1_g4_300x260";
  3 2 37 41    "g3_g2_37x41";
  1 2 16 16    "g1_g2_16";
  1 2 48 40    "g1_g2_48x40";
  5 3 512 512  "g5_g3_512";
};
for i = 1:rows(cases)
  kr = cases{i,1}; kd = cases{i,2}; w = cases{i,3}; h = cases{i,4};
  nm = cases{i,5};
  r = gen(kr,w,h); d = gen(kd,w,h);
  s = run(r, d);
  printf("%s %d %d L=%d msssim=%.12f\n", nm, w, h, auto_level(w,h), s);
endfor

% identical / constant degenerates
s = run(gen(1,256,256), gen(1,256,256));
printf("identical_256 256 256 L=5 msssim=%.12f\n", s);
s = run(128*ones(64,64), 128*ones(64,64));
printf("const_identical_64 64 64 L=3 msssim=%.12f\n", s);
s = run(zeros(64,64), zeros(64,64));
printf("zero_identical_64 64 64 L=3 msssim=%.12f\n", s);
s = run(128*ones(64,64), gen(1,64,64));
printf("const_vs_g1_64 64 64 L=3 msssim=%.12f\n", s);

% The RGB golden lives in rgb_golden.m — Octave's rgb2gray rounds to
% uint8, so it computes the unrounded 0.2989/0.5870/0.1140 luma
% explicitly (what `msssim_rgb8` feeds).
