% MAD goldens — drives the authors' `hi_index.m` + `lo_index.m`
% (Larson & Chandler, JEI 19(1) 2010, via the STMAD_2011 distribution
% archived in Netflix/vmaf) with ical_std.m / ical_stat.m shims ported
% verbatim from the release's C mex sources. Prints
% `name w h hi lo mad`. The combine is the JEI paper's geometric
% blend: sig = 1/(1+b1*HI^b2), MAD = HI^sig * LO^(1-sig).
pkg load image;

b1 = exp(-2.55/3.35);
b2 = 1/(log(10)*3.35);

cases = {
  1 2 64 64    "g1_g2_64";
  1 2 65 63    "g1_g2_65x63";
  4 2 96 80    "g4_g2_96x80";
  1 2 256 256  "g1_g2_256";
  2 3 256 256  "g2_g3_256";
  1 4 300 260  "g1_g4_300x260";
  3 2 37 41    "g3_g2_37x41";
  1 2 48 40    "g1_g2_48x40";
  5 3 512 512  "g5_g3_512";
  1 2 34 34    "g1_g2_34";
};
for i = 1:rows(cases)
  kr = cases{i,1}; kd = cases{i,2}; w = cases{i,3}; h = cases{i,4};
  nm = cases{i,5};
  r = gen(kr,w,h); d = gen(kd,w,h);
  hi = hi_index(r,d); lo = lo_index(r,d);
  sig = 1/(1+b1*hi^b2);
  s = hi^sig * lo^(1-sig);
  printf("%s %d %d hi=%.12f lo=%.12f mad=%.12f\n", nm, w, h, hi, lo, s);
endfor

% identical / constant degenerates (all zero — MAD is a distance)
r = gen(1,256,256);
hi = hi_index(r,r); lo = lo_index(r,r);
sig = 1/(1+b1*hi^b2);
printf("identical_256 256 256 hi=%.12f lo=%.12f mad=%.12f\n", hi, lo, hi^sig*lo^(1-sig));
c = 128*ones(64,64);
hi = hi_index(c,c); lo = lo_index(c,c);
sig = 1/(1+b1*hi^b2);
printf("const_identical_64 64 64 hi=%.12f lo=%.12f mad=%.12f\n", hi, lo, hi^sig*lo^(1-sig));
% constant reference vs textured distorted
hi = hi_index(c, gen(1,64,64)); lo = lo_index(c, gen(1,64,64));
sig = 1/(1+b1*hi^b2);
printf("constref_vs_g1_64 64 64 hi=%.12f lo=%.12f mad=%.12f\n", hi, lo, hi^sig*lo^(1-sig));
