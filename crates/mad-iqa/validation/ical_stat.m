function [outStd outSkw outKrt] = ical_stat(x)
% Octave port of the authors' ical_stat.c mex (verbatim semantics).
% 16x16 windows at stride 4 -> blocky 4x4 tiles of sample std
% (ddof=1), skewness and kurtosis normalised by the population
% moments (mean/256, sums/256, stmp = sqrt(sum d^2/256); flat blocks
% yield skw = krt = 0).
[M N] = size(x);
outStd = zeros(M, N);
outSkw = zeros(M, N);
outKrt = zeros(M, N);

for i = 0:4:N-16
  for j = 0:4:M-16
    bx = x(j+1:j+16, i+1:i+16);
    mean = sum(bx(:)) / 256.0;
    d = bx(:) - mean;
    stdev = sqrt(sum(d.^2) / 255.0);
    stmp  = sqrt(sum(d.^2) / 256.0);
    skw = 0; krt = 0;
    if stmp != 0
      skw = (sum(d.^3)/256.0) / (stmp^3);
      krt = (sum(d.^4)/256.0) / (stmp^4);
    endif
    outStd(j+1:j+4, i+1:i+4) = stdev;
    outSkw(j+1:j+4, i+1:i+4) = skw;
    outKrt(j+1:j+4, i+1:i+4) = krt;
  endfor
endfor
end
