function [outStd outStdMod outMean] = ical_std(x, y)
% Octave port of the authors' ical_std.c mex (verbatim semantics).
% x = dst-ref (CSF-filtered), y = ref (CSF-filtered).
% All maps are blocky 4x4 tiles written at stride-4 16x16 windows
% over the covered grid; positions outside coverage stay 0.
% Column-major indexing mirrors the C (i = col, j = row).
[M N] = size(x);
outStd = zeros(M, N);
outStdMod = zeros(M, N);
outMean = zeros(M, N);
TMP = zeros(M, N);

% Pass 1: 16x16 windows at stride 4 -> std of x (ddof=1), mean of y.
for i = 0:4:N-16
  for j = 0:4:M-16
    bx = x(j+1:j+16, i+1:i+16);
    by = y(j+1:j+16, i+1:i+16);
    mean  = sum(bx(:)) / 256.0;
    mean2 = sum(by(:)) / 256.0;
    stdev = sqrt(sum((bx(:)-mean).^2) / 255.0);
    outStd(j+1:j+4, i+1:i+4) = stdev;
    outMean(j+1:j+4, i+1:i+4) = mean2;
  endfor
endfor

% Pass 2: 8x8 windows at the same stride-4 grid -> std of y (ddof=1).
for i = 0:4:N-16
  for j = 0:4:M-16
    by = y(j+1:j+8, i+1:i+8);
    mean = sum(by(:)) / 64.0;
    stdev = sqrt(sum((by(:)-mean).^2) / 63.0);
    TMP(j+1:j+4, i+1:i+4) = stdev;
    outStdMod(j+1:j+4, i+1:i+4) = stdev;
  endfor
endfor

% Pass 3: min-pool over {0,5} offsets, bounds-checked positions.
for i = 0:4:N-16
  for j = 0:4:M-16
    mean = TMP(j+1, i+1);
    for iB = i:5:i+7
      for jB = j:5:j+7
        if iB < N-15 && jB < M-15 && mean > TMP(jB+1, iB+1)
          mean = TMP(jB+1, iB+1);
        endif
      endfor
    endfor
    outStdMod(j+1:j+4, i+1:i+4) = mean;
  endfor
endfor
end
