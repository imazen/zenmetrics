% Independent upstream oracle; no reference implementation is bundled.
% octave --quiet oracle.m TREE PAIRS OUTPUT VERSION PEAK BITS PPD ENCODING MATRIX [OPTION=VALUE ...]
a = argv();
pkg load image;
addpath(a{1});
v = str2double(a{4}); peak = str2double(a{5}); bits = str2double(a{6});
ppd = str2double(a{7}); encoding = a{8}; matrix = a{9};
if ~isempty(getenv('HDRVDP_STATISTICS_PATH'))
  addpath(getenv('HDRVDP_STATISTICS_PATH'));
end
if v == 2
  addpath(fullfile(a{1}, 'matlabPyrTools_1.4_fixed'));
  opts = {};
else
  addpath(fullfile(a{1}, 'utils'));
  addpath(fullfile(a{1}, 'data'));
  if exist(fullfile(a{1}, 'matlabPyrTools'), 'dir')
    addpath(fullfile(a{1}, 'matlabPyrTools'));
  else
    addpath(fullfile(a{1}, 'matlabPyrTools_1.4_fixed'));
  end
  % geomean is supplied by Octave's statistics package.
  if ~isempty(getenv('HDRVDP_STATISTICS_PATH'))
    addpath(getenv('HDRVDP_STATISTICS_PATH'));
  else
    pkg load statistics;
  end
  opts = {'use_gpu', false, 'quiet', true, 'age', 24};
end
for k = 10:numel(a)
  kv = strsplit(a{k}, '='); n = str2double(kv{2});
  if isnan(n), val = kv{2}; else, val = n; end
  opts(end+1:end+2) = {kv{1}, val};
end
M = [0.4124564 0.3575761 0.1804375; 0.2126729 0.7151522 0.0721750; 0.0193339 0.1191920 0.9503041];
if strcmp(matrix, 'iec')
 M = [0.4124 0.3576 0.1805; 0.2126 0.7152 0.0722; 0.0193 0.1192 0.9505];
end
f = fopen(a{2}, 'r'); assert(f >= 0); fgetl(f);
out = fopen(a{3}, 'w'); assert(out >= 0); fprintf(out, 'distorted\tq\tq_jod\n');
while true
 line = fgetl(f); if ~ischar(line), break; end
 fields = strsplit(line, '\t');
 imgs = cell(1,2);
 for j = 1:2
  src = imread(fields{j}); assert(isa(src, 'uint8') && size(src,3) == 3);
  x = double(src)/255;
  if bits > 0, q = 2^bits-1; x = round(x*q)/q; end
  codes = x;
  if strcmp(matrix, 'single'), x = single(x); end
  if ~strcmp(matrix, 'display')
   lo = x <= 0.04045;
   x(lo) = x(lo)/12.92; x(~lo) = ((x(~lo)+0.055)/1.055).^2.4;
   x = double(peak*x);
   if strcmp(matrix, 'gog')
    x = (peak-peak/1000)*codes.^2.2 + peak/1000;
   end
   if strcmp(encoding, 'XYZ')
    if strcmp(matrix, 'octave')
     x = peak*rgb2xyz(codes);
    else
     y = zeros(size(x));
     for c = 1:3
      y(:,:,c) = M(c,1)*x(:,:,1)+M(c,2)*x(:,:,2)+M(c,3)*x(:,:,3);
     end
     x = y;
    end
   elseif strcmp(encoding, 'luminance')
    x = 0.2126*x(:,:,1)+0.7152*x(:,:,2)+0.0722*x(:,:,3);
   end
  end
  imgs{j} = x;
 end
 if v == 2
  res = hdrvdp(imgs{2}, imgs{1}, encoding, ppd, opts); jod = NaN;
 else
  res = hdrvdp3('quality', imgs{2}, imgs{1}, encoding, ppd, opts); jod = res.Q_JOD;
 end
 fprintf(out, '%s\t%.17g\t%.17g\n', fields{3}, res.Q, jod); fflush(out);
 fprintf('%s Q=%.17g\n', fields{3}, res.Q);
end
fclose(f); fclose(out);
if exist('symmetric_ifft_control','file')
 global HDRVDP_FFT_STATS;
 fprintf('FFT_CONTROL calls,abs_defect,relative_defect,row_delta,column_delta,axis_delta:');
 fprintf(' %.17g', HDRVDP_FFT_STATS); fprintf('\n');
end
