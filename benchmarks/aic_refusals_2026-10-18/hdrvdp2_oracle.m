% Usage: octave --quiet hdrvdp2_oracle.m ORACLE_DIR PAIRS_TSV OUTPUT_TSV
% Independent HDR-VDP-2.2.2 oracle. No reference implementation is bundled.
args = argv();
pkg load image;
addpath(args{1});
addpath(fullfile(args{1}, 'matlabPyrTools_1.4_fixed'));
f = fopen(args{2}, 'r');
fgetl(f);
out = fopen(args{3}, 'w');
fprintf(out, 'distorted\tq\n');
while true
  line = fgetl(f);
  if ~ischar(line), break; end
  fields = strsplit(line, '\t');
  r = double(imread(fields{1})) / 255;
  d = double(imread(fields{2})) / 255;
  for which = 1:2
    if which == 1, v = r; else, v = d; end
    lo = v <= 0.04045;
    v(lo) = v(lo) / 12.92;
    v(~lo) = ((v(~lo) + 0.055) / 1.055) .^ 2.4;
    if which == 1, r = 200*v; else, d = 200*v; end
  end
  result = hdrvdp(d, r, 'rgb-bt.709', 37.84);
  fprintf(out, '%s\t%.17g\n', fields{3}, result.Q);
  fflush(out);
  fprintf('%s Q=%.17g\n', fields{3}, result.Q);
end
fclose(out); fclose(f);
