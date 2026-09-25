% Verify our port of octave-image's conv_interp_vec (bilinear + AA on
% shrink + symmetric padding) against the real imresize.
pkg load image;

function w = tri(h)
  w = (1 - abs(h)) .* (abs(h) <= 1);
endfunction

function idx = sympad(i, sz)
  idx = i - 1;
  m = mod(idx, sz);
  odd = mod(floor(idx / sz), 2) == 1;
  idx(odd) = sz - m(odd);
  idx(!odd) = m(!odd) + 1;
endfunction

function out = our_resize1d(vec, out_n)
  in_n = numel(vec);
  scale = out_n / in_n;
  off = 1 / scale / 2;
  ZI = 0.5 + off + (0:out_n-1) / scale;   % 1-based coords
  if scale < 1
    ksize = 2 / scale;
    kern = @(h) scale * tri(scale * h);
  else
    ksize = 2;
    kern = @tri;
  endif
  pad = ceil(ksize / 2) + 2;
  idx0 = floor(ZI); DZ = ZI - idx0;
  out = zeros(1, out_n);
  for o = 1:out_n
    acc = 0; ws = 0;
    for shift = 1-pad:pad
      w = kern(shift - DZ(o));
      if w == 0, continue; endif
      acc = acc + w * vec(sympad(idx0(o) + shift, in_n));
      ws = ws + w;
    endfor
    if ws == 0, ws = 1; endif
    out(o) = acc / ws;
  endfor
endfunction

function out = our_imresize2(A, out_sz)
  [h, w] = size(A);
  tmp = zeros(h, out_sz(2));
  if out_sz(2) == w
    tmp = A;
  else
    for i = 1:h
      tmp(i,:) = our_resize1d(A(i,:), out_sz(2));
    endfor
  endif
  out = zeros(out_sz(1), out_sz(2));
  if out_sz(1) == h
    out = tmp;
  else
    for j = 1:out_sz(2)
      out(:,j) = our_resize1d(tmp(:,j)', out_sz(1))';
    endfor
  endif
endfunction

rng(42);
maxerr = 0;
pairs = [64 256; 65 256; 63 256; 300 256; 520 256; 256 256; 256 64; 256 65; 256 63; 256 300; 256 520; 100 37; 37 100; 511 256; 256 511; 384 256; 256 384];
for k = 1:rows(pairs)
  insz = pairs(k,1); outsz = pairs(k,2);
  A = rand(insz, insz);
  ref = imresize(A, [outsz outsz], 'bilinear');
  got = our_imresize2(A, [outsz outsz]);
  e = max(abs(ref(:) - got(:)));
  printf('%d -> %d: %e\n', insz, outsz, e);
  maxerr = max(maxerr, e);
endfor
printf('MAX %e\n', maxerr);
