function rung(id, a, b, wstep)
  if nargin < 4, wstep = 8; end
  [m,h] = psnrhvsm(a, b, wstep);
  printf('%s %d %d %.12f %.12f\n', id, rows(a), columns(a), m, h);
end
