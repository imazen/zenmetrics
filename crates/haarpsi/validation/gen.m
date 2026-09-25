function img = gen(kind, w, h)
  [x,y] = meshgrid(0:w-1, 0:h-1);
  x = double(x); y = double(y);
  switch kind
    case 1, img = mod(37*x + 91*y + x.*y, 256);
    case 2, img = mod(bitxor(uint64(17*x), uint64(29*y)) + 3*x + 5*y, 256);
    case 3, img = mod(x + 2*y, 256);
    case 4
      img = mod(37*x + 91*y + x.*y, 256);
      sel = x>=20 & x<40 & y>=10 & y<30;
      img(sel) = mod(bitxor(uint64(17*x(sel)), uint64(29*y(sel))) + 3*x(sel) + 5*y(sel), 256);
    case 5, img = 128*ones(h,w);
  end
end
