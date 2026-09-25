function h = fspecial(type, arg)
% Minimal shim: FR_FSIMc only needs fspecial('average', F) = ones(F,F)/F^2
% (GNU Octave's image package isn't installed; 'average' is trivial).
  if ~strcmp(type, 'average')
    error('fspecial shim only implements ''average''');
  endif
  if nargin < 2, arg = 3; endif
  h = ones(arg, arg) / (arg * arg);
endfunction
