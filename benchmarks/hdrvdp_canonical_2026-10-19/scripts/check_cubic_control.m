addpath(fileparts(mfilename('fullpath')));
x=(0:5)'; q=linspace(-1,6,101)';
for power=0:2
 y=x.^power; got=cubic_convolution_control(x,y,q,0);
 expected=q.^power; expected(q<x(1)|q>x(end))=0;
 assert(max(abs(got-expected))<1e-12);
 assert(isequal(cubic_convolution_control(x,y,x,0),y));
end
% Interior Keys a=-1/2 negative lobe; PCHIP preserves this flat interval.
y=[0;0;0;1;1;1];
assert(abs(cubic_convolution_control(x,y,1.5,0)+.0625)<1e-15);
assert(interp1(x,y,1.5,'pchip',0)==0);
disp('PASS: constants/lines/quadratics, knots/fill, known interior negative lobe');
