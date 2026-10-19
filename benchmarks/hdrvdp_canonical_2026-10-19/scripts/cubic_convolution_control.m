function v = cubic_convolution_control(x,y,q,fill)
% Diagnostic Keys a=-1/2 convolution, quadratic endpoint extension.
% Models documented modern MATLAB cubic/spline dispatch, not runtime identity.
x=x(:); y=y(:); q=q(:); h=diff(x);
assert(numel(x)>=3 && all(h>0));
if max(abs(h-h(1))) > 1e-10*max(1,abs(h(1)))
 v=interp1(x,y,q,'spline',fill); return;
end
% Three-point one-sided slopes at boundaries; centred slopes inside.
d=[(-3*y(1)+4*y(2)-y(3))/2; (y(3:end)-y(1:end-2))/2; ...
   (3*y(end)-4*y(end-1)+y(end-2))/2]/h(1);
v=ones(size(q))*fill;
for k=1:numel(q)
 if q(k)<x(1) || q(k)>x(end), continue; end
 i=find(x<=q(k),1,'last'); i=min(i,numel(x)-1);
 t=(q(k)-x(i))/h(i);
 v(k)=(2*t^3-3*t^2+1)*y(i)+(t^3-2*t^2+t)*h(i)*d(i) ...
     +(-2*t^3+3*t^2)*y(i+1)+(t^3-t^2)*h(i)*d(i+1);
end
end
