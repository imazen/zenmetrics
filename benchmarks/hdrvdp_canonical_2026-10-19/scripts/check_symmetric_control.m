addpath(fileparts(mfilename('fullpath')));
for dims = {[3 5], [4 6], [3 6], [4 5]}
 d=dims{1}; X=reshape(sin(1:prod(d)),d);
 for axis = {'rows','columns'}
  setenv('HDRVDP_FFT_AXIS',axis{1});
  Y=symmetric_ifft_control(fft2(X));
  assert(max(abs(Y(:)-X(:))) < 1e-12);
  setenv('HDRVDP_FFT_AUDIT','0'); Z=symmetric_ifft_control(fft2(X));
  assert(isequal(Y,Z)); setenv('HDRVDP_FFT_AUDIT','1');
 end
end
% Deliberately asymmetric spectrum: positive row wins over conflicting negative.
F=zeros(4,4); F(2,1)=4; F(4,1)=20;
setenv('HDRVDP_FFT_AXIS','rows'); Y=symmetric_ifft_control(F);
assert(max(abs(Y(:,1)-[.5;0;-.5;0])) < 1e-12);
assert(max(abs(Y(:)-real(ifft2(F))(:))) > .9);
% The two possible retention axes differ for an interior conflicting pair.
F=zeros(4,4); F(2,4)=4; F(4,2)=20;
setenv('HDRVDP_FFT_AXIS','rows'); R=symmetric_ifft_control(F);
setenv('HDRVDP_FFT_AXIS','columns'); C=symmetric_ifft_control(F);
assert(max(abs(R(:)-C(:))) > 1);
% MathWorks' documented example.
Y=symmetric_ifft_control([3+1e-15*i 5;5 3]);
assert(max(abs(Y(:)-[4;0;0;-1])) < 1e-12);
disp('PASS: odd/even real spectra, conflicting halves, axes, documented example');
