function Y = symmetric_ifft_control(F)
% Diagnostic Hermitian completion, not qualified MATLAB runtime equivalence.
% MathWorks documents retention of the nonnegative-frequency half, without
% specifying the 2-D storage axis. Measure both conventions independently.
global HDRVDP_FFT_STATS;
mode = getenv('HDRVDP_FFT_AXIS');
assert(isempty(mode) || strcmp(mode,'rows') || strcmp(mode,'columns'));
if strcmp(getenv('HDRVDP_FFT_AUDIT'),'0')
 if strcmp(mode,'columns'), H=complete_rows(F.').';
 else, H=complete_rows(F); end
 Y=real(ifft2(H)); return;
end
R = complete_rows(F);
C = complete_rows(F.').';
base = real(ifft2(F));
yr = real(ifft2(R)); yc = real(ifft2(C));
partner = conj(F([1 end:-1:2], [1 end:-1:2]));
values = [1, max(abs(F(:)-partner(:))), ...
          max(abs(F(:)-partner(:)))/max(1,max(abs(F(:)))), ...
          max(abs(yr(:)-base(:))), max(abs(yc(:)-base(:))), ...
          max(abs(yr(:)-yc(:)))];
if isempty(HDRVDP_FFT_STATS), HDRVDP_FFT_STATS = zeros(1,6); end
HDRVDP_FFT_STATS(1) = HDRVDP_FFT_STATS(1)+1;
HDRVDP_FFT_STATS(2:end) = max(HDRVDP_FFT_STATS(2:end), values(2:end));
if strcmp(mode,'columns'), Y = yc;
else, assert(isempty(mode) || strcmp(mode,'rows')); Y = yr; end
end

function H = complete_rows(F)
[m,n] = size(F); H = F;
% DC and (when present) Nyquist rows are themselves Hermitian in columns.
boundary = 1;
if mod(m,2)==0, boundary(end+1)=m/2+1; end
for r = boundary
 H(r,1) = real(H(r,1));
 if mod(n,2)==0, H(r,n/2+1)=real(H(r,n/2+1)); end
 positive = 2:floor((n+1)/2);
 H(r,n-positive+2) = conj(H(r,positive));
end
positive = 2:floor((m+1)/2);
H(m-positive+2,:) = conj(H(positive,[1 n:-1:2]));
end
