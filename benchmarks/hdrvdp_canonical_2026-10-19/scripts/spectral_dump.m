% TREE OUTPUT VERSION MODE. MODE=release|cubic|modern.
1; % Octave script with helper defined before its first call.
function emit(f,name,A)
for r=1:size(A,1)
 for c=1:size(A,2), fprintf(f,'%s\t%d\t%d\t%.17g\n',name,r-1,c-1,A(r,c)); end
end
end
a=argv(); addpath(a{1}); addpath(fileparts(mfilename('fullpath')));
version=str2double(a{3}); mode=a{4};
if version==3
 addpath(fullfile(a{1},'utils')); data=fullfile(a{1},'data');
 display='emission_spectra_oled.csv';
else
 data=a{1}; display='emission_spectra_ccfl-lcd.csv';
end
assert(~exist(a{2},'file')); f=fopen(a{2},'w'); assert(f>=0);
fprintf(f,'stage\trow\tcolumn\tvalue\n');
tables={'log_cone_smith_pokorny_1975.csv','cie_scotopic_lum.txt',display};
labels={'cone_log','rod','display'}; sampled=cell(1,3);
for j=1:3
 file=fullfile(data,tables{j}); D=dlmread(file);
 if strcmp(mode,'release')
  [grid,R]=load_spectral_resp(file);
 else
  grid=linspace(360,780,420); R=zeros(420,size(D,2)-1);
  for c=2:size(D,2)
   if strcmp(mode,'modern'), R(:,c-1)=cubic_convolution_control(D(:,1),D(:,c),grid,0);
   else, assert(strcmp(mode,'cubic')); R(:,c-1)=interp1(D(:,1),D(:,c),grid,'cubic',0); end
  end
 end
 sampled{j}=R; emit(f,['raw_' labels{j}],R);
end
emit(f,'grid',grid(:));
C=sampled{1}; C(C==0)=min(C(:)); C=10.^C;
S=[C sampled{2}]; E=sampled{3};
emit(f,'receptors',S); emit(f,'emission',E);
M=zeros(3,4);
for c=1:3
 for r=1:4
  product=S(:,r).*E(:,c);
  if version==3, product=product*683.002; end
  M(c,r)=trapz(grid,product);
 end
end
if version==3, M=M/sum(sum(M(:,1:2))); end
if version==3
 emit(f,'mixing_unaged',M);
 [~,cmf]=load_spectral_resp(fullfile(data,'ciexyz31.csv')); emit(f,'xyz_cmf',cmf);
 addpath(data); emit(f,'xyz_to_native',hdrvdp_iturgb2native('xyz',E));
 % Execute the original release's optical-density block, without copying it.
 source=fileread(fullfile(a{1},'hdrvdp_visual_pathway.m'));
 start=strfind(source,'if( metric_par.do_aod )'); assert(numel(start)==1);
 tail=source(start:end); finish=strfind(tail,'% =================================');
 assert(~isempty(finish)); block=tail(1:finish(1)-1);
 metric_par.do_aod=true; metric_par.age=24; IMG_E=E; lambda=grid;
 eval(block);
 emit(f,'aging24',trans_filter(:)); emit(f,'emission_after_age',IMG_E);
 for c=1:3
  for r=1:4, M(c,r)=trapz(grid,S(:,r).*IMG_E(:,c)*683.002); end
 end
 M=M/sum(sum(M(:,1:2)));
end
emit(f,'mixing',M);
fclose(f);
