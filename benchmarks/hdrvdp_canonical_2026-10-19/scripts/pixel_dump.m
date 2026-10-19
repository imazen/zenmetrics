% Preserve exactly the uint8 arrays consumed by the Octave oracle.
a=argv(); pkg load image;
assert(~exist(a{2},'dir')); mkdir(a{2});
f=fopen(a{1},'r'); assert(f>=0);
out=fopen(fullfile(a{2},'manifest.tsv'),'w'); assert(out>=0);
fprintf(out,'name\twidth\theight\tbytes\n');
while true
 p=fgetl(f); if ~ischar(p), break; end
 x=imread(p); assert(isa(x,'uint8') && ndims(x)==3 && size(x,3)==3);
 [~,name,ext]=fileparts(p); name=[name ext];
 target=fullfile(a{2},[name '.rgb']); assert(~exist(target,'file'));
 g=fopen(target,'wb'); assert(g>=0);
 n=fwrite(g,permute(x,[3 2 1]),'uint8'); fclose(g);
 assert(n==numel(x));
 fprintf(out,'%s\t%d\t%d\t%d\n',name,size(x,2),size(x,1),n);
end
fclose(f); fclose(out);
