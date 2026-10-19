"""Supplement launcher hashes with resolved tool executables; never runs a metric."""
import hashlib,json,subprocess,sys,shutil
from pathlib import Path

def capture(root, output):
    commands=[]
    def query(argv):
        value=subprocess.check_output(argv,text=True).strip()
        commands.append(dict(argv=argv,stdout=value))
        return value
    paths={}
    for tool in ('rustc','cargo'):
        paths[tool]=query(['rustup','which',tool])
    query([paths['rustc'],'-vV'])
    sysroot=Path(query([paths['rustc'],'--print','sysroot']))
    for line in (root/'native-build/CMakeCache.txt').read_text().splitlines():
        if line.startswith(tuple('CMAKE_'+x+':' for x in ('COMMAND','CXX_COMPILER','C_COMPILER','AR','RANLIB','LINKER','MAKE_PROGRAM'))):
            key,value=line.split('=',1);paths[key.split(':')[0]]=value
    for label,driver,part in [('cc1','CMAKE_C_COMPILER','cc1'),('cc1plus','CMAKE_CXX_COMPILER','cc1plus'),('assembler','CMAKE_C_COMPILER','as')]:
        value=query([paths[driver],'-print-prog-name='+part])
        paths[label]=shutil.which(value) or value
    for p in (sysroot/'lib').glob('*.so*'):paths['rust-library:'+p.name]=str(p)
    records={}
    for label,value in paths.items():
        p=Path(value).resolve(strict=True)
        with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(f.read()).hexdigest()
        records[label]=dict(requested=value,resolved=str(p),sha256=digest,size=p.stat().st_size)
    result=dict(schema=1,observation='post-build resolved tool identities; not an exec trace',commands=commands,tools=records,scorer_executed=False)
    with output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(tools=len(records),scorer_executed=False)))
if __name__=='__main__':capture(Path(sys.argv[1]),Path(sys.argv[2]))
