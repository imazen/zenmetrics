#!/usr/bin/env python3
"""Build a scalar resident-RGB callable from the unchanged pinned libgmsd ZIP."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

from build_gmsd_primary import REV, ZIP_SHA
from compare_canonical import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    if sha(a.archive)!=ZIP_SHA:raise ValueError('original libgmsd archive differs')
    a.output.mkdir();out=a.output.resolve();source=out/('libgmsd-'+REV)
    with zipfile.ZipFile(a.archive) as archive:
        for item in archive.infolist():
            path=Path(item.filename)
            if path.is_absolute() or '..' in path.parts or path.parts[0]!='libgmsd-'+REV:raise ValueError('archive path')
            if not item.is_dir():
                destination=out/path;destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes(archive.read(item))
    base=Path(__file__).resolve().parent
    prefix=(base/'gmsd_primary_compute.c').read_text().split('int main(',1)[0]
    body=r'''
int main(int argc,char **argv) {
  if(argc!=7)return 2;
  const size_t w=positive(argv[1]),h=positive(argv[2]),threads=positive(argv[5]),samples=positive(argv[6]);
  if(w<2 || h<2 || w%2 || h%2 || w>SIZE_MAX/h/3)return 2;
  (void)threads; /* original library is single-threaded; caller sets the budget */
  bmpread_t r={(int)w,(int)h,BMPREAD_TOP_DOWN|BMPREAD_BYTE_ALIGN,read_rgb(argv[3],w*h*3)};
  bmpread_t d={(int)w,(int)h,BMPREAD_TOP_DOWN|BMPREAD_BYTE_ALIGN,read_rgb(argv[4],w*h*3)};
  for(size_t call=0;call<=samples;++call) {
    struct timespec start,stop;double value=0;
    clock_gettime(CLOCK_MONOTONIC,&start);
    gmsd_image_data_t *rg=rgb2gray(&r),*dg=rgb2gray(&d);
    if(!rg || !dg || !gmsd(rg,dg,w,h,&value,NULL))return 4;
    free(rg);free(dg);
    clock_gettime(CLOCK_MONOTONIC,&stop);
    printf("%s %.0f %.17g\n",call?"SAMPLE":"COLD",ns(start,stop),value);fflush(stdout);
  }
  free(r.data);free(d.data);return 0;
}
'''
    cpp=out/'reference.c';cpp.write_text(prefix+body)
    compiler=Path(shutil.which('cc')).resolve(strict=True);binary=out/'gmsd-rgb'
    command=[str(compiler),'-std=gnu11','-DNDEBUG','-O3','-ffp-contract=off',
        *['-I'+str(source/part) for part in ['include','3rdparty','bin']],str(cpp),
        *map(str,sorted((source/'src').glob('*.c'))),str(source/'3rdparty/bmpread.c'),'-lm','-o',str(binary)]
    subprocess.run(command,check=True)
    (out/'build.json').write_text(json.dumps(dict(source_revision=REV,archive_sha256=ZIP_SHA,command=command,
        compiler_sha256=sha(compiler),binary_sha256=sha(binary),wrapper_sha256=sha(cpp),
        scope='Unchanged original RGB conversion and scalar gmsd; all per-call allocations and release timed; even geometry required'),indent=2)+'\n')
    print(str(binary))


if __name__=='__main__':main()
