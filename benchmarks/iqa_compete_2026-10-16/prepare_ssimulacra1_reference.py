#!/usr/bin/env python3
"""Wrap an explicit original SSIMULACRA1 source in a resident-RGB callable.

The metric body remains a byte-identical substring. Only file ingress and
stdout in the original main are replaced; original notices remain intact.
Run this build under the host admission/resource wrapper.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();source=a.source.resolve(strict=True).read_text()
    prefix,main=source.split('int main(int argc, char** argv) {',1)
    start=main.index('    if (nChan == 4) {')
    end=main.index('    printf("%.8f\\n", score);')
    body=main[start:end]
    assert 'argv[' not in body and 'imread(' not in body
    a.output.mkdir();cpp=a.output/'reference.cpp';binary=a.output/'ssimulacra1-rgb'
    prelude='''
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
static double reference_score(const std::vector<unsigned char>& r,
                             const std::vector<unsigned char>& d,int w,int h) {
    cv::Mat img1_temp,img2_temp,img1,img2;
    cv::Mat rv(h,w,CV_8UC3,const_cast<unsigned char*>(r.data()));
    cv::Mat dv(h,w,CV_8UC3,const_cast<unsigned char*>(d.data()));
    cv::cvtColor(rv,img1_temp,cv::COLOR_RGB2BGR);
    cv::cvtColor(dv,img2_temp,cv::COLOR_RGB2BGR);
    const int nChan=3,pixels=w*h;
    Scalar sC1={C1,C1,C1,C1};
'''
    postlude='''
    return score;
}
static std::vector<unsigned char> read_rgb(const char* path,size_t size) {
    std::ifstream f(path,std::ios::binary);std::vector<unsigned char> bytes(size);
    if(!f || !f.read(reinterpret_cast<char*>(bytes.data()),size) || f.peek()!=EOF)
        throw std::runtime_error("RGB file length/read failure");
    return bytes;
}
int main(int argc,char** argv) {try {
    if(argc!=7)throw std::runtime_error("width height ref.rgb dist.rgb threads samples");
    int w=std::stoi(argv[1]),h=std::stoi(argv[2]),threads=std::stoi(argv[5]),samples=std::stoi(argv[6]);
    if(w<8 || h<8 || threads<1 || samples<1)throw std::runtime_error("invalid geometry/budget");
    auto r=read_rgb(argv[3],static_cast<size_t>(w)*h*3),d=read_rgb(argv[4],static_cast<size_t>(w)*h*3);
    cv::setNumThreads(threads);std::cout<<std::setprecision(17);
    for(int i=0;i<=samples;++i) {
        auto start=std::chrono::steady_clock::now();
        double score=reference_score(r,d,w,h);
        auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
        std::cout<<(i?"SAMPLE ":"COLD ")<<ns<<' '<<score<<'\\n'<<std::flush;
    }
}catch(const std::exception& e){std::cerr<<e.what()<<'\\n';return 1;}return 0;}
'''
    text=prefix+prelude+body+postlude
    assert body in text
    cpp.write_text(text)
    flags=shlex.split(subprocess.check_output(['pkg-config','--cflags','--libs','opencv4'],text=True))
    command=['c++','-O3','-std=c++17','-ffp-contract=off',str(cpp),*flags,'-o',str(binary)]
    subprocess.run(command,check=True)
    receipt=dict(source=str(a.source.resolve()),source_sha256=sha(a.source),
                 metric_body_sha256=hashlib.sha256(body.encode()).hexdigest(),
                 generated_source_sha256=sha(cpp),binary_sha256=sha(binary),command=command,
                 scope='Resident RGB8, RGB-to-BGR, original body and temporary destruction inside each callable')
    (a.output/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
