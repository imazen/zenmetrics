// Benchmark glue: use the standalone author's ingress and metric unchanged.
// The caller provides the pinned upstream source tree through its include path.
#define main original_butteraugli_file_main
#include "butteraugli/butteraugli_main.cc"
#undef main
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

static std::vector<unsigned char> read_rgb(const char* name,size_t size) {
    std::ifstream f(name,std::ios::binary);std::vector<unsigned char> bytes(size);
    if(!f || !f.read(reinterpret_cast<char*>(bytes.data()),size) || f.peek()!=EOF)
        throw std::runtime_error("RGB length/read failure");
    return bytes;
}
static std::vector<butteraugli::Image8> planes(const std::vector<unsigned char>& rgb,size_t w,size_t h) {
    auto out=butteraugli::CreatePlanes<uint8_t>(w,h,3);
    for(size_t c=0;c<3;++c)for(size_t y=0;y<h;++y)for(size_t x=0;x<w;++x)
        out[c].Row(y)[x]=rgb[(y*w+x)*3+c];
    return out;
}
static double score(const std::vector<unsigned char>& r,const std::vector<unsigned char>& d,size_t w,size_t h) {
    auto a=planes(r,w,h),b=planes(d,w,h);
    std::vector<butteraugli::ImageF> x,y;
    butteraugli::FromSrgbToLinear(a,x,0);butteraugli::FromSrgbToLinear(b,y,0);
    butteraugli::ImageF map;double result=0;
    if(!butteraugli::ButteraugliInterface(x,y,1.f,map,result))
        throw std::runtime_error("ButteraugliInterface failed");
    return result;
}
int main(int argc,char** argv) {try {
    if(argc!=7)throw std::runtime_error("width height ref.rgb dist.rgb threads samples");
    size_t w=std::stoul(argv[1]),h=std::stoul(argv[2]);unsigned threads=std::stoul(argv[5]),samples=std::stoul(argv[6]);
    if(w<8 || h<8 || threads!=1 || !samples)throw std::runtime_error("original reference requires one worker and valid geometry");
    auto r=read_rgb(argv[3],w*h*3),d=read_rgb(argv[4],w*h*3);
    std::cout<<std::setprecision(17);
    for(unsigned i=0;i<=samples;++i) {
        auto start=std::chrono::steady_clock::now();double value=score(r,d,w,h);
        auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
        std::cout<<(i?"SAMPLE ":"COLD ")<<ns<<' '<<value<<'\n'<<std::flush;
    }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}return 0;}
