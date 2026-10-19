#include "FLIP.h"
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
int main() {
    const int w = 64, h = 64;
    FLIP::image<FLIP::color3> r(w,h), d(w,h);
    for (int y=0;y<h;y++) for (int x=0;x<w;x++) {
        float a[3],b[3];
        for(int c=0;c<3;c++) {
            const int i=(y*w+x)*3+c;
            const unsigned char v=(i*37+i/11)%256;
            a[c]=float(v)/255.0f;
            b[c]=float(i%17==0 ? v^63 : v)/255.0f;
        }
        r.set(x,y,FLIP::color3(a[0],a[1],a[2]));
        d.set(x,y,FLIP::color3(b[0],b[1],b[2]));
    }
    r.sRGBToLinearRGB(); d.sRGBToLinearRGB();
    FLIP::Parameters params;
    FLIP::image<float> map(w,h);
    FLIP::evaluate(r,d,false,params,map);
    double sum64=0; float sum32=0;
    for(int y=0;y<h;y++) for(int x=0;x<w;x++) {
        const float v=map.get(x,y);sum64+=v;sum32+=v;
    }
    std::cout<<std::setprecision(17)<<"AUTHOR_FLIP "<<sum64/(w*h)<<"\nAUTHOR_FLIP_V17 "<<sum32/float(w*h)<<'\n';
    for(int i=0;i<256;i++) {
        float v=FLIP::color3::sRGBToLinearRGB(float(i)/255.0f);
        std::uint32_t bits;std::memcpy(&bits,&v,4);
        std::cout<<"LUT "<<std::dec<<i<<' '<<std::hex<<bits<<'\n';
    }
}
