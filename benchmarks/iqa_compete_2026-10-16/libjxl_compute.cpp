// Benchmark-only adapter. Raw RGB unpacking, metadata and worker setup are untimed.
#include <jxl/cms.h>
#include <jxl/cms_interface.h>
#include <jxl/types.h>
#include "lib/jxl/enc_butteraugli_comparator.h"
#include "lib/jxl/enc_comparator.h"
#include "lib/jxl/enc_external_image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"
#include "tools/ssimulacra2.h"
#include "tools/no_memory_manager.h"
#include "tools/thread_pool_internal.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
static std::vector<unsigned char> read(const char* path,size_t bytes) {
    std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("open RGB input");
    std::vector<unsigned char> v(bytes);f.read(reinterpret_cast<char*>(v.data()),bytes);
    if(static_cast<size_t>(f.gcount())!=bytes || f.peek()!=EOF)throw std::runtime_error("RGB byte count");return v;
}
static jxl::Status ssim_score(const jxl::ImageBundle& r,const jxl::ImageBundle& d,double* score) {
    JXL_ASSIGN_OR_RETURN(auto result,ComputeSSIMULACRA2(r,d));*score=result.Score();return true;
}
int main(int argc,char**argv) {try {
    if(argc!=9)throw std::runtime_error("metric width height ref.rgb dist.rgb threads mode samples");
    const std::string metric=argv[1],mode=argv[7];const unsigned w=std::stoul(argv[2]),h=std::stoul(argv[3]),threads=std::stoul(argv[6]),samples=std::stoul(argv[8]);
    if(!w || !h || !threads || !samples)throw std::runtime_error("positive dimensions/threads/samples");
    if(metric!="butteraugli" && metric!="ssim2")throw std::runtime_error("unknown metric");
    if(metric=="ssim2" && mode=="score+map"){std::cout<<"UNSUPPORTED_MAP no native pixel-map output in the v0.12 SSIMULACRA2 API\n";return 0;}
    const auto rb=read(argv[4],static_cast<size_t>(w)*h*3),db=read(argv[5],static_cast<size_t>(w)*h*3);
    jpegxl::tools::ThreadPoolInternal pool(threads==1?0:threads);
    jxl::ImageMetadata rm,dm;rm.SetUintSamples(8);dm.SetUintSamples(8);const float intensity=metric=="ssim2"?255.f:80.f;rm.SetIntensityTarget(intensity);dm.SetIntensityTarget(intensity);
    rm.color_encoding=jxl::ColorEncoding::SRGB();dm.color_encoding=jxl::ColorEncoding::SRGB();
    auto* manager=jpegxl::tools::NoMemoryManager();
    jxl::ImageBundle r(manager,&rm),d(manager,&dm);
    JxlPixelFormat format{3,JXL_TYPE_UINT8,JXL_NATIVE_ENDIAN,0};
    if(!jxl::BufferToImageBundle(format,w,h,rb.data(),rb.size(),pool.get(),jxl::ColorEncoding::SRGB(),&r) || !jxl::BufferToImageBundle(format,w,h,db.data(),db.size(),pool.get(),jxl::ColorEncoding::SRGB(),&d))throw std::runtime_error("BufferToImageBundle");
    jxl::ButteraugliParams params;params.hf_asymmetry=1.f;params.xmul=1.f;params.intensity_target=80.f;
    const auto& cms=*JxlGetDefaultCms();jxl::JxlButteraugliComparator comparator(params,cms);
    std::cout<<std::setprecision(17)<<"VERSION libjxl-v0.12.0\n";
    for(unsigned i=0;i<=samples;++i) {
        double score=0;float butter_score=0;jxl::ImageF map;
        const auto start=std::chrono::steady_clock::now();
        if(metric=="ssim2") {if(!ssim_score(r,d,&score))throw std::runtime_error("SSIMULACRA2");}
        else {if(!jxl::ComputeScore(r,d,&comparator,cms,&butter_score,mode=="score+map"?&map:nullptr,pool.get(),false))throw std::runtime_error("Butteraugli");score=butter_score;}
        map=jxl::ImageF(); // Include native output release, matching zenbench iter().
        const auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
        if(i)std::cout<<"SAMPLE "<<elapsed<<' '<<score<<'\n'<<std::flush;
    }return 0;
}catch(const std::exception&e){std::cerr<<"ERROR "<<e.what()<<'\n';return 1;}}
