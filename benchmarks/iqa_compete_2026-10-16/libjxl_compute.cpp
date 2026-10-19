// Benchmark-only adapter. The optional trailing scope argument selects which
// stage the per-call clock covers (default preserves the original staging):
//   native-buffer (argc 9): RGB unpack, metadata, pool, bundles, comparator
//     are staged untimed; the clock covers the metric plus output release.
//   rgb-input (argc 10): raw RGB vectors and the worker pool stay untimed;
//     each iteration's clock covers metadata, bundles, both
//     BufferToImageBundle conversions, the per-call comparator (butteraugli
//     only), the metric, and release of the map and every per-call object.
// See native-rgb-scope.md. In native-buffer the VERSION header is emitted at
// the original point (after staging, before the sample loop); rgb-input emits
// it plus an explicit "SCOPE rgb-input" line before its loop. SAMPLE lines
// carry measured times; no cross-scope output or score equivalence is claimed.
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
    if(argc!=9 && argc!=10)throw std::runtime_error("metric width height ref.rgb dist.rgb threads mode samples [native-buffer|rgb-input]");
    const std::string metric=argv[1],mode=argv[7],scope=argc==10?argv[9]:"native-buffer";
    const unsigned w=std::stoul(argv[2]),h=std::stoul(argv[3]),threads=std::stoul(argv[6]),samples=std::stoul(argv[8]);
    if(!w || !h || !threads || !samples)throw std::runtime_error("positive dimensions/threads/samples");
    if(metric!="butteraugli" && metric!="ssim2")throw std::runtime_error("unknown metric");
    if(scope!="native-buffer" && scope!="rgb-input")throw std::runtime_error("unknown scope");
    if(metric=="ssim2" && mode=="score+map"){std::cout<<"UNSUPPORTED_MAP no native pixel-map output in the v0.12 SSIMULACRA2 API\n";return 0;}
    const auto rb=read(argv[4],static_cast<size_t>(w)*h*3),db=read(argv[5],static_cast<size_t>(w)*h*3);
    jpegxl::tools::ThreadPoolInternal pool(threads==1?0:threads);
    const float intensity=metric=="ssim2"?255.f:80.f;
    jxl::ButteraugliParams params;params.hf_asymmetry=1.f;params.xmul=1.f;params.intensity_target=80.f;
    const auto& cms=*JxlGetDefaultCms();
    auto* manager=jpegxl::tools::NoMemoryManager();
    const JxlPixelFormat format{3,JXL_TYPE_UINT8,JXL_NATIVE_ENDIAN,0};
    auto init_meta=[&](jxl::ImageMetadata&m){m.SetUintSamples(8);m.SetIntensityTarget(intensity);m.color_encoding=jxl::ColorEncoding::SRGB();};
    auto convert=[&](const std::vector<unsigned char>&buf,jxl::ImageBundle&bundle){
        return jxl::BufferToImageBundle(format,w,h,buf.data(),buf.size(),pool.get(),jxl::ColorEncoding::SRGB(),&bundle);
    };
    auto run_metric=[&](const jxl::ImageBundle&r,const jxl::ImageBundle&d,jxl::Comparator*comparator,jxl::ImageF&map,double&score){
        float butter_score=0;
        if(metric=="ssim2"){if(!ssim_score(r,d,&score))throw std::runtime_error("SSIMULACRA2");}
        else{if(!jxl::ComputeScore(r,d,comparator,cms,&butter_score,mode=="score+map"?&map:nullptr,pool.get(),false))throw std::runtime_error("Butteraugli");score=butter_score;}
    };
    auto emit_header=[&]{
        std::cout<<std::setprecision(17)<<"VERSION libjxl-v0.12.0\n";
        if(scope=="rgb-input")std::cout<<"SCOPE rgb-input\n";
    };
    if(scope=="native-buffer") {
        jxl::ImageMetadata rm,dm;init_meta(rm);init_meta(dm);
        jxl::ImageBundle r(manager,&rm),d(manager,&dm);
        if(!convert(rb,r) || !convert(db,d))throw std::runtime_error("BufferToImageBundle");
        jxl::JxlButteraugliComparator comparator(params,cms);
        emit_header(); // Original position: after staged setup, before the loop.
        for(unsigned i=0;i<=samples;++i) {
            double score=0;jxl::ImageF map;
            const auto start=std::chrono::steady_clock::now();
            run_metric(r,d,&comparator,map,score);
            map=jxl::ImageF(); // Include native output release, matching zenbench iter().
            const auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
            if(i)std::cout<<"SAMPLE "<<elapsed<<' '<<score<<'\n'<<std::flush;
        }
    } else {
        emit_header(); // New scope: VERSION + SCOPE before the loop.
        for(unsigned i=0;i<=samples;++i) {
            double score=0;
            const auto start=std::chrono::steady_clock::now();
            {   // Timed RAII scope: every per-call object releases before the clock stops.
                jxl::ImageMetadata rm,dm;init_meta(rm);init_meta(dm);
                jxl::ImageBundle r(manager,&rm),d(manager,&dm);jxl::ImageF map;
                if(!convert(rb,r) || !convert(db,d))throw std::runtime_error("BufferToImageBundle");
                if(metric=="butteraugli") {
                    jxl::JxlButteraugliComparator comparator(params,cms);
                    run_metric(r,d,&comparator,map,score);
                } else run_metric(r,d,nullptr,map,score);
                map=jxl::ImageF(); // Include native output release, matching zenbench iter().
            }
            const auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
            if(i)std::cout<<"SAMPLE "<<elapsed<<' '<<score<<'\n'<<std::flush;
        }
    }
    return 0;
}catch(const std::exception&e){std::cerr<<"ERROR "<<e.what()<<'\n';return 1;}}
