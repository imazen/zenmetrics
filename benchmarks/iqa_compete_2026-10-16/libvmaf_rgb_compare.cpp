// Reference-only libvmaf adapter. Resident RGB8 -> named planes -> complete
// one-frame scoring call, including context/picture lifetime, inside timing.
// Immutable model loading and input file reads stay outside the warm samples.
#include <libvmaf/libvmaf.h>
#include <libvmaf/model.h>
#include <libvmaf/picture.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

static void checked(int rc, const char* operation) {
    if (rc) throw std::runtime_error(std::string(operation)+": "+std::to_string(rc));
}
static std::vector<unsigned char> read_rgb(const char* file, size_t bytes) {
    std::ifstream f(file, std::ios::binary);
    std::vector<unsigned char> v(bytes);
    if (!f || !f.read(reinterpret_cast<char*>(v.data()), bytes) || f.peek()!=EOF)
        throw std::runtime_error("RGB length/read failure");
    return v;
}
static unsigned char quant(double value, double lo=0, double hi=255) {
    return static_cast<unsigned char>(std::clamp(std::round(value),lo,hi));
}
static std::vector<unsigned char> planes(const std::vector<unsigned char>& rgb,
        unsigned stride, unsigned w, unsigned h, bool studio, bool mono,
        bool float_studio, bool integer_studio) {
    std::vector<unsigned char> v(static_cast<size_t>(w)*h*(mono?1:3)/(mono?1:2));
    const size_t pixels=static_cast<size_t>(w)*h;
    for (unsigned y=0;y<h;++y) for(unsigned x=0;x<w;++x) {
        const auto p=rgb.data()+3*(static_cast<size_t>(y)*stride+x);
        if(integer_studio) {
            const unsigned t=(8414u*p[0]+16519u*p[1]+3208u*p[2]+(32u<<14)+(1u<<8))>>9;
            v[static_cast<size_t>(y)*w+x]=static_cast<unsigned char>((t+32)>>6);
        } else if(float_studio) {
            const float value=16.f+(65.481f*p[0]+128.553f*p[1]+24.966f*p[2])/255.f;
            v[static_cast<size_t>(y)*w+x]=quant(value);
        } else if(studio) {
            v[static_cast<size_t>(y)*w+x]=quant(16.+(65.481*p[0]+128.553*p[1]+24.966*p[2])/255.);
        } else {
            const double r=p[0]/255.,g=p[1]/255.,b=p[2]/255.;
            v[static_cast<size_t>(y)*w+x]=quant(16.+219.*(.2126*r+.7152*g+.0722*b),16,235);
        }
    }
    if (!mono) for(unsigned y=0;y<h/2;++y) for(unsigned x=0;x<w/2;++x) {
        double u=0,vv=0;
        for(unsigned j=0;j<2;++j) for(unsigned i=0;i<2;++i) {
            const auto p=rgb.data()+3*(static_cast<size_t>(2*y+j)*stride+2*x+i);
            if(studio) {
                u+=(-37.797*p[0]-74.203*p[1]+112.*p[2])/255.;
                vv+=(112.*p[0]-93.786*p[1]-18.214*p[2])/255.;
            } else {
                const double r=p[0]/255.,g=p[1]/255.,b=p[2]/255.;
                u+=-.114572*r-.385428*g+.5*b;
                vv+=.5*r-.454153*g-.045847*b;
            }
        }
        const double scale=studio?1.:224.;
        v[pixels+static_cast<size_t>(y)*(w/2)+x]=quant(128.+scale*u/4.,16,240);
        v[pixels+pixels/4+static_cast<size_t>(y)*(w/2)+x]=quant(128.+scale*vv/4.,16,240);
    }
    return v;
}
static VmafPicture picture(const std::vector<unsigned char>& data,unsigned w,unsigned h,bool mono) {
    VmafPicture p{};
    checked(vmaf_picture_alloc(&p,mono?VMAF_PIX_FMT_YUV400P:VMAF_PIX_FMT_YUV420P,8,w,h),"picture");
    size_t offset=0;
    for(unsigned plane=0;plane<(mono?1u:3u);++plane) {
        for(unsigned y=0;y<p.h[plane];++y)
            std::memcpy(static_cast<unsigned char*>(p.data[plane])+y*p.stride[plane],
                        data.data()+offset+static_cast<size_t>(y)*p.w[plane],p.w[plane]);
        offset+=static_cast<size_t>(p.w[plane])*p.h[plane];
    }
    return p;
}
int main(int argc,char** argv) { try {
    if(argc!=9) throw std::runtime_error("metric width height ref.rgb dist.rgb threads samples convention");
    const std::string metric=argv[1],convention=argv[8];
    const std::set<std::string> accepted={"vmaf","vmaf-neg","vmaf-4k","vmaf-v1","vmaf-studio601","vmaf-neg-studio601",
        "cambi","cambi-aic2026","ssim-libvmaf","msssim-libvmaf","psnr-y-libvmaf"};
    if(!accepted.count(metric) || !std::set<std::string>{"709","601","601-f32","601-integer"}.count(convention))
        throw std::runtime_error("unknown metric or RGB convention");
    const unsigned iw=std::stoul(argv[2]),ih=std::stoul(argv[3]),threads=std::stoul(argv[6]),samples=std::stoul(argv[7]);
    const bool model_metric=metric.rfind("vmaf",0)==0;
    const bool nr=metric=="cambi" || metric=="cambi-aic2026";
    const bool mono=!model_metric;
    const unsigned w=(model_metric || nr)?iw&~1u:iw,h=(model_metric || nr)?ih&~1u:ih;
    if(!w || !h || !threads || !samples) throw std::runtime_error("positive geometry/budget required");
    const auto r=read_rgb(argv[4],static_cast<size_t>(iw)*ih*3),d=read_rgb(argv[5],static_cast<size_t>(iw)*ih*3);
    VmafModel* model=nullptr;
    if(model_metric) {
        const char* name=metric=="vmaf-4k"?"vmaf_4k_v0.6.1":metric=="vmaf-v1"?"vmaf_v1.0.16_3d0h":
                         metric.find("neg")!=std::string::npos?"vmaf_v0.6.1neg":"vmaf_v0.6.1";
        VmafModelConfig cfg{"canonical-rgb",VMAF_MODEL_FLAGS_DEFAULT};
        checked(vmaf_model_load(&model,&cfg,name),"model_load");
    }
    const std::string feature=nr?"cambi":metric=="ssim-libvmaf"?"float_ssim":metric=="msssim-libvmaf"?"float_ms_ssim":"float_psnr";
    const std::string output=nr?"Cambi_feature_cambi_score":feature;
    std::cout<<"VERSION "<<vmaf_version()<<"\nSCOPE resident-rgb-full-frame\n"<<std::setprecision(17);
    for(unsigned i=0;i<=samples;++i) {
        double score=0;
        const auto start=std::chrono::steady_clock::now();
        {
            const bool studio=convention!="709";
            const auto dyuv=planes(d,iw,w,h,studio,mono,convention=="601-f32",convention=="601-integer");
            const auto ryuv=nr?std::vector<unsigned char>{}:planes(r,iw,w,h,studio,mono,convention=="601-f32",false);
            VmafContext* ctx=nullptr;
            VmafConfiguration cfg{};cfg.log_level=VMAF_LOG_LEVEL_ERROR;cfg.n_threads=threads;cfg.n_subsample=1;
            checked(vmaf_init(&ctx,cfg),"init");
            if(model_metric) checked(vmaf_use_features_from_model(ctx,model),"features_from_model");
            else checked(vmaf_use_feature(ctx,feature.c_str(),nullptr),"feature");
            auto a=picture(nr?dyuv:ryuv,w,h,mono),b=picture(dyuv,w,h,mono);
            checked(vmaf_read_pictures(ctx,&a,&b,0),"read");
            checked(vmaf_read_pictures(ctx,nullptr,nullptr,0),"flush");
            if(model_metric) checked(vmaf_score_at_index(ctx,model,&score,0),"score");
            else checked(vmaf_feature_score_at_index(ctx,output.c_str(),&score,0),"feature_score");
            checked(vmaf_close(ctx),"close");
        }
        const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
        std::cout<<(i?"SAMPLE ":"COLD ")<<ns<<' '<<score<<'\n'<<std::flush;
    }
    if(model) vmaf_model_destroy(model);
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;} }
