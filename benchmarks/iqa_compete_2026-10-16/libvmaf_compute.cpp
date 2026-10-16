// Benchmark-only in-process adapter to vmaf-head-sys's vendored public C API.
// Files, picture allocation/fill, context setup and model load precede timing.
#include <libvmaf/libvmaf.h>
#include <libvmaf/model.h>
#include <libvmaf/picture.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(int value,const char* operation) {
    if(value) throw std::runtime_error(std::string(operation)+" failed: "+std::to_string(value));
}
static std::vector<unsigned char> read(const char* path,size_t bytes) {
    std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("open input");
    std::vector<unsigned char> data(bytes);f.read(reinterpret_cast<char*>(data.data()),bytes);
    if(static_cast<size_t>(f.gcount())!=bytes || f.peek()!=EOF)throw std::runtime_error("YUV input length");return data;
}
static VmafPicture picture(const std::vector<unsigned char>& src,unsigned w,unsigned h,VmafPixelFormat format) {
    VmafPicture p{};check(vmaf_picture_alloc(&p,format,8,w,h),"picture_alloc");
    size_t offset=0;
    for(unsigned plane=0;plane<3;++plane) {
        const unsigned pw=p.w[plane],ph=p.h[plane];
        for(unsigned y=0;y<ph;++y)std::memcpy(static_cast<unsigned char*>(p.data[plane])+y*p.stride[plane],src.data()+offset+y*pw,pw);
        offset+=static_cast<size_t>(pw)*ph;
    }return p;
}
int main(int argc,char**argv) {try {
    if(argc!=8 && argc!=9)throw std::runtime_error("metric width height ref.yuv dist.yuv threads samples");
    const std::string metric=argv[1];const unsigned w=std::stoul(argv[2]),h=std::stoul(argv[3]),threads=std::stoul(argv[6]),samples=std::stoul(argv[7]);
    const bool mono=argc==9 && std::string(argv[8])=="400";
    const bool full=argc==9 && std::string(argv[8])=="444";
    const auto format=mono?VMAF_PIX_FMT_YUV400P:(full?VMAF_PIX_FMT_YUV444P:VMAF_PIX_FMT_YUV420P);
    if((!mono && !full && (w%2 || h%2)) || !w || !h || !threads || !samples)throw std::runtime_error("positive even geometry/threads/samples required");
    const size_t bytes=static_cast<size_t>(w)*h*(mono?1:3)/((mono||full)?1:2);
    const auto ref=read(argv[4],bytes),dist=read(argv[5],bytes);
    VmafModel* model=nullptr;const bool is_model=metric=="vmaf" || metric=="vmaf-neg";
    if(is_model) {VmafModelConfig cfg{"iqa-compete",VMAF_MODEL_FLAGS_DEFAULT};check(vmaf_model_load(&model,&cfg,metric=="vmaf"?"vmaf_v0.6.1":"vmaf_v0.6.1neg"),"model_load");}
    std::string feature=metric,output=metric;
    if(metric=="psnr") {feature="float_psnr";output="float_psnr";}
    if(metric=="psnr_hvs")output="psnr_hvs_y";
    if(metric=="vif") {feature="float_vif";output="vif";}
    if(metric=="cambi")output="Cambi_feature_cambi_score";
    std::cout<<"VERSION "<<vmaf_version()<<'\n'<<std::setprecision(17);
    for(unsigned iteration=0;iteration<=samples;++iteration) {
        VmafContext* ctx=nullptr;VmafConfiguration cfg{};cfg.log_level=VMAF_LOG_LEVEL_ERROR;cfg.n_threads=threads;cfg.n_subsample=1;
        check(vmaf_init(&ctx,cfg),"vmaf_init");
        if(is_model)check(vmaf_use_features_from_model(ctx,model),"use_model");
        else {VmafFeatureDictionary* options=nullptr;if(metric=="vif")check(vmaf_feature_dictionary_set(&options,"debug","true"),"vif debug options");check(vmaf_use_feature(ctx,feature.c_str(),options),"use_feature");}
        auto r=picture(ref,w,h,format),d=picture(dist,w,h,format);
        const auto start=std::chrono::steady_clock::now();
        check(vmaf_read_pictures(ctx,&r,&d,0),"read_pictures");
        check(vmaf_read_pictures(ctx,nullptr,nullptr,0),"flush");
        double score=0;
        if(is_model)check(vmaf_score_at_index(ctx,model,&score,0),"score");
        else check(vmaf_feature_score_at_index(ctx,output.c_str(),&score,0),"feature_score");
        const auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
        check(vmaf_close(ctx),"close");
        if(iteration)std::cout<<"SAMPLE "<<elapsed<<' '<<score<<'\n'<<std::flush;
    }
    if(model)vmaf_model_destroy(model);return 0;
}catch(const std::exception&e){std::cerr<<"ERROR "<<e.what()<<'\n';return 1;}}
