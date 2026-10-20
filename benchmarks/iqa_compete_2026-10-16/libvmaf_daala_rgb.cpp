// Original libvmaf psnr_hvs feature with the CLI's explicit studio-601 YUV444
// ingress. No metric kernel is replaced. All three plane scores are computed.
#define main canonical_generic_libvmaf_main
#include "libvmaf_rgb_compare.cpp"
#undef main

static VmafPicture studio444(const std::vector<unsigned char>& rgb,unsigned w,unsigned h) {
    VmafPicture p{};
    checked(vmaf_picture_alloc(&p,VMAF_PIX_FMT_YUV444P,8,w,h),"picture444");
    for(unsigned y=0;y<h;++y) for(unsigned x=0;x<w;++x) {
        const auto pixel=rgb.data()+3*(static_cast<size_t>(y)*w+x);
        const float r=pixel[0],g=pixel[1],b=pixel[2];
        const float values[]={16.f+(65.481f*r+128.553f*g+24.966f*b)/255.f,
            128.f+(-37.797f*r-74.203f*g+112.f*b)/255.f,
            128.f+(112.f*r-93.786f*g-18.214f*b)/255.f};
        for(unsigned c=0;c<3;++c)
            static_cast<unsigned char*>(p.data[c])[y*p.stride[c]+x]=
                static_cast<unsigned char>(values[c]+.5f);
    }
    return p;
}

int main(int argc,char** argv) {try {
    if(argc!=7)throw std::runtime_error("width height ref.rgb dist.rgb threads samples");
    const unsigned w=std::stoul(argv[1]),h=std::stoul(argv[2]),threads=std::stoul(argv[5]),samples=std::stoul(argv[6]);
    if(w<8 || h<8 || !threads || !samples)throw std::runtime_error("invalid geometry/budget");
    const auto r=read_rgb(argv[3],static_cast<size_t>(w)*h*3),d=read_rgb(argv[4],static_cast<size_t>(w)*h*3);
    std::cout<<"VERSION "<<vmaf_version()<<" / original psnr_hvs / studio601-f32-444\n"<<std::setprecision(17);
    for(unsigned i=0;i<=samples;++i) {
        double score=0;
        auto start=std::chrono::steady_clock::now();
        {
            auto a=studio444(r,w,h),b=studio444(d,w,h);
            VmafContext* ctx=nullptr;
            VmafConfiguration cfg{};cfg.log_level=VMAF_LOG_LEVEL_ERROR;cfg.n_threads=threads;cfg.n_subsample=1;
            checked(vmaf_init(&ctx,cfg),"init");
            checked(vmaf_use_feature(ctx,"psnr_hvs",nullptr),"feature");
            checked(vmaf_read_pictures(ctx,&a,&b,0),"read");
            checked(vmaf_read_pictures(ctx,nullptr,nullptr,0),"flush");
            checked(vmaf_feature_score_at_index(ctx,"psnr_hvs_y",&score,0),"score_y");
            checked(vmaf_close(ctx),"close");
        }
        const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
        std::cout<<(i?"SAMPLE ":"COLD ")<<ns<<' '<<score<<'\n'<<std::flush;
    }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}return 0;}
