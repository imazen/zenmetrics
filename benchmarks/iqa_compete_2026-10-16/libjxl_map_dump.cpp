// Accuracy-only glue using the same source, ingress and comparator as the scalar adapter.
#define main frozen_scalar_benchmark_main
#include "libjxl_compute.cpp"
#undef main
#include <filesystem>

int main(int argc, char** argv) {
  try {
    if (argc != 7) throw std::runtime_error("width height ref.rgb dist.rgb threads output.f32");
    const size_t w = std::stoul(argv[1]), h = std::stoul(argv[2]);
    const unsigned threads = std::stoul(argv[5]);
    if (!w || !h || !threads || std::filesystem::exists(argv[6]))
      throw std::runtime_error("invalid geometry, threads or existing output");
    const auto rb = read(argv[3], w*h*3), db = read(argv[4], w*h*3);
    jpegxl::tools::ThreadPoolInternal pool(threads == 1 ? 0 : threads);
    auto* manager = jpegxl::tools::NoMemoryManager();
    const auto& cms = *JxlGetDefaultCms();
    jxl::ImageMetadata rm, dm;
    for (auto* meta : {&rm, &dm}) {
      meta->SetUintSamples(8);
      meta->SetIntensityTarget(80.f);
      meta->color_encoding = jxl::ColorEncoding::SRGB();
    }
    jxl::ImageBundle r(manager, &rm), d(manager, &dm);
    const JxlPixelFormat format{3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
    if (!jxl::BufferToImageBundle(format, w, h, rb.data(), rb.size(), pool.get(),
                                jxl::ColorEncoding::SRGB(), &r) ||
        !jxl::BufferToImageBundle(format, w, h, db.data(), db.size(), pool.get(),
                                jxl::ColorEncoding::SRGB(), &d))
      throw std::runtime_error("BufferToImageBundle");
    jxl::ButteraugliParams params;
    params.hf_asymmetry = 1.f;
    params.xmul = 1.f;
    params.intensity_target = 80.f;
    jxl::JxlButteraugliComparator comparator(params, cms);
    jxl::ImageF map;
    float score = 0;
    if (!jxl::ComputeScore(r, d, &comparator, cms, &score, &map, pool.get(), false))
      throw std::runtime_error("ComputeScore");
    if (map.xsize() != w || map.ysize() != h)
      throw std::runtime_error("map geometry");
    std::ofstream output(argv[6], std::ios::binary);
    for (size_t y = 0; y < h; ++y)
      output.write(reinterpret_cast<const char*>(map.Row(y)), w*sizeof(float));
    output.close();
    if (!output) throw std::runtime_error("map write");
    std::cout << std::setprecision(17) << "BUTTER_MAP " << score << ' ' << w << ' ' << h << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
