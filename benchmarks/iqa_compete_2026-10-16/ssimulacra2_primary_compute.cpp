// Benchmark-only RGB8 adapter for unchanged cloudinary/ssimulacra2 v2.1.
// File reads and the reusable runner are outside the clock. Every call owns
// its metadata, image bundles, RGB conversion and canonical score result.
#include "ssimulacra2.h"
#include "jxl/thread_parallel_runner_cxx.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/enc_external_image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static uint32_t positive(const char* text) {
  const std::string value(text);
  if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
    throw std::runtime_error("expected positive uint32");
  const auto number = std::stoull(value);
  if (!number || number > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("expected positive uint32");
  return static_cast<uint32_t>(number);
}

static std::vector<uint8_t> read_rgb(const char* path, size_t bytes) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("open RGB input");
  const auto length = input.tellg();
  if (length < 0 || static_cast<uintmax_t>(length) != bytes)
    throw std::runtime_error("RGB byte count");
  input.seekg(0);
  std::vector<uint8_t> data(bytes);
  if (!input.read(reinterpret_cast<char*>(data.data()),
                  static_cast<std::streamsize>(bytes)))
    throw std::runtime_error("read RGB input");
  if (input.peek() != std::ifstream::traits_type::eof())
    throw std::runtime_error("RGB byte count changed");
  return data;
}

static void init_metadata(jxl::ImageMetadata& metadata) {
  metadata.SetUintSamples(8);
  metadata.SetIntensityTarget(255.f);
  metadata.color_encoding = jxl::ColorEncoding::SRGB();
}

int main(int argc, char** argv) {
  try {
    if (argc != 9)
      throw std::runtime_error(
          "ssim2 width height ref.rgb dist.rgb threads score|score+map samples");
    if (std::string(argv[1]) != "ssim2")
      throw std::runtime_error("unknown metric");
    const uint32_t width = positive(argv[2]), height = positive(argv[3]);
    const uint32_t threads = positive(argv[6]), samples = positive(argv[8]);
    const std::string mode(argv[7]);
    if (mode != "score" && mode != "score+map")
      throw std::runtime_error("unknown mode");
    if (width < 8 || height < 8)
      throw std::runtime_error("minimum image size is 8x8 pixels");
    if (static_cast<size_t>(width) > std::numeric_limits<size_t>::max() / height / 3)
      throw std::runtime_error("RGB size overflow");
    const size_t bytes = static_cast<size_t>(width) * height * 3;
    if (bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()))
      throw std::runtime_error("RGB stream size overflow");
    if (mode == "score+map") {
      std::cout << "UNSUPPORTED_MAP no native pixel-map output in Cloudinary v2.1\n";
      return 0;
    }

    const auto reference = read_rgb(argv[4], bytes);
    const auto distorted = read_rgb(argv[5], bytes);
    auto runner = JxlThreadParallelRunnerMake(nullptr, threads == 1 ? 0 : threads);
    if (!runner) throw std::runtime_error("thread runner allocation");
    jxl::ThreadPool pool(JxlThreadParallelRunner, runner.get());
    const JxlPixelFormat format{3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
    std::cout << std::setprecision(17)
              << "VERSION cloudinary-ssimulacra2-v2.1-fb5d0dbe\n"
              << "SCOPE rgb-input\n"
              << "THREADS ingress=" << threads << " canonical_score=caller-only\n";
    // First call is retained separately. Subsequent SAMPLE records are warm
    // calls with the same per-call work; stdout transport follows the clock.
    for (uint64_t call = 0; call <= samples; ++call) {
      double score = 0;
      const auto start = std::chrono::steady_clock::now();
      {
        jxl::ImageMetadata ref_metadata, dist_metadata;
        init_metadata(ref_metadata);
        init_metadata(dist_metadata);
        jxl::ImageBundle ref_bundle(&ref_metadata), dist_bundle(&dist_metadata);
        if (!jxl::BufferToImageBundle(format, width, height, reference.data(),
                                     reference.size(), &pool,
                                     jxl::ColorEncoding::SRGB(), &ref_bundle) ||
            !jxl::BufferToImageBundle(format, width, height, distorted.data(),
                                     distorted.size(), &pool,
                                     jxl::ColorEncoding::SRGB(), &dist_bundle))
          throw std::runtime_error("BufferToImageBundle");
        const Msssim result = ComputeSSIMULACRA2(ref_bundle, dist_bundle);
        score = result.Score();
      }
      const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - start).count();
      if (!std::isfinite(score)) throw std::runtime_error("nonfinite score");
      std::cout << (call == 0 ? "COLD " : "SAMPLE ") << elapsed << ' ' << score
                << '\n' << std::flush;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR " << error.what() << '\n';
    return 2;
  }
}
