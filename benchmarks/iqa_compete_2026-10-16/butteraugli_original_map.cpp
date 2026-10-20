// Accuracy-only output from the unchanged original Google Butteraugli interface.
#define main original_butteraugli_file_main
#include "butteraugli/butteraugli_main.cc"
#undef main
#include <bit>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

static std::vector<unsigned char> read_rgb(const char* name, size_t n) {
    std::ifstream input(name, std::ios::binary);
    std::vector<unsigned char> bytes(n);
    if (!input || !input.read(reinterpret_cast<char*>(bytes.data()), n) || input.peek() != EOF)
        throw std::runtime_error("RGB length/read failure");
    return bytes;
}

static std::vector<butteraugli::ImageF> linear(const std::vector<unsigned char>& rgb,
                                             size_t w, size_t h) {
    auto planes = butteraugli::CreatePlanes<uint8_t>(w, h, 3);
    for (size_t c = 0; c < 3; ++c)
        for (size_t y = 0; y < h; ++y)
            for (size_t x = 0; x < w; ++x)
                planes[c].Row(y)[x] = rgb[(y * w + x) * 3 + c];
    std::vector<butteraugli::ImageF> result;
    butteraugli::FromSrgbToLinear(planes, result, 0);
    return result;
}

int main(int argc, char** argv) {
    try {
        if (argc != 7) throw std::runtime_error("width height ref.rgb dist.rgb worker-budget output.f32");
        static_assert(sizeof(float) == 4 && std::endian::native == std::endian::little);
        size_t w = std::stoul(argv[1]), h = std::stoul(argv[2]);
        unsigned budget = std::stoul(argv[5]);
        if (w < 8 || h < 8 || !budget) throw std::runtime_error("invalid geometry/budget");
        auto r = linear(read_rgb(argv[3], w * h * 3), w, h);
        auto d = linear(read_rgb(argv[4], w * h * 3), w, h);
        butteraugli::ImageF map;
        double score = 0;
        if (!butteraugli::ButteraugliInterface(r, d, 1.f, map, score))
            throw std::runtime_error("ButteraugliInterface failed");
        if (map.xsize() != w || map.ysize() != h) throw std::runtime_error("map geometry");
        std::ofstream output(argv[6], std::ios::binary | std::ios::out);
        if (!output) throw std::runtime_error("map output open");
        for (size_t y = 0; y < h; ++y)
            output.write(reinterpret_cast<const char*>(map.Row(y)), w * sizeof(float));
        output.close();
        if (!output) throw std::runtime_error("map output write");
        std::cout << "BUTTER_ORIGINAL_MAP " << std::setprecision(17) << score
                  << ' ' << w << ' ' << h << ' ' << budget << " 1\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
