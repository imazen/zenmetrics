// Benchmark-only caller for the unchanged original thorfdbg/mDCTpsnr classes.
// Resident P6 backing bytes avoid process launch and file I/O in every call.
#include "img/image.hpp"
#include "io/bytestream.hpp"
#include "measure/pooling.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

class ResidentP6 final : public ByteStream {
public:
  explicit ResidentP6(std::vector<uint8_t> &bytes) : ByteStream(0) {
    if (bytes.size() > std::numeric_limits<uint32_t>::max())
      throw std::runtime_error("resident stream too large");
    m_pucBuffer = m_pucBufPtr = bytes.data();
    m_ulBufSize = m_ulBufBytes = uint32_t(bytes.size());
  }
  void Flush() override { throw std::runtime_error("read-only stream"); }
protected:
  int32_t Fill() override { return 0; }
};

static std::vector<uint8_t> input(const char *path, unsigned w, unsigned h) {
  std::string header = "P6\n"+std::to_string(w)+" "+std::to_string(h)+"\n255\n";
  std::vector<uint8_t> bytes(header.begin(), header.end());
  size_t count = size_t(w)*h*3;
  bytes.resize(header.size()+count);
  std::ifstream file(path, std::ios::binary);
  file.read(reinterpret_cast<char *>(bytes.data()+header.size()), count);
  if (size_t(file.gcount()) != count || file.peek() != EOF)
    throw std::runtime_error("RGB byte count differs");
  return bytes;
}

int main(int argc, char **argv) {
  if (argc != 7) return 2;
  unsigned w=std::stoul(argv[1]), h=std::stoul(argv[2]);
  int threads=std::stoi(argv[5]), samples=std::stoi(argv[6]);
  if (w<64 || h<64 || w>65536 || h>65536 || threads<1 || samples<1) return 2;
  auto r=input(argv[3],w,h), d=input(argv[4],w,h);
  for (int i=0;i<=samples;++i) {
    auto start=std::chrono::steady_clock::now();
    float score;
    {
      ResidentP6 rs(r), ds(d);
      Image ri, di;
      Pooling pool;
      ri.OpenPNM(&rs,threads);di.OpenPNM(&ds,threads);
      score=pool.Measure(&ri,&di,nullptr,nullptr,threads,0);
    }
    auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now()-start).count();
    std::printf("%s %lld %.17g\n",i?"SAMPLE":"COLD",(long long)elapsed,double(score));
    std::fflush(stdout);
  }
}
