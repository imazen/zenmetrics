/* Benchmark-only raw RGB adapter. The original CLI source is compiled with
 * its main renamed, so rgb2gray is the unchanged upstream function. */
#define _POSIX_C_SOURCE 200809L
#define main libgmsd_file_main
#include "gmsd.c"
#undef main
#include <errno.h>
#include <limits.h>
#include <time.h>

static size_t positive(const char *text) {
  char *end = NULL;
  errno = 0;
  unsigned long value = strtoul(text, &end, 10);
  if (errno || !*text || *end || text[0] == '-' || !value || value > INT_MAX)
    exit(2);
  return (size_t)value;
}

static unsigned char *read_rgb(const char *path, size_t bytes) {
  FILE *f = fopen(path, "rb");
  if (!f) exit(3);
  unsigned char *data = malloc(bytes);
  if (!data || fread(data, 1, bytes, f) != bytes || fgetc(f) != EOF || fclose(f))
    exit(3);
  return data;
}

static double ns(struct timespec a, struct timespec b) {
  return (b.tv_sec - a.tv_sec) * 1e9 + (b.tv_nsec - a.tv_nsec);
}

int main(int argc, char **argv) {
  if (argc != 6) return 2;
  const size_t w = positive(argv[1]), h = positive(argv[2]);
  if (w < 2 || h < 2 || w > SIZE_MAX / h / 3) return 2;
  const size_t bytes = w * h * 3, count = (w / 2) * (h / 2);
  bmpread_t r = {(int)w, (int)h, BMPREAD_TOP_DOWN | BMPREAD_BYTE_ALIGN,
                read_rgb(argv[3], bytes)};
  bmpread_t d = {(int)w, (int)h, BMPREAD_TOP_DOWN | BMPREAD_BYTE_ALIGN,
                read_rgb(argv[4], bytes)};
  for (int call = 0; call < 2; ++call) {
    struct timespec start, stop;
    clock_gettime(CLOCK_MONOTONIC, &start);
    gmsd_image_data_t *rg = rgb2gray(&r), *dg = rgb2gray(&d);
    gmsd_image_data_t *map = malloc(count * sizeof(*map));
    double value = 0;
    if (!rg || !dg || !map || !gmsd(rg, dg, w, h, &value, &map)) return 4;
    free(rg); free(dg);
    clock_gettime(CLOCK_MONOTONIC, &stop);
    printf("%s %.0f %.17g\n", call ? "SAMPLE" : "COLD", ns(start, stop), value);
    fflush(stdout);
    if (call == 1) {
      FILE *f = fopen(argv[5], "wbx");
      if (!f || fwrite(map, sizeof(*map), count, f) != count || fclose(f)) return 5;
    }
    free(map);
  }
  free(r.data); free(d.data);
  return 0;
}
