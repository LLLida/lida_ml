#include "lida_ml.hpp"

#include "util.h"

struct Dataset {
  std::vector<lida::Tensor> images;
  std::vector<char> labels;
  std::vector<float> data;
};

auto load_dataset(const char* imagepath, const char* labelpath)
{
  Dataset dataset;
  FILE* tf = fopen(imagepath, "rb");
  if (tf == NULL) {
    printf("failed to open file '%s'", imagepath);
    return dataset;
  }
  FILE* lf = fopen(labelpath, "rb");
  if (lf == NULL) {
    printf("failed to open file '%s'", labelpath);
    return dataset;
  }

  auto reverseInt = [](int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
  };

  struct {
    uint32_t magic;
    uint32_t count;
    uint32_t nrows;
    uint32_t ncols;
  } t_header;
  fread(&t_header, sizeof(t_header), 1, tf);
  t_header.magic = reverseInt(t_header.magic);
  t_header.count = reverseInt(t_header.count);
  t_header.nrows = reverseInt(t_header.nrows);
  t_header.ncols = reverseInt(t_header.ncols);

  struct {
    uint32_t magic;
    uint32_t count;
  } l_header;
  fread(&l_header, sizeof(l_header), 1, lf);
  l_header.magic = reverseInt(l_header.magic);
  l_header.count = reverseInt(l_header.count);

  if (t_header.magic != 2051 || l_header.magic != 2049) {
    printf("invalid files\n");
    return dataset;
  }
  if (t_header.count != l_header.count) {
    printf("number of samples in image file doesn't match number of samples in label file(%u != %u)",
	   t_header.count, l_header.count);
    return dataset;
  }

  dataset.images.reserve(t_header.count);
  dataset.labels.reserve(l_header.count);

  char buff[28*28];
  dataset.data.resize(t_header.count * sizeof(buff));

  for (int i = 0; i < t_header.count; i++) {
    char label;
    fread(buff, sizeof(buff), 1, tf);
    fread(&label, sizeof(label), 1, lf);

    auto ptr = dataset.data.data() + i * sizeof(buff);
    std::span<float> span(ptr, ptr + sizeof(buff));
    for (int j = 0; j < sizeof(buff); j++) {
      span[j] = (float)buff[j];
    }

    uint32_t dims[1] = {sizeof(buff)};
    dataset.labels.push_back(label);
    dataset.images.push_back(lida::Tensor(span, dims));
  }
  printf("loaded %d images\n", int(dataset.images.size()));

  fclose(tf);
  fclose(lf);
  return dataset;
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("Usage: %s PATH-TO-DATASET\n", argv[0]);
    return -1;
  }

  auto ml_lib = lida::ML_Library::init((struct lida_ML) {
      .alloc   = malloc,
      .dealloc = free,
      .log     = log_func
  });

  // lida::rand_seed(time(NULL));

  printf("Loading dataset... ");
  auto dataset = load_dataset(argv[1], argv[2]);
  shuffle(dataset.images, dataset.labels);

  auto& digit = dataset.images.at(9);
  for (uint32_t i = 0; i < 28; i++)
    for (uint32_t j = 0; j < 28; j++) {
      uint32_t indices[] = {j + i * 28};
      auto v = (float*)digit.get(indices);
      printf("%+003.1f%c", *v, " \n"[j==27]);
    }
  printf("(Label: %d)\n", int(dataset.labels.at(9)));

  return 0;
}
