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
      span[j] = (float)buff[j] / 255.0;
    }

    dataset.labels.push_back(label);
    uint32_t dims[1] = {sizeof(buff)};
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

  lida::rand_seed(time(NULL));

  printf("Loading dataset... ");
  auto dataset = load_dataset(argv[1], argv[2]);
  shuffle(dataset.images, dataset.labels);

  const size_t batch_size = 96;

  auto batch_inputs = lida::Tensor::stack({dataset.images.data(), batch_size});

  uint32_t batch_target_shape[] = {10, batch_size};
  auto batch_target = lida::Tensor(batch_target_shape, LIDA_FORMAT_F32);
  batch_target.fill_zeros();
  for (uint32_t i = 0; i < batch_size; i++) {
    // one-hot encoding
    uint32_t indices[] = { uint32_t(dataset.labels[i]), i };
    auto v = (float*)batch_target.get(indices);
    *v = 1.0;
  }

  uint32_t w1_shape[] = {784, 16};
  lida::Tensor w1(w1_shape, LIDA_FORMAT_F32);
  w1.fill_normal();

  uint32_t w2_shape[] = {16, 16};
  lida::Tensor w2(w2_shape, LIDA_FORMAT_F32);
  w2.fill_normal();

  uint32_t w3_shape[] = {16, 10};
  lida::Tensor w3(w3_shape, LIDA_FORMAT_F32);
  w3.fill_normal();

#define USE_BIASES 0

#if USE_BIASES
  uint32_t b1_shape[] = {784};
  lida::Tensor b1(b1_shape, LIDA_FORMAT_F32);
  b1.fill_uniform(-1.0, 1.0);

  uint32_t b2_shape[] = {16};
  lida::Tensor b2(b2_shape, LIDA_FORMAT_F32);
  b2.fill_uniform(-1.0, 1.0);
#endif

  lida::Compute_Graph cg{};
  uint32_t batch_shape[] = {784, batch_size};
  cg.add_input("digit", batch_shape)
    // first layer
    .add_parameter(w1)
    .add_gate(lida::mm())
#if USE_BIASES
    .add_parameter(b1)
    .add_gate(lida::plus())
#endif
    .add_gate(lida::relu())
    // second layer
    .add_parameter(w2)
    .add_gate(lida::mm())
#if USE_BIASES
    .add_parameter(b2)
    .add_gate(lida::plus())
#endif
    .add_gate(lida::relu())
    // third layer
    .add_parameter(w3)
    .add_gate(lida::mm())
    .add_gate(lida::sigmoid());

  lida::SGD_Optimizer optim(0.01);

  for (int i = 0; i < 5; i++) {
    cg.set_input("digit", batch_inputs);
    cg.forward();
    auto output = cg.get_output(0);

    auto loss = lida::Loss::MSE(output, batch_target);
    printf("MSE loss is %.3f\n", loss.value());

    cg.zero_grad()
      .backward(loss)
      .optimizer_step(optim);
  }

  cg.forward();
  auto output = cg.get_output(0);
  auto loss = lida::Loss::MSE(output, batch_target);
  printf("MSE loss is %.3f\n", loss.value());

  return 0;
}
