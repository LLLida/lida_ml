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

  uint8_t buff[28*28];
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

void print_digit(const lida::Tensor& tensor)
{
  for (uint32_t j = 0; j < 28; j++) {
    for (uint32_t i = 0; i < 28; i++) {
      uint32_t indices[] = {i + 28*j};
      float* a = (float*)tensor.get(indices);
      printf("%3d ", int(*a * 255));
    }
    printf("\n");
  }
}

int main(int argc, char** argv)
{
  if (argc != 5) {
    printf("Usage: %s PATH-TO-TRAIN-IMAGES PATH-TO-TRAIN-LABELS PATH-TO-TEST-IMAGES PATH-TO-TEST-LABELS\n", argv[0]);
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

  lida::Linear_Layer layer1(784, 16);
  lida::Linear_Layer layer2(16, 16);
  lida::Linear_Layer layer3(16, 10);

  // Hyper parameters
  const size_t batch_size = 50;
  const int epochs = 2;

  lida::Compute_Graph cg{};
  uint32_t batch_shape[] = {784, batch_size};
  cg.add_input("digit", batch_shape)
    // first layer
    .add_layer(layer1)
    .add_gate(lida::relu())
    // second layer
    .add_layer(layer2)
    .add_gate(lida::relu())
    // third layer
    .add_layer(layer3)
    .add_gate(lida::sigmoid());


  lida::SGD_Optimizer optim(0.001);

  for (int epoch = 0; epoch < epochs; epoch++)
    for (size_t i = 0; i < dataset.images.size()/batch_size; i++) {
      // grab next batch of images
      auto batch_inputs = lida::Tensor::stack({dataset.images.data() + i*batch_size, batch_size});
      // grab next batch of labels
      uint32_t batch_target_shape[] = {10, batch_size};
      auto batch_target = lida::Tensor(batch_target_shape, LIDA_FORMAT_F32);
      batch_target.fill_zeros();
      // one-hot encoding
      for (uint32_t j = 0; j < batch_size; j++) {
	uint32_t indices[] = { uint32_t(dataset.labels[j + i*batch_size]), j };
	auto v = (float*)batch_target.get(indices);
	*v = 1.0;
      }

      cg.set_input("digit", batch_inputs).forward();
      auto output = cg.get_output(0);

      auto loss = lida::Loss::MSE(output, batch_target);
      if (i % 50 == 0)
	printf("MSE loss is %.3f\n", loss.value());

      cg.zero_grad()
	.backward(loss)
	.optimizer_step(optim);
    }

  auto test_dataset = load_dataset(argv[3], argv[4]);

  int count = 0;
  for (size_t batch = 0; batch < test_dataset.images.size()/batch_size; batch++) {
    auto input = lida::Tensor::stack({&test_dataset.images[batch*batch_size], batch_size});
    cg.set_input("digit", input).forward();
    auto output = cg.get_output(0);

    for (uint32_t b = 0; b < batch_size; b++) {
      int label = 0;
      float max_val = 0.0;
      for (uint32_t j = 0; j < 10; j++) {
	uint32_t indices[] = {j, b};
	float* c = (float*)output.get(indices);
	if (*c > max_val) {
	  label = j;
	  max_val = *c;
	}
      }
      count += (label == test_dataset.labels[b + batch*batch_size]);
    }
  }
  printf("Test accuracy: %f\n", (float)count / test_dataset.images.size());

  return 0;
}
