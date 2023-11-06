#include "lida_ml.hpp"

#include "util.h"

int main()
{
  auto ml_lib = lida::ML_Library::init((struct lida_ML) {
      .alloc   = malloc,
      .dealloc = free,
      .log     = log_func
    });

  // lida::rand_seed(time(NULL));

  size_t count = 15;
  uint32_t input_size = 2;
  uint32_t output_size = 2;

  std::vector<float> inputs(count*input_size);
  std::vector<float> targets(count*output_size);
  for (size_t i = 0; i < count; i++) {
    float x = lida::rand_uniform()*5.0 - 2.5;
    float y = lida::rand_uniform()*5.0 - 2.5;

    inputs[input_size*i]   = x;
    inputs[input_size*i+1] = y;

    targets[output_size*i]   = x + 0.45*y;
    targets[output_size*i+1] = x - y;
  }

  uint32_t input_shape[] = {input_size, (uint32_t)count};
  lida::Tensor input(std::span{inputs}, input_shape);
  uint32_t target_shape[] = {output_size, (uint32_t)count};
  lida::Tensor target(std::span{targets}, target_shape);

  uint32_t w_shape[] = { input_size, output_size };
  lida::Tensor w(w_shape, LIDA_FORMAT_F32);
  w.fill_normal();

  lida::Compute_Graph cg{};
  cg.add_input("x", input_shape)
    .add_parameter(w)
    .add_gate(lida::mm());

  lida::SGD_Optimizer optim(0.01);
  for (int i = 0; i < 15; i++) {
    cg.set_input("x", input).forward();
    auto y = cg.get_output(0);

    auto loss = lida::Loss::MSE(y, target);
    if (i % 10 == 0)
      printf("MSE loss is %.3f\n", loss.value());

    cg.zero_grad()
      .backward(loss)
      .optimizer_step(optim);

    uint32_t indices[] = {0, 0};
    float* v = (float*)w.get(indices);
    printf("Learned: %f %f\n", v[0], v[1]);
  }

  return 0;
}
