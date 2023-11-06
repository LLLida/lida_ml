#include "lida_ml.hpp"

#include "util.h"

int main()
{
  auto ml_lib = lida::ML_Library::init((struct lida_ML) {
      .alloc   = malloc,
      .dealloc = free,
      .log     = log_func
    });

  std::vector<float> inputs = {
    0.2, 0.4,
    0.6, 0.7,
    2.8, 0.9,
    0.1, -1.2,
    0.9, 0.3
  };
  std::vector<float> targets(inputs.size()/2);
  for (size_t i = 0; i < targets.size(); i++) {
    targets[i] = inputs[2*i] + inputs[2*i+1];
  }

  uint32_t input_shape[] = {2, (uint32_t)targets.size()};
  lida::Tensor input(std::span{inputs}, input_shape);
  uint32_t target_shape[] = {1, (uint32_t)targets.size()};
  lida::Tensor target(std::span{targets}, target_shape);

  uint32_t w_shape[] = { 2, 1 };
  lida::Tensor w(w_shape, LIDA_FORMAT_F32);

  lida::Compute_Graph cg{};
  cg.add_input("x", input_shape)
    .add_parameter(w)
    .add_gate(lida::mm());

  lida::SGD_Optimizer optim(0.1);
  for (int i = 0; i < 11; i++) {
    cg.set_input("x", input).forward();
    auto y = cg.get_output(0);

    auto loss = lida::Loss::MSE(y, target);
    if (i % 10 == 0)
      printf("MSE loss is %.3f\n", loss.value());

    cg.zero_grad()
      .backward(loss)
      .optimizer_step(optim);
  }

  uint32_t indices[] = {0, 0};
  float* v = (float*)w.get(indices);
  printf("Learned: %f %f\n", v[0], v[1]);

  return 0;
}
