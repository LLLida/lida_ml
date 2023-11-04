#include "lida_ml.hpp"

#include "util.h"

int main()
{
  auto ml_lib = lida::ML_Library::init((struct lida_ML) {
      .alloc   = malloc,
      .dealloc = free,
      .log     = log_func
  });

  float a_data[] = {
    1.0, 2.0,
    3.0, 4.0,

    10.0, 11.0,
    14.0, 15.0,

    16.0, 100.5,
    1.0, 4.0
  };
  float b_data[] = {
    1.0, 2.0,
    3.0, 4.0,

    66.0, 70.0,
    13.0, 101.0,

    15.0, -1.0,
    69.69, 0.001,
  };
  uint32_t input_shape[3] = { 2, 2, 3 };
  auto a = lida::Tensor(std::span{a_data}, input_shape);
  auto b = lida::Tensor(std::span{b_data}, input_shape);

  lida::Compute_Graph cg{};
  cg.add_input("a", input_shape)
    .add_parameter(b)
    .add_gate(lida::plus());

  float c_data[] = {
    2.0, 4.1,
    5.5, 8.0,

    75.0, 80.0,
    26.9, 115.0,

    32.0, 100.0,
    70.0, 4.004
  };
  auto c_actual = lida::Tensor(std::span{c_data}, input_shape);
  print_tensor(c_actual);

  cg.set_input("a", a);

  lida::SGD_Optimizer optim(0.1);

  int epochs = 10;
  for (int i = 0; i < epochs; i++) {
    cg.forward();
    auto c_pred = cg.get_output(0);

    auto loss = lida::Loss::MSE(c_pred, c_actual);
    printf("MSE loss is %.3f\n", loss.value());
    if (i == epochs-1)
      print_tensor(c_pred);

    cg.zero_grad()
      .backward(loss)
      .optimizer_step(optim);
  }

  return 0;
}
