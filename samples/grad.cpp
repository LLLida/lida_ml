#include "lida_ml.hpp"

#include "util.h"

int main()
{
  auto ml_lib = lida::ML_Library::init((struct lida_ML) {
      .alloc   = malloc,
      .dealloc = free,
      .log     = log_func
  });

  uint32_t input_shape[3] = { 2, 2, 3 };
  lida::Compute_Graph cg{};
  cg.add_input("a", input_shape)
    .add_input("b", input_shape)
    .add_gate(lida::plus());

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
  auto a = lida::Tensor(std::span{a_data}, input_shape);
  auto b = lida::Tensor(std::span{b_data}, input_shape);

  cg.set_input("a", a)
    .set_input("b", b)
    .forward();
  auto c_pred = cg.get_output(0);

  print_tensor(c_pred);

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

  auto loss = lida::Loss::MSE(c_pred, c_actual);
  cg.backward(loss);

  auto grad = cg.get_output_grad(0);
  print_tensor(grad);

  return 0;
}