#include "lida_ml.hpp"

#include "util.h"

int main()
{
  auto ml_lib = lida::ML_Library::init((struct lida_ML) {
      .alloc   = malloc,
      .dealloc = free,
      .log     = log_func
  });

  uint32_t dims[2] = { 4, 3 };
  lida::Tensor t1 {dims, LIDA_FORMAT_F32};
  t1.fill_zeros();

  printf("hello tensors!\n");

  print_tensor(t1);
  {
    uint32_t indices[2] = {0};
    float* fst = (float*)t1.get(indices);
    for (uint32_t i = 0; i < dims[0]*dims[1]; i++)
      fst[i] = (float)(i*i);
  }
  print_tensor(t1);

  print_tensor_dim(t1);
  uint32_t tdims[2] = { 1, 0 };
  lida::Tensor t2 = t1.transpose(tdims);
  print_tensor_dim(t2);
  print_tensor(t2);

  uint32_t start[2] = { 2, 1 };
  uint32_t stop[2] = { 3, 4 };
  lida::Tensor t3 = t2.slice(start, stop);
  print_tensor_dim(t3);
  print_tensor(t3);

  lida::Tensor t4 = t3.deep_copy();
  print_tensor(t4);
  // lida_tensor_fill_zeros(t3);
  t4.fill_zeros();
  print_tensor(t1);
  print_tensor(t2);

  uint32_t newdims[2] = { 6, 2 };
  lida::Tensor t5 = t2.reshape(newdims);
  print_tensor(t5);

  float raw_matrix[] = {
    1.0,  2.0,  3.0,  4.0, 5.0,
    6.0,  7.0,  8.0,  9.0, 10.0,
    11.0, 12.0, 13.0, 14.0, 15.0,
    16.0, 17.0, 18.0, 19.0, 20.0,
  };
  uint32_t t6_dims[2] = { 5, 4 };
  auto t6 = lida::Tensor(std::span{raw_matrix}, t6_dims).transpose(tdims);
  print_tensor(t6);

  uint32_t t7_start[2] = { 1, 1 };
  uint32_t t7_stop[2] = { 3, 4 };
  uint32_t t7_shape[2] = { 1, 6 };
  auto t7 = t6.slice(t7_start, t7_stop).reshape(t7_shape);
  print_tensor(t7);

  uint32_t t8_start[2] = { 1, 1 };
  uint32_t t8_stop[2] = { 2, 4 };
  auto t8 = t6.slice(t8_start, t8_stop);
  t8.fill(69.69f);
  // t8.fill_zeros();

  print_tensor(t6);
  print_tensor(t7);
  print_tensor_dim(t8);
  print_tensor(t8);

  uint32_t t6_start[2] = {0, 0};
  uint32_t t6_stop[2] = {4, 3};
  t6 = t6.slice(t6_start, t6_stop);

  uint32_t t9_flip_axes[] = {1};
  auto t9 = t6.flip(t9_flip_axes);
  print_tensor(t9);

  uint32_t t10_flip_axes[] = {0};
  auto t10 = t6.flip(t10_flip_axes);
  print_tensor(t10);

  auto t11 = t10.flip(t9_flip_axes);
  print_tensor(t11);

  auto t12 = t9.flip(t10_flip_axes);
  print_tensor(t12);

  uint32_t t13_start[2] = {0, 0};
  uint32_t t13_stop[2] = {3, 3};
  auto t13 = t6.slice(t13_start, t13_stop);
  print_tensor(t13);

  auto t14 = t13.rot90(0, 1, /*n=*/1);
  print_tensor(t14);

  auto t15 = t13.rot90(0, 1, /*n=*/2);
  print_tensor(t15);

  auto t16 = t13.rot90(0, 1, /*n=*/-1);
  print_tensor(t16);

  return 0;
}
