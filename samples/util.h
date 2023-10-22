#pragma once

#include "stdlib.h"
#include "stdarg.h"
#include "stdio.h"

static void
log_func(int sev, const char* fmt, ...)
{
  (void)sev;
  va_list ap;
  va_start(ap, fmt);
  vprintf(fmt, ap);
  va_end(ap);
  printf("\n");
}

void print_tensor_(const lida::Tensor& tensor, const char* str)
{
  printf("====== %s\n", str);
  uint32_t dims[2];
  tensor.dims(dims);

  uint32_t indices[2];
  for (uint32_t i = 0; i < dims[1]; i++) {
    for (uint32_t j = 0; j < dims[0]; j++) {
      indices[0] = j;
      indices[1] = i;
      float* val = (float*)tensor.get(indices);
      printf("%f%c", *val, " \n"[j == dims[0]-1]);
    }
  }
}
#define print_tensor(tensor) print_tensor_(tensor, #tensor)

void print_tensor_dim(const lida::Tensor& tensor)
{
  uint32_t dims[LIDA_MAX_DIMENSIONALITY];
  int rank = tensor.rank();
  tensor.dims({dims, size_t(rank)});
  printf("(");
  const char* m[2] = {
    ", ", ")\n"
  };
  for (int i = 0; i < rank; i++) {
    printf("%u%s", dims[i], m[i==rank-1]);
  }
}
