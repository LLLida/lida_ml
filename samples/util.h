#pragma once

#include "stdlib.h"
#include "stdarg.h"
#include "stdio.h"
#include "time.h"

#include <vector>

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
  int rank = tensor.rank();
  std::vector<uint32_t> dims(rank);
  tensor.dims(dims);

  std::vector<uint32_t> indices(rank, 0u);
  while (indices.back() != dims.back()) {
    for (uint32_t i = 0; i < dims[1]; i++) {
      for (uint32_t j = 0; j < dims[0]; j++) {
	indices[0] = j;
	indices[1] = i;
	float* val = (float*)tensor.get(indices);
	for (int i = 2; i < rank; i++) {
	  printf("\t");
	}
	printf("%f%c", *val, " \n"[j == dims[0]-1]);
      }
    }
    for (int i = 2; i < rank; i++) {
      indices[i]++;
      for (int j = rank-1; j >= i; j--) {
	printf("====");
      }
      printf("\n");
      if (indices[i] == dims[i]) {
	if (i != rank-1)
	  indices[i] = 0;
      } else {
	break;
      }
    }
  }
  // for (uint32_t i = 0; i < dims[1]; i++) {
  // for (uint32_t j = 0; j < dims[0]; j++) {
  //     indices[0] = j;
  //     indices[1] = i;
  //     float* val = (float*)tensor.get(indices);
  //     printf("%f%c", *val, " \n"[j == dims[0]-1]);
  //   }
  // }
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
