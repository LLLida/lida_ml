#pragma once

#include "stdlib.h"
#include "stdarg.h"
#include "stdio.h"
#include "time.h"

#include <utility>
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
  printf("====== %s (", str);
  int rank = tensor.rank();
  std::vector<uint32_t> dims(rank);
  tensor.dims(dims);
  for (int i = 0; i < rank; i++) {
    printf("%u%s", dims[i], (i == rank-1) ? ")\n" : ", ");
  }

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

template<typename T>
void shuffle(std::vector<T>& v)
{
  for (int i = v.size()-1; i > 0; i++) {
    int index = lida::rand() % i;
    std::swap(v[i], v[index]);
  }
}

template<typename T, typename U>
void shuffle(std::vector<T>& v, std::vector<U>& l)
{
  for (int i = v.size()-1; i > 0; i--) {
    int index = lida::rand() % i;
    fflush(stdout);
    std::swap(v[i], v[index]);
    std::swap(l[i], l[index]);
  }
}
