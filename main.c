#include "lida_ml.h"

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

void print_tensor(struct lida_Tensor* tensor)
{
  uint32_t dims[2];
  lida_tensor_get_dims(tensor, dims, NULL);

  uint32_t indices[2];
  for (uint32_t i = 0; i < dims[1]; i++) {
    for (uint32_t j = 0; j < dims[0]; j++) {
      indices[0] = j;
      indices[1] = i;
      float* val = lida_tensor_get(tensor, indices, 2);
      printf("%f%c", *val, " \n"[j == dims[0]-1]);
    }
  }
  printf("======\n");
}

void print_tensor_dim(struct lida_Tensor* tensor)
{
  uint32_t dims[LIDA_MAX_DIMENSIONALITY];
  int rank;
  lida_tensor_get_dims(tensor, dims, &rank);
  printf("(");
  const char* m[2] = {
    ", ", ")\n"
  };
  for (int i = 0; i < rank; i++) {
    printf("%u%s", dims[i], m[i==rank-1]);
  }
}

int main()
{
  lida_ml_init(&(struct lida_ML) {
      .alloc   = malloc,
      .dealloc = free,
      .log     = log_func
    });

  uint32_t dims[2] = { 4, 3 };
  struct lida_Tensor* t1 = lida_tensor_create(dims, 2, LIDA_FORMAT_F32);
  lida_tensor_fill_zeros(t1);

  printf("hello tensors!\n");

  print_tensor(t1);
  {
    uint32_t indices[2] = {0};
    float* fst = lida_tensor_get(t1, indices, 2);
    for (uint32_t i = 0; i < dims[0]*dims[1]; i++)
      fst[i] = (float)(i*i);
  }
  print_tensor(t1);

  print_tensor_dim(t1);
  uint32_t tdims[2] = { 1, 0 };
  struct lida_Tensor* t2 = lida_tensor_transpose(t1, tdims, 2);
  print_tensor_dim(t2);
  print_tensor(t2);

  uint32_t start[2] = { 2, 1 };
  uint32_t stop[2] = { 3, 4 };
  struct lida_Tensor* t3 = lida_tensor_slice(t2, start, stop, 2);
  print_tensor_dim(t3);
  print_tensor(t3);

  lida_ml_done();
  return 0;
}
