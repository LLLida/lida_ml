#include "lida_ml.hpp"

#include "util.h"

int main()
{
  auto ml_lib = lida::ML_Library::init((struct lida_ML) {
      .alloc   = malloc,
      .dealloc = free,
      .log     = log_func
  });

  printf("Hello digits!\n");

  lida_rand_seed(time(NULL));

  for (int i = 0; i < 10; i++) {
    printf("%u\n", lida_rand());
  }

  return 0;
}
