#ifndef LIDA_ML_H
#define LIDA_ML_H

#include "stdint.h"
#include "stddef.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef LIDA_MAX_DIMENSIONALITY
#define LIDA_MAX_DIMENSIONALITY 8
#endif

struct lida_ML {
  void* (*alloc)(size_t bytes);
  void (*dealloc)(void* mem);
  void (*log)(int severity, const char* fmt, ...);
};

typedef enum {
  LIDA_FORMAT_U16 = 0,
  LIDA_FORMAT_U32 = 1,
  LIDA_FORMAT_I16 = 2,
  LIDA_FORMAT_I32 = 3,
  LIDA_FORMAT_F16 = 4,
  LIDA_FORMAT_F32 = 5,

  LIDA_FORMAT_MASK = 7
} lida_Format;

struct lida_Tensor;


void lida_ml_init(const struct lida_ML* ml);
void lida_ml_done();
struct lida_Tensor* lida_tensor_create(const uint32_t dims[], int rank, lida_Format format);
void lida_tensor_destroy(struct lida_Tensor* tensor);
struct lida_Tensor* lida_tensor_create_from_memory(void* memory, uint32_t bytes, const uint32_t dims[], int rank, lida_Format format);
/* dims or rank can be null */
void lida_tensor_get_dims(const struct lida_Tensor* tensor, uint32_t* dims, int* rank);
void* lida_tensor_get(struct lida_Tensor* tensor, const uint32_t indices[], int num_indices);
uint32_t lida_tensor_size(const struct lida_Tensor* tensor);
void lida_tensor_fill_zeros(struct lida_Tensor* tensor);
struct lida_Tensor* lida_tensor_transpose(struct lida_Tensor* tensor, const uint32_t dims[], int rank);
struct lida_Tensor* lida_tensor_slice(struct lida_Tensor* tensor, const uint32_t left[], const uint32_t right[], int rank);
struct lida_Tensor* lida_tensor_deep_copy(struct lida_Tensor* tensor);
/* does a deep copy if tensor is not packed in memory */
struct lida_Tensor* lida_tensor_reshape(struct lida_Tensor* tensor, const uint32_t dims[], int rank);

#ifdef __cplusplus
}
#endif

#endif	// LIDA_ML_H
