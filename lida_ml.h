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

struct lida_Compute_Graph;

struct lida_Gate {
  const char* name;
  void* udata;
  struct lida_Tensor* (*forward)(void* udata, const struct lida_Tensor** args);
  /* NOTE: this function must ADD gradients, not ASSIGN them */
  void (*backward)(void* udata, const struct lida_Tensor* output, const struct lida_Tensor* args[], struct lida_Tensor* grads[]);
  size_t num_args;
};

struct lida_Loss {
  void* udata;
  void (*forward)(struct lida_Loss* self, const struct lida_Tensor* pred, const struct lida_Tensor* target);
  struct lida_Tensor* (*backward)(struct lida_Loss* self);

  float value;
  const struct lida_Tensor* pred;
  const struct lida_Tensor* target;
};

struct lida_Optimizer {
  void* udata;
  void (*step)(struct lida_Optimizer* self, struct lida_Tensor* param, const struct lida_Tensor* grad);
};

void lida_ml_init(const struct lida_ML* ml);
void lida_ml_done();
struct lida_Tensor* lida_tensor_create(const uint32_t dims[], int rank, lida_Format format);
void lida_tensor_destroy(struct lida_Tensor* tensor);
struct lida_Tensor* lida_tensor_create_from_memory(void* memory, uint32_t bytes, const uint32_t dims[], int rank, lida_Format format);
lida_Format lida_tensor_get_format(const struct lida_Tensor* tensor);
/* dims or rank can be null */
void lida_tensor_get_dims(const struct lida_Tensor* tensor, uint32_t* dims, int* rank);
/* O(1) indexing operation */
void* lida_tensor_get(struct lida_Tensor* tensor, const uint32_t indices[], int num_indices);
void* lida_tensor_get_unchecked(const struct lida_Tensor* tensor, const uint32_t indices[]);
/* Get number of values in tensor */
uint32_t lida_tensor_size(const struct lida_Tensor* tensor);
void lida_tensor_fill_zeros(struct lida_Tensor* tensor);
void lida_tensor_fill(struct lida_Tensor* tensor, const void* obj);
/* O(1) */
struct lida_Tensor* lida_tensor_transpose(struct lida_Tensor* tensor, const uint32_t dims[], int rank);
/* O(1) */
struct lida_Tensor* lida_tensor_slice(struct lida_Tensor* tensor, const uint32_t left[], const uint32_t right[], int rank);
struct lida_Tensor* lida_tensor_alike(const struct lida_Tensor* tensor);
/* O(1). makes a copy of tensor without copying it's data */
struct lida_Tensor* lida_tensor_copy(struct lida_Tensor* tensor);
/* makes a deep copy of tensor. New data is tightly packed in memory */
struct lida_Tensor* lida_tensor_deep_copy(struct lida_Tensor* tensor);
/* does a deep copy if tensor is not packed in memory */
struct lida_Tensor* lida_tensor_reshape(struct lida_Tensor* tensor, const uint32_t dims[], int rank);

struct lida_Tensor* lida_tensor_flip(struct lida_Tensor* tensor, const uint32_t axes[], int num_axes);
/* counter-clockwise rotation for n*90 degrees */
struct lida_Tensor* lida_tensor_rot90(struct lida_Tensor* tensor, uint32_t ax1, uint32_t ax2, int n);
/* add tensor multiplied by a scalar to other tensor */
int lida_tensor_add(struct lida_Tensor* tensor, struct lida_Tensor* other, float scalar);

/* usage:
    LIDA_TENSOR_ITER_LOOP(tensor, indices) {
      void* elem = lida_tensor_get_unchecked(tensor, indices);
      ... do some things
      LIDA_TENSOR_ITER_STEP(tensor, indices);
    }
 */
#define LIDA_TENSOR_ITER_LOOP(tensor, indices) int32_t rank_lida__;	\
  uint32_t dims_lida__[LIDA_MAX_DIMENSIONALITY];			\
  lida_tensor_get_dims(tensor, dims_lida__, &rank_lida__);		\
  uint32_t indices[LIDA_MAX_DIMENSIONALITY+1] = {0};			\
  while (indices[rank_lida__] == 0)
#define LIDA_TENSOR_ITER_STEP(tensor, indices) do {			\
    for (int32_t i = 0; i <= rank_lida__; i++) {			\
      indices[i]++;							\
      if (i != rank_lida__ && indices[i] == dims_lida__[i]) {	\
	indices[i] = 0;							\
      } else {								\
	break;								\
      }									\
    }									\
  } while (0)

struct lida_Compute_Graph* lida_compute_graph_create(int requires_grad);
void lida_compute_graph_destroy(struct lida_Compute_Graph* cg);
int lida_compute_graph_add_input(struct lida_Compute_Graph* cg, const char* name, const uint32_t dims[], int rank);
int lida_compute_graph_add_parameter(struct lida_Compute_Graph* cg, struct lida_Tensor* parameter, int frozen);
int lida_compute_graph_add_gate(struct lida_Compute_Graph* cg, const struct lida_Gate* gate);
int lida_compute_graph_add_child(struct lida_Compute_Graph* cg, struct lida_Compute_Graph* child);
int lida_compute_graph_set_input(struct lida_Compute_Graph* cg, const char* name, const struct lida_Tensor* tensor);
void lida_compute_graph_forward(struct lida_Compute_Graph* cg);
void lida_compute_graph_zero_grad(struct lida_Compute_Graph* cg);
/* number of losses must match number of outputs */
void lida_compute_graph_backward(struct lida_Compute_Graph* cg, struct lida_Loss* losses, int count);
/* graph doesn't own returned tensor */
const struct lida_Tensor* lida_compute_graph_get_output(struct lida_Compute_Graph* cg, size_t index);
void lida_compute_graph_optimizer_step(struct lida_Compute_Graph* cg, struct lida_Optimizer* opt);

#ifdef __cplusplus
}
#endif

#endif	// LIDA_ML_H
