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
  void (*backward)(void* udata, const struct lida_Tensor* output, struct lida_Tensor** args);
  int num_args;
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
/* makes a deep copy of tensor. New data is tightly packed in memory */
struct lida_Tensor* lida_tensor_deep_copy(struct lida_Tensor* tensor);
/* does a deep copy if tensor is not packed in memory */
struct lida_Tensor* lida_tensor_reshape(struct lida_Tensor* tensor, const uint32_t dims[], int rank);

struct lida_Tensor* lida_tensor_flip(struct lida_Tensor* tensor, const uint32_t axes[], int num_axes);
/* counter-clockwise rotation for n*90 degrees */
struct lida_Tensor* lida_tensor_rot90(struct lida_Tensor* tensor, uint32_t ax1, uint32_t ax2, int n);

struct lida_Compute_Graph* lida_compute_graph_create(int requires_grad);
void lida_compute_graph_destroy(struct lida_Compute_Graph* cg);
int lida_compute_graph_add_input(struct lida_Compute_Graph* cg, const char* name, const uint32_t dims[], int rank);
int lida_compute_graph_add_parameter(struct lida_Compute_Graph* cg, struct lida_Tensor* parameter, int frozen);
int lida_compute_graph_add_gate(struct lida_Compute_Graph* cg, const struct lida_Gate* gate);
int lida_compute_graph_add_child(struct lida_Compute_Graph* cg, struct lida_Compute_Graph* child);
int lida_compute_graph_set_input(struct lida_Compute_Graph* cg, const char* name, const struct lida_Tensor* tensor);
void lida_compute_graph_forward(struct lida_Compute_Graph* cg);
/* graph doesn't own returned tensor */
const struct lida_Tensor* lida_compute_graph_get_output(struct lida_Compute_Graph* cg, int index);

const struct lida_Gate* lida_gate_plus();

#ifdef __cplusplus
}
#endif

#endif	// LIDA_ML_H
