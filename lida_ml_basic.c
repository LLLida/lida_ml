#include "lida_ml_basic.h"

#include "math.h"


/// static functions

static int
compare_tensor_shapes(const struct lida_Tensor* a, const struct lida_Tensor* b)
{
  int arank, brank;
  uint32_t adims[LIDA_MAX_DIMENSIONALITY];
  uint32_t bdims[LIDA_MAX_DIMENSIONALITY];
  lida_tensor_get_dims(a, adims, &arank);
  lida_tensor_get_dims(b, bdims, &brank);

  if (arank != brank) {
    return (arank > brank) - (arank < brank);
  }
  for (int32_t i = 0; i < arank; i++) {
    uint32_t x = adims[i];
    uint32_t y = bdims[i];
    if (x != y) {
      return (x > y) - (x < y);
    }
  }
  return 0;
}

#define PERFORM_ELEMENTWISE_OP2(a, b, c, type, op) {			\
  uint32_t indices[LIDA_MAX_DIMENSIONALITY] = {0};			\
  while (indices[rank-1] < dims[rank-1]) {	\
    type* va = lida_tensor_get_unchecked(a, indices);			\
    type* vb = lida_tensor_get_unchecked(b, indices);			\
    type* vc = lida_tensor_get_unchecked(c, indices);			\
    *vc = *va op *vb;							\
    for (int32_t i = 0; i < rank; i++) {			\
      indices[i]++;							\
      if (indices[i] == dims[i] && i < rank-1) {	\
	indices[i] = 0;							\
      } else {								\
	break;								\
      }									\
    }									\
  }									\
  }

static struct lida_Tensor*
plus_gate_forward(void* udata, const struct lida_Tensor** args)
{
  (void)udata;

  const struct lida_Tensor* a = args[0];
  const struct lida_Tensor* b = args[1];
  if (compare_tensor_shapes(a, b) != 0) {
    // LOG_ERROR("+: tensors must be the same shape");
    return NULL;
  }
  lida_Format format = lida_tensor_get_format(a);
  if (format != lida_tensor_get_format(b)) {
    // LOG_ERROR("+: tensors must have the same format");
    return NULL;
  }

  int rank;
  uint32_t dims[LIDA_MAX_DIMENSIONALITY];
  lida_tensor_get_dims(a, dims, &rank);

  struct lida_Tensor* c = lida_tensor_create(dims, rank, format);
  switch (format)
    {
    case LIDA_FORMAT_I32:
      PERFORM_ELEMENTWISE_OP2(a, b, c, int32_t, +);
      break;
    case LIDA_FORMAT_U32:
      PERFORM_ELEMENTWISE_OP2(a, b, c, uint32_t, +);
      break;
    case LIDA_FORMAT_F32:
      PERFORM_ELEMENTWISE_OP2(a, b, c, float, +);
      break;
    default:
      // LOG_ERROR("+ on this format is not supported");
      lida_tensor_destroy(c);
      return NULL;
    }
  return c;
}

static void
plus_gate_backward(void* udata, const struct lida_Tensor* output, const struct lida_Tensor* args[], struct lida_Tensor* grads[])
{
  (void)udata;
  (void)args;

  if (args[0]) grads[0] = lida_tensor_copy((struct lida_Tensor*)output);
  if (args[1]) grads[1] = lida_tensor_copy((struct lida_Tensor*)output);
}

static void
MSE_Loss_forward(struct lida_Loss* self, const struct lida_Tensor* pred, const struct lida_Tensor* actual)
{
  if (compare_tensor_shapes(pred, actual) != 0) {
    // LOG_ERROR("MSE loss: tensors must be the same shape");
    return;
  }

  int rank;
  uint32_t dims[LIDA_MAX_DIMENSIONALITY];
  lida_tensor_get_dims(pred, dims, &rank);

  self->value = 0.0;
  self->pred = pred;
  self->actual = actual;
  uint32_t indices[LIDA_MAX_DIMENSIONALITY];
  while (indices[rank-1] < dims[rank-1]) {
    float* y1 = lida_tensor_get_unchecked(pred, indices);
    float* y2 = lida_tensor_get_unchecked(actual, indices);
    float d = y1-y2;
    self->value += d*d;
    for (int i = 0; i < rank; i++) {
      indices[i]++;
      if (indices[i] == dims[i]) {
	if (i != rank-1)
	  indices[i] = 0;
      } else {
	break;
      }
    }
  }
}

static struct lida_Tensor*
MSE_Loss_backward(struct lida_Loss* self)
{
  uint32_t dims[LIDA_MAX_DIMENSIONALITY];
  int rank;
  lida_tensor_get_dims(self->pred, dims, &rank);

  struct lida_Tensor* grad = lida_tensor_create(dims, rank, lida_tensor_get_format(self->pred));
  uint32_t indices[LIDA_MAX_DIMENSIONALITY];
  while (indices[rank-1] < dims[rank-1]) {
    float* y1 = lida_tensor_get_unchecked(self->pred, indices);
    float* y2 = lida_tensor_get_unchecked(self->actual, indices);
    float* g = lida_tensor_get_unchecked(grad, indices);
    *g = 2 * fabs(*y2 - *y1);
    for (int i = 0; i < rank; i++) {
      indices[i]++;
      if (indices[i] == dims[i]) {
	if (i != rank-1)
	  indices[i] = 0;
      } else {
	break;
      }
    }
  }
  return grad;
}


/// implementation

static struct lida_Gate g_plus_gate;
static struct lida_Gate g_mul_gate;

static struct lida_Gate g_relu_gate;
static struct lida_Gate g_sigmoid_gate;
static struct lida_Gate g_tanh_gate;

const struct lida_Gate*
lida_gate_plus()
{
  if (g_plus_gate.name == NULL) {
    g_plus_gate = (struct lida_Gate) {
      .name = "+",
      .forward = &plus_gate_forward,
      .backward = &plus_gate_backward,
      .num_args = 2
    };
  }
  return &g_plus_gate;
}

const struct lida_Gate*
lida_gate_mul()
{
  return &g_mul_gate;
}

const struct lida_Gate*
lida_gate_relu()
{
  return &g_relu_gate;
}

const struct lida_Gate*
lida_gate_sigmoid()
{
  return &g_sigmoid_gate;
}

const struct lida_Gate*
lida_gate_tanh()
{
  return &g_tanh_gate;
}

void
lida_MSE_loss(struct lida_Loss* loss)
{
  loss->udata = NULL;
  loss->forward = MSE_Loss_forward;
  loss->backward = MSE_Loss_backward;
}
