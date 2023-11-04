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

  for (int i = 0; i < 2; i++)
    if (grads[i]) {
      LIDA_TENSOR_ITER_LOOP(grads[i], indices) {
	float* y = lida_tensor_get_unchecked(output, indices);
	float* g = lida_tensor_get_unchecked(grads[i], indices);
	*g += *y;
	LIDA_TENSOR_ITER_STEP(grads[i], indices);
      }
    }
}
static struct lida_Tensor*
relu_gate_forward(void* udata, const struct lida_Tensor** args)
{
  (void)udata;

  const struct lida_Tensor* x = args[0];
  struct lida_Tensor* y = lida_tensor_alike(x);
  LIDA_TENSOR_ITER_LOOP(y, indices) {
    float* xi = lida_tensor_get_unchecked(x, indices);
    float* yi = lida_tensor_get_unchecked(y, indices);
    if (*xi > 0.0) {
      *yi = *xi;
    } else {
      *yi = 0.0;
    }
    LIDA_TENSOR_ITER_STEP(y, indices);
  }
  return y;
}

static void
relu_gate_backward(void* udata, const struct lida_Tensor* output, const struct lida_Tensor* args[], struct lida_Tensor* grads[])
{
  (void)udata;
  (void)output;

  if (grads[0]) {
    LIDA_TENSOR_ITER_LOOP(grads[0], indices) {
      float* x = lida_tensor_get_unchecked(args[0], indices);
      float* g = lida_tensor_get_unchecked(grads[0], indices);
      if (*x > 0.0) {
	float* y = lida_tensor_get_unchecked(output, indices);
	*g += *y;
      }
      LIDA_TENSOR_ITER_STEP(grads[0], indices);
    }
  }
}

static void
MSE_Loss_forward(struct lida_Loss* self, const struct lida_Tensor* pred, const struct lida_Tensor* target)
{
  if (compare_tensor_shapes(pred, target) != 0) {
    // LOG_ERROR("MSE loss: tensors must be the same shape");
    return;
  }

  self->value = 0.0;
  self->pred = pred;
  self->target = target;

  LIDA_TENSOR_ITER_LOOP(pred, indices) {
    float* y1 = lida_tensor_get_unchecked(pred, indices);
    float* y2 = lida_tensor_get_unchecked(target, indices);
    float d = *y1-*y2;
    self->value += d*d;
    LIDA_TENSOR_ITER_STEP(pred, indices);
  }

  self->value /= (float)lida_tensor_size(pred);
}

static struct lida_Tensor*
MSE_Loss_backward(struct lida_Loss* self)
{
  uint32_t dims[LIDA_MAX_DIMENSIONALITY];
  int rank;
  lida_tensor_get_dims(self->pred, dims, &rank);

  // float n = (float)lida_tensor_size(self->pred);

  struct lida_Tensor* grad = lida_tensor_create(dims, rank, lida_tensor_get_format(self->pred));
  LIDA_TENSOR_ITER_LOOP(grad, indices) {
    float* y1 = lida_tensor_get_unchecked(self->pred, indices);
    float* y2 = lida_tensor_get_unchecked(self->target, indices);
    float* g = lida_tensor_get_unchecked(grad, indices);
    // FIXME: should we divide by n?
    *g = 2.0 * (*y1 - *y2);
    LIDA_TENSOR_ITER_STEP(grad, indices);
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
  if (g_relu_gate.name == NULL) {
    g_relu_gate = (struct lida_Gate) {
      .name = "ReLU",
      .forward = &relu_gate_forward,
      .backward = &relu_gate_backward,
      .num_args = 1
    };
  }
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
