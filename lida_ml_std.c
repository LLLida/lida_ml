#include "lida_ml_std.h"

#include "math.h"

#define LOG_DEBUG(...) lida_ml_get_log()(0, __VA_ARGS__)
#define LOG_INFO(...)  lida_ml_get_log()(1, __VA_ARGS__)
#define LOG_WARN(...)  lida_ml_get_log()(2, __VA_ARGS__)
#define LOG_ERROR(...) lida_ml_get_log()(3, __VA_ARGS__)


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

  int arank;
  uint32_t adims[LIDA_MAX_DIMENSIONALITY];
  lida_tensor_get_dims(a, adims, &arank);
  int brank;
  uint32_t bdims[LIDA_MAX_DIMENSIONALITY];
  lida_tensor_get_dims(b, bdims, &brank);

  if (arank != brank && arank - brank != 1) {
    LOG_ERROR("+: tensor shapes don't match");
    return NULL;
  }

  lida_Format format = lida_tensor_get_format(a);
  if (format != lida_tensor_get_format(b)) {
    LOG_ERROR("+: tensors must have the same format");
    return NULL;
  }

  struct lida_Tensor* c = lida_tensor_create(adims, arank, format);
  LIDA_TENSOR_ITER_LOOP(c, indices) {
    void* ai = lida_tensor_get_unchecked(a, indices);
    void* bi = lida_tensor_get_unchecked(b, indices);
    void* ci = lida_tensor_get_unchecked(c, indices);
    switch (format)
      {
      case LIDA_FORMAT_I32:
	*(int32_t*)ci = *(int32_t*)ai + *(int32_t*)bi;
	break;
      case LIDA_FORMAT_U32:
	*(uint32_t*)ci = *(uint32_t*)ai + *(uint32_t*)bi;
	break;
      case LIDA_FORMAT_F32:
	*(float*)ci = *(float*)ai + *(float*)bi;
	break;
      default:
	LOG_ERROR("+ on this format is not supported");
	lida_tensor_destroy(c);
	return NULL;
      }
    LIDA_TENSOR_ITER_STEP(c, indices);
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
mul_gate_forward(void* udata, const struct lida_Tensor** args)
{
  (void)udata;

  const struct lida_Tensor* a = args[0];
  const struct lida_Tensor* b = args[1];
  if (compare_tensor_shapes(a, b) != 0) {
    LOG_ERROR("*: tensors must be the same shape");
    return NULL;
  }
  lida_Format format = lida_tensor_get_format(a);
  if (format != lida_tensor_get_format(b)) {
    LOG_ERROR("*: tensors must have the same format");
    return NULL;
  }

  int rank;
  uint32_t dims[LIDA_MAX_DIMENSIONALITY];
  lida_tensor_get_dims(a, dims, &rank);

  struct lida_Tensor* c = lida_tensor_create(dims, rank, format);
  switch (format)
    {
    case LIDA_FORMAT_I32:
      PERFORM_ELEMENTWISE_OP2(a, b, c, int32_t, *);
      break;
    case LIDA_FORMAT_U32:
      PERFORM_ELEMENTWISE_OP2(a, b, c, uint32_t, *);
      break;
    case LIDA_FORMAT_F32:
      PERFORM_ELEMENTWISE_OP2(a, b, c, float, *);
      break;
    default:
      LOG_ERROR("+ on this format is not supported");
      lida_tensor_destroy(c);
      return NULL;
    }
  return c;
}

static void
mul_gate_backward(void* udata, const struct lida_Tensor* output, const struct lida_Tensor* args[], struct lida_Tensor* grads[])
{
  (void)udata;
  (void)args;

  // f(x, y) = x * y
  // df/dx = y
  // df/dy = x
  for (int i = 0; i < 2; i++)
    if (grads[i]) {
      LIDA_TENSOR_ITER_LOOP(grads[i], indices) {
	float* y = lida_tensor_get_unchecked(output, indices);
	float* other = lida_tensor_get_unchecked(args[1-i], indices);
	float* g = lida_tensor_get_unchecked(grads[i], indices);
	*g += *y * *other;
	LIDA_TENSOR_ITER_STEP(grads[i], indices);
      }
    }
}

static struct lida_Tensor*
mm_gate_forward(void* udata, const struct lida_Tensor** args)
{
  (void)udata;

  const struct lida_Tensor* x = args[0];
  const struct lida_Tensor* W = args[1];

  lida_Format format = lida_tensor_get_format(W);
  if (format != LIDA_FORMAT_F32 && format != lida_tensor_get_format(x)) {
    LOG_ERROR("mm: tensors must have the float format");
    return NULL;
  }

  int rank;
  uint32_t dims[LIDA_MAX_DIMENSIONALITY];
  lida_tensor_get_dims(W, dims, &rank);
  if (rank != 2) {
    LOG_ERROR("mm: second argument must be a matrix");
    return NULL;
  }
  uint32_t width = dims[0];
  uint32_t height = dims[1];
  lida_tensor_get_dims(x, dims, &rank);
  if (rank != 2) {
    LOG_ERROR("mm: first argument must be a batch of vectors");
    return NULL;
  }
  if (width != dims[0]) {
    LOG_ERROR("mm: matrix width must be equal to height of vector");
    return NULL;
  }

  dims[0] = height;
  struct lida_Tensor* c = lida_tensor_create(dims, 2, LIDA_FORMAT_F32);
  lida_tensor_fill_zeros(c);
  for (uint32_t b = 0; b < dims[1]; b++)
    for (uint32_t j = 0; j < height; j++)
      // NOTE: we could easily vectorize this loop
      for (uint32_t i = 0; i < width; i++) {
	uint32_t indices[2] = { i, j };
	float* w = lida_tensor_get_unchecked(W, indices);
	indices[1] = b;
	float* x_ = lida_tensor_get_unchecked(x, indices);
	indices[0] = j;
	float* y = lida_tensor_get_unchecked(c, indices);
	*y += *w * *x_;
      }
  return c;
}

static void
mm_gate_backward(void* udata, const struct lida_Tensor* output, const struct lida_Tensor* args[], struct lida_Tensor* grads[])
{
  (void)udata;
  (void)args;

  const struct lida_Tensor* a = args[0];
  const struct lida_Tensor* W = args[1];

  uint32_t dims[2];
  lida_tensor_get_dims(W, dims, NULL);
  const uint32_t width = dims[0], height = dims[1];
  lida_tensor_get_dims(a, dims, NULL);
  const uint32_t batches = dims[1];

  if (grads[0]) {
    /* add W^T y */
    for (uint32_t k = 0; k < batches; k++)
      for (uint32_t j = 0; j < height; j++)
	for (uint32_t i = 0; i < width; i++) {
	  uint32_t indices[2] = {i, j};
	  float* w = lida_tensor_get_unchecked(W, indices);
	  indices[0] = i;
	  indices[1] = k;
	  float* da = lida_tensor_get_unchecked(grads[0], indices);
	  indices[0] = j;
	  float* y = lida_tensor_get_unchecked(output, indices);
	  *da += *w * *y;
	}
  }

  if (grads[1]) {
    /* add Jacobi matrices */
    for (uint32_t k = 0; k < batches; k++)
      for (uint32_t j = 0; j < height; j++)
	for (uint32_t i = 0; i < width; i++) {
	  uint32_t indices[2] = {i, j};
	  float* dw = lida_tensor_get_unchecked(grads[1], indices);
	  indices[0] = i;
	  indices[1] = k;
	  float* da = lida_tensor_get_unchecked(args[0], indices);
	  indices[0] = j;
	  float* y = lida_tensor_get_unchecked(output, indices);
	  *dw += *da * *y;
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

static struct lida_Tensor*
sigmoid_gate_forward(void* udata, const struct lida_Tensor** args)
{
  (void)udata;

  const struct lida_Tensor* x = args[0];
  struct lida_Tensor* y = lida_tensor_alike(x);
  LIDA_TENSOR_ITER_LOOP(y, indices) {
    float* xi = lida_tensor_get_unchecked(x, indices);
    float* yi = lida_tensor_get_unchecked(y, indices);
    *yi = 1.0 / (1.0 + expf(-*xi));
    LIDA_TENSOR_ITER_STEP(y, indices);
  }
  return y;
}

static void
sigmoid_gate_backward(void* udata, const struct lida_Tensor* output, const struct lida_Tensor* args[], struct lida_Tensor* grads[])
{
  (void)udata;
  (void)output;

  if (grads[0]) {
    LIDA_TENSOR_ITER_LOOP(grads[0], indices) {
      float* x = lida_tensor_get_unchecked(args[0], indices);
      float* y = lida_tensor_get_unchecked(output, indices);
      float* g = lida_tensor_get_unchecked(grads[0], indices);
      float s = 1.0 / (1.0 + expf(-*x));
      *g += s * (1.0 - s) * *y;
      LIDA_TENSOR_ITER_STEP(grads[0], indices);
    }
  }
}

static void
MSE_Loss_forward(struct lida_Loss* self, const struct lida_Tensor* pred, const struct lida_Tensor* target)
{
  if (compare_tensor_shapes(pred, target) != 0) {
    LOG_ERROR("MSE loss: tensors must be the same shape");
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
static struct lida_Gate g_mm_gate;

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
  if (g_mul_gate.name == NULL) {
    g_mul_gate = (struct lida_Gate) {
      .name = "*",
      .forward = &mul_gate_forward,
      .backward = &mul_gate_backward,
      .num_args = 2
    };
  }
  return &g_mul_gate;
}

const struct lida_Gate*
lida_gate_mm()
{
  if (g_mm_gate.name == NULL) {
    g_mm_gate = (struct lida_Gate) {
      .name = "mm",
      .forward = &mm_gate_forward,
      .backward = &mm_gate_backward,
      .num_args = 2
    };
  }
  return &g_mm_gate;
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
  if (g_sigmoid_gate.name == NULL) {
    g_sigmoid_gate = (struct lida_Gate) {
      .name = "sigmoid",
      .forward = &sigmoid_gate_forward,
      .backward = &sigmoid_gate_backward,
      .num_args = 1
    };
  }
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
