/*
  My small machine learning framework.
 */
#include "lida_ml.h"

#include "math.h"
#include "string.h"

#define ARR_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

struct Dim {
  uint32_t num;
  uint32_t pitch;
  int32_t index;
  uint32_t _padding;
};

struct Allocation {
  void* ptr;
  size_t refs;
};

struct lida_Tensor {
  lida_Format format;
  int32_t rank;
  struct Dim dims[LIDA_MAX_DIMENSIONALITY];
  void* cpu_mem;
  // alloc == NULL means cpu_mem points to external memory
  struct Allocation* alloc;
};

enum {
  NODE_INPUT,
  NODE_PARAMETER,
  NODE_GATE,
  NODE_GRAPH,
};

struct Node_Input {
  const char* name;
  uint32_t dims[LIDA_MAX_DIMENSIONALITY];
  int32_t rank;
  const struct lida_Tensor* tensor;
};

struct Node_Parameter {
  struct lida_Tensor* value;
  struct lida_Tensor* grad;
};

struct Node_Gate {
  const struct lida_Gate* gate;
  struct lida_Tensor* output;
  struct lida_Tensor* grad;
  size_t first_id;
  struct Node_Gate* left;
  struct Node_Gate* right;
};

struct Compute_Node {
  union {
    struct Node_Input input;
    struct Node_Parameter param;
    struct Node_Gate gate;
    struct lida_Compute_Graph* graph;
  } u;
  int type;
};

struct Compute_Node_Arena {
  struct Compute_Node* data;
  size_t size;
  size_t offset;
  struct Compute_Node_Arena* next;
};

struct lida_Compute_Graph {
  struct Compute_Node_Arena* first_arena;
  struct Compute_Node_Arena* last_arena;

  struct Compute_Node* inputs[32];
  struct Compute_Node* outputs[32];
  size_t num_inputs;
  size_t num_outputs;
  size_t node_counter;
  struct Node_Gate* first_layer;
  struct Node_Gate* last_layer;

  int requires_grad;
};

#define tdim(tensor, dim) ((tensor)->dims[(tensor)->dims[dim].index])

#define POOL_DEF(type, num) struct type##_Pool {	\
    struct type data[num];				\
    uint32_t num_free;					\
    uint32_t free_index;				\
    struct type##_Pool* next;				\
  }
#define POOL_NAME(type) g_##type##_pools
#define POOL_DECL(type) static struct type##_Pool* POOL_NAME(type) = NULL
#define ADD_POOL_NAME(type) add_##type##_pool
#define ADD_POOL_DECL(type) static void					\
  ADD_POOL_NAME(type) ()						\
  {									\
    struct type##_Pool* pool = g_ml.alloc(sizeof(struct type##_Pool));	\
    pool->num_free = ARR_SIZE(pool->data);				\
    pool->free_index = 0;						\
    pool->next = POOL_NAME(type);					\
    for (size_t i = 0; i < ARR_SIZE(pool->data); i++) {			\
      uint32_t* index = (uint32_t*)&pool->data[i];			\
      *index = i+1;							\
    }									\
    POOL_NAME(type) = pool;						\
  }
#define ALLOCATE_DECL(type) static struct type*			\
  allocate_##type ()						\
  {								\
    struct type##_Pool* pool = POOL_NAME(type);		\
    while (pool != NULL && pool->num_free > 0) {	\
      pool = pool->next;				\
    }								\
    if (pool == NULL || pool->num_free == 0) {	\
      ADD_POOL_NAME(type)();					\
      pool = POOL_NAME(type);				\
    }								\
    struct type* ret = &pool->data[pool->free_index];	\
    uint32_t* index = (uint32_t*)ret;				\
    pool->free_index = *index;					\
    pool->num_free--;						\
    return ret;							\
  }
#define CLEAR_POOLS(type)						\
  while (POOL_NAME(type)) {					\
    if (POOL_NAME(type)->num_free != ARR_SIZE(POOL_NAME(type)->data)) { \
      do_warning = 1;							\
    }									\
    struct type##_Pool* next = POOL_NAME(type)->next;	\
    g_ml.dealloc(POOL_NAME(type));				\
    POOL_NAME(type) = next;					\
  }

POOL_DEF(Allocation, 512);
POOL_DEF(lida_Tensor, 1024);

static struct lida_ML g_ml;
POOL_DECL(Allocation);
POOL_DECL(lida_Tensor);

static struct lida_Gate g_plus_gate;
static struct lida_Gate g_mul_gate;

static struct lida_Gate g_relu_gate;
static struct lida_Gate g_sigmoid_gate;
static struct lida_Gate g_tanh_gate;

#define LOG_DEBUG(...) g_ml.log(0, __VA_ARGS__)
#define LOG_INFO(...)  g_ml.log(1, __VA_ARGS__)
#define LOG_WARN(...)  g_ml.log(2, __VA_ARGS__)
#define LOG_ERROR(...) g_ml.log(3, __VA_ARGS__)


/// static functions

ADD_POOL_DECL(Allocation)
ADD_POOL_DECL(lida_Tensor)
ALLOCATE_DECL(Allocation)
ALLOCATE_DECL(lida_Tensor)

static uint32_t
format_num_bytes(lida_Format format)
{
  switch (format & LIDA_FORMAT_MASK)
    {
    case LIDA_FORMAT_U16:
    case LIDA_FORMAT_I16:
    case LIDA_FORMAT_F16:
      return 2;
    case LIDA_FORMAT_U32:
    case LIDA_FORMAT_I32:
    case LIDA_FORMAT_F32:
      return 4;
    default:
      // error maybe?
      return 0;
    }
}

static void
release_tensor(struct lida_Tensor* tensor)
{
  // find parent pool
  struct lida_Tensor_Pool* prev = NULL;
  struct lida_Tensor_Pool* pool = POOL_NAME(lida_Tensor);
  while (tensor < pool->data || tensor >= pool->data+ARR_SIZE(pool->data)) {
    prev = pool;
    pool = pool->next;
  }

  uint32_t* index = (uint32_t*)tensor;
  *index = pool->free_index;
  pool->free_index = tensor - pool->data;
  pool->num_free++;

  // release entire pool
  if (pool->num_free == ARR_SIZE(pool->data)) {
    if (POOL_NAME(lida_Tensor) == pool) {
      POOL_NAME(lida_Tensor) = pool->next;
    } else {
      prev->next = pool->next;
    }
    g_ml.dealloc(pool);
  }
}

static struct Allocation*
do_allocation(uint32_t bytes)
{
  struct Allocation* ret = allocate_Allocation();
  // do actual allocation
  ret->ptr = g_ml.alloc(bytes);
  ret->refs = 1;

  return ret;
}

static void
free_allocation(struct Allocation* alloc)
{
  g_ml.dealloc(alloc->ptr);

  // find parent pool
  struct Allocation_Pool* prev = NULL;
  struct Allocation_Pool* pool = POOL_NAME(Allocation);
  while (alloc < pool->data || alloc >= pool->data+ARR_SIZE(pool->data)) {
    prev = pool;
    pool = pool->next;
  }

  uint32_t* index = (uint32_t*)alloc;
  *index = pool->free_index;
  pool->free_index = alloc - pool->data;
  pool->num_free++;

  // release entire pool
  if (pool->num_free == ARR_SIZE(pool->data)) {
    if (POOL_NAME(Allocation) == pool) {
      POOL_NAME(Allocation) = pool->next;
    } else {
      prev->next = pool->next;
    }
    g_ml.dealloc(pool);
  }
}

// python's % operator
static int32_t
python_mod(int32_t a, int32_t b)
{
  return (b + (a % b)) % b;
}

static int32_t
seq_rank(const struct lida_Tensor* tensor)
{
  int32_t r = 1;
  for (int32_t i = 0; i < tensor->rank-1; i++) {
    if (tensor->dims[i].num == tensor->dims[i].pitch) {
      r++;
    } else {
      break;
    }
  }
  return r;
}

static int32_t
seq_full_rank(const struct lida_Tensor* tensor)
{
  int32_t r = 0;
  for (int32_t i = 0; i < tensor->rank; i++) {
    if (tensor->dims[i].index == i && tensor->dims[i].num == tensor->dims[i].pitch) {
      r++;
    } else {
      break;
    }
  }
  return r;
}

static uint32_t
tensor_offset(const struct lida_Tensor* tensor, const uint32_t indices[])
{
  uint32_t offset = 0;
  for (int i = (int)tensor->rank-1; i >= 0; i--) {
    offset *= tensor->dims[i].pitch;
    offset += indices[tensor->dims[i].index];
  }
  return offset * format_num_bytes(tensor->format);
}

static struct lida_Tensor*
tensor_copy(struct lida_Tensor* tensor)
{
  struct lida_Tensor* ret = allocate_lida_Tensor();
  memcpy(ret, tensor, sizeof(struct lida_Tensor));
  if (tensor->alloc)
    tensor->alloc->refs++;
  return ret;
}

static const struct lida_Tensor*
tensor_const_copy(const struct lida_Tensor* tensor)
{
  struct lida_Tensor* ret = allocate_lida_Tensor();
  memcpy(ret, tensor, sizeof(struct lida_Tensor));
  if (tensor->alloc)
    tensor->alloc->refs++;
  return ret;
}

static int
compare_tensor_shapes(const struct lida_Tensor* a, const struct lida_Tensor* b)
{
  if (a->rank != b->rank) {
    return (a->rank > b->rank) - (a->rank < b->rank);
  }
  for (int32_t i = 0; i < a->rank; i++) {
    uint32_t x = tdim(a, i).num;
    uint32_t y = tdim(b, i).num;
    if (x != y) {
      return (x > y) - (x < y);
    }
  }
  return 0;
}

static struct Compute_Node_Arena*
allocate_compute_node_arena(size_t size)
{
  void* ptr = g_ml.alloc(size * sizeof(struct Compute_Node) + sizeof(struct Compute_Node_Arena));
  struct Compute_Node_Arena* ret = (void*)((uint8_t*)ptr + size * sizeof(struct Compute_Node));
  ret->data = ptr;
  ret->size = size;
  ret->offset = 0;
  ret->next = NULL;
  return ret;
}

static struct Compute_Node*
allocate_compute_node(struct lida_Compute_Graph* cg)
{
  if (cg->last_arena->offset > cg->last_arena->size) {
    struct Compute_Node_Arena* a = allocate_compute_node_arena(64);
    cg->last_arena->next = a;
    cg->last_arena = a;
  }
  struct Compute_Node* ret = cg->last_arena->data + cg->last_arena->offset;
  cg->last_arena->offset += 1;
  cg->node_counter++;
  return ret;
}

static struct Compute_Node*
get_node_by_id(struct lida_Compute_Graph* cg, size_t id)
{
  struct Compute_Node_Arena* arena = cg->first_arena;
  while (arena) {
    if (id < arena->offset) {
      return &arena->data[id];
    }
    id -= arena->offset;
    arena = arena->next;
  }
  return NULL;
}

static const struct lida_Tensor*
get_node_tensor(struct Compute_Node* node)
{
  if (node->type == NODE_INPUT) {
    return node->u.input.tensor;
  } else if (node->type == NODE_PARAMETER) {
    return node->u.param.value;
  } else {
    return node->u.gate.output;
  }
}

static void
compute_node_clear_output(struct lida_Compute_Graph* cg, struct Compute_Node* node)
{
  if (node->type == NODE_INPUT || node->type == NODE_PARAMETER)
    return;

  struct Node_Gate* gate = &node->u.gate;
  if (gate->output) {
    lida_tensor_destroy(gate->output);
    gate->output = NULL;
  }
  if (gate->grad) {
    lida_tensor_destroy(gate->grad);
    gate->grad = NULL;
  }
  for (size_t i = 0; i < gate->gate->num_args; i++) {
    compute_node_clear_output(cg, get_node_by_id(cg, gate->first_id + i));
  }
}

static void
eval_compute_node(struct lida_Compute_Graph* cg, struct Compute_Node* node)
{
  if (node->type == NODE_INPUT || node->type == NODE_PARAMETER)
    return;

  struct Node_Gate* gate = &node->u.gate;
  if (gate->output)
    return;
  const struct lida_Tensor* outputs[4];
  for (size_t i = 0; i < gate->gate->num_args; i++) {
    struct Compute_Node* arg = get_node_by_id(cg, gate->first_id + i);
    eval_compute_node(cg, arg);
    outputs[i] = get_node_tensor(arg);
  }
  gate->output = gate->gate->forward(gate->gate->udata, outputs);
}

static void
destroy_compute_node(struct Compute_Node* node)
{
  switch (node->type)
    {
    case NODE_INPUT:
      // do nothing, cg doesn't own input tensors
      break;
    case NODE_PARAMETER:
      // cg doesn't own parameter's value
      if (node->u.param.grad)
	lida_tensor_destroy(node->u.param.grad);
      break;
    case NODE_GATE:
      lida_tensor_destroy(node->u.gate.output);
      if (node->u.gate.grad)
	lida_tensor_destroy(node->u.gate.grad);
      break;
    }
}

#define PERFORM_ELEMENTWISE_OP2(a, b, c, type, op) {			\
  uint32_t indices[LIDA_MAX_DIMENSIONALITY] = {0};			\
  while (indices[a->rank-1] < a->dims[a->rank-1].num) {	\
    type* va = lida_tensor_get_unchecked(a, indices);			\
    type* vb = lida_tensor_get_unchecked(b, indices);			\
    type* vc = lida_tensor_get_unchecked(c, indices);			\
    *vc = *va op *vb;							\
    for (int32_t i = 0; i < a->rank; i++) {			\
      indices[i]++;							\
      if (indices[i] == a->dims[i].num && i < a->rank-1) {	\
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
    LOG_ERROR("+: tensors must be the same shape");
    return NULL;
  }
  if (a->format != b->format) {
    LOG_ERROR("+: tensors must have the same format");
  }

  uint32_t dims[LIDA_MAX_DIMENSIONALITY];
  lida_tensor_get_dims(a, dims, NULL);
  struct lida_Tensor* c = lida_tensor_create(dims, a->rank, a->format);

  switch (a->format)
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
      LOG_ERROR("+ on this format is not supported");
      lida_tensor_destroy(c);
      return NULL;
    }
  return c;
}

static void
plus_gate_backward(void* udata, const struct lida_Tensor* output, struct lida_Tensor** args)
{
  (void)udata;
  (void)output;
  (void)args;
}

static void
MSE_Loss_forward(struct lida_Loss* self, const struct lida_Tensor* pred, const struct lida_Tensor* actual)
{
  if (compare_tensor_shapes(pred, actual) != 0) {
    LOG_ERROR("MSE loss: tensors must be the same shape");
    return;
  }
  self->value = 0.0;
  self->pred = pred;
  self->actual = actual;
  uint32_t indices[LIDA_MAX_DIMENSIONALITY];
  while (indices[pred->rank-1] < tdim(pred, pred->rank-1).num) {
    float* y1 = lida_tensor_get_unchecked(pred, indices);
    float* y2 = lida_tensor_get_unchecked(actual, indices);
    float d = y1-y2;
    self->value += d*d;
    for (int i = 0; i < pred->rank; i++) {
      indices[i]++;
      if (indices[i] == tdim(pred, i).num) {
	if (i != pred->rank-1)
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

  struct lida_Tensor* grad = lida_tensor_create(dims, rank, self->pred->format);
  uint32_t indices[LIDA_MAX_DIMENSIONALITY];
  while (indices[rank-1] < tdim(self->pred, rank-1).num) {
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



/// library functions

void
lida_ml_init(const struct lida_ML* ml)
{
  memcpy(&g_ml, ml, sizeof(struct lida_ML));

  g_plus_gate = (struct lida_Gate) {
    .name = "+",
    .forward = &plus_gate_forward,
    .backward = &plus_gate_backward,
    .num_args = 2
  };
}

void
lida_ml_done()
{
  int do_warning = 0;
  CLEAR_POOLS(lida_Tensor)
  CLEAR_POOLS(Allocation)
  if (do_warning) {
    LOG_WARN("not all tensors were destroyed");
  }
}

struct lida_Tensor*
lida_tensor_create(const uint32_t dims[], int rank, lida_Format format)
{
  if (rank == 0) {
    LOG_ERROR("can't create a tensor with rank = 0");
    return NULL;
  }
  if (rank > LIDA_MAX_DIMENSIONALITY) {
    LOG_ERROR("can't create a tensor with rank higher than %d", LIDA_MAX_DIMENSIONALITY);
    return NULL;
  }

  struct lida_Tensor* ret = allocate_lida_Tensor();
  ret->format = format;
  ret->rank = rank;
  uint32_t size = 1;
  for (int i = 0; i < rank; i++) {
    ret->dims[i] = (struct Dim) {
      .num = dims[i],
      .pitch = dims[i],
      .index = i
    };
    size *= dims[i];
  }
  ret->alloc = do_allocation(size * format_num_bytes(format));
  ret->cpu_mem = ret->alloc->ptr;

  return ret;
}

void
lida_tensor_destroy(struct lida_Tensor* tensor)
{
  if (tensor->alloc) {
    if (tensor->alloc->refs == 1) {
      free_allocation(tensor->alloc);
    } else {
      tensor->alloc->refs--;
    }
  }
  release_tensor(tensor);
}

struct lida_Tensor*
lida_tensor_create_from_memory(void* memory, uint32_t bytes, const uint32_t dims[], int rank, lida_Format format)
{
  if (rank == 0) {
    LOG_ERROR("can't create a tensor with rank = 0");
    return NULL;
  }
  if (rank > LIDA_MAX_DIMENSIONALITY) {
    LOG_ERROR("can't create a tensor with rank higher than %d", LIDA_MAX_DIMENSIONALITY);
    return NULL;
  }
  uint32_t s = format_num_bytes(format);
  for (int i = 0; i < rank; i++) {
    s *= dims[i];
  }
  if (bytes != s) {
    LOG_ERROR("bytes and dims mismatch(%u != %u)", bytes, s);
    return NULL;
  }
  struct lida_Tensor* ret = allocate_lida_Tensor();
  ret->format = format;
  ret->rank = rank;
  uint32_t size = 1;
  for (int i = 0; i < rank; i++) {
    ret->dims[i] = (struct Dim) {
      .num = dims[i],
      .pitch = dims[i],
      .index = i
    };
    size *= dims[i];
  }
  ret->alloc = NULL;
  ret->cpu_mem = memory;

  return ret;
}

lida_Format
lida_tensor_get_format(const struct lida_Tensor* tensor)
{
  return tensor->format;
}

void
lida_tensor_get_dims(const struct lida_Tensor* tensor, uint32_t* dims, int* rank)
{
  if (dims) {
    for (int32_t i = 0; i < tensor->rank; i++) {
      dims[i] = tdim(tensor, i).num;
    }
  }
  if (rank) {
    *rank = (int)tensor->rank;
  }
}

void*
lida_tensor_get(struct lida_Tensor* tensor, const uint32_t indices[], int num_indices)
{
  if (num_indices != (int)tensor->rank) {
    LOG_ERROR("num_indices(which is %d) doesn't match tensor's rank(which is %u)",
	      num_indices, tensor->rank);
    return NULL;
  }
  for (int i = 0; i < num_indices; i++) {
    if (indices[i] >= tdim(tensor, i).num) {
      LOG_ERROR("index out of bounds: indices[%u] > %u", indices[i], tensor->dims[i].num);
      return NULL;
    }
  }

  return lida_tensor_get_unchecked(tensor, indices);
}

void*
lida_tensor_get_unchecked(const struct lida_Tensor* tensor, const uint32_t indices[])
{
  uint32_t offset = tensor_offset(tensor, indices);
  uint8_t* bytes = tensor->cpu_mem;
  return bytes + offset;
}

uint32_t
lida_tensor_size(const struct lida_Tensor* tensor)
{
  uint32_t s = 1;
  for (int32_t i = 0; i < tensor->rank; i++) {
    s *= tensor->dims[i].num;
  }
  return s;
}

void
lida_tensor_fill_zeros(struct lida_Tensor* tensor)
{
  uint32_t indices[LIDA_MAX_DIMENSIONALITY] = {0};
  uint32_t bytes_per_elem = format_num_bytes(tensor->format);
  uint8_t* bytes = tensor->cpu_mem;
  // NOTE: we don't use the tdim macro in here because the order of
  // dimensions doesn't matter

  int32_t seq = seq_rank(tensor);
  uint32_t mag = bytes_per_elem;
  for (int32_t i = 0; i < seq; i++) {
    mag *= tensor->dims[i].num;
  }
  if (seq == tensor->rank) {
    memset(bytes, 0, mag);
    return;
  }

  while (indices[tensor->rank-1] < tensor->dims[tensor->rank-1].num) {
    uint32_t offset = 0;
    for (int32_t i = tensor->rank-1; i >= seq; i--) {
      offset *= tensor->dims[i].pitch;
      offset += indices[i];
    }
    offset *= mag;
    memset(&bytes[offset], 0, mag);

    for (int32_t i = seq; i < tensor->rank; i++) {
      indices[i]++;
      if (indices[i] == tensor->dims[i].num && i < tensor->rank-1) {
	indices[i] = 0;
      } else {
	break;
      }
    }
  }
}

void
lida_tensor_fill(struct lida_Tensor* tensor, const void* obj)
{
  uint32_t indices[LIDA_MAX_DIMENSIONALITY] = {0};
  uint32_t bytes_per_elem = format_num_bytes(tensor->format);
  uint8_t* bytes = tensor->cpu_mem;
  // NOTE: we don't use the tdim macro in here because the order of
  // dimensions doesn't matter

  while (indices[tensor->rank-1] < tensor->dims[tensor->rank-1].num) {
    // we fill values one by one. This is very slow, it'd be better to
    // do that with intrinsics or smth.
    uint32_t offset = 0;
    for (int32_t i = tensor->rank-1; i >= 0; i--) {
      offset *= tensor->dims[i].pitch;
      offset += indices[i];
    }
    offset *= bytes_per_elem;
    memcpy(&bytes[offset], obj, bytes_per_elem);

    for (int32_t i = 0; i < tensor->rank; i++) {
      indices[i]++;
      if (indices[i] == tensor->dims[i].num && i < tensor->rank-1) {
	indices[i] = 0;
      } else {
	break;
      }
    }
  }
}

struct lida_Tensor*
lida_tensor_transpose(struct lida_Tensor* tensor, const uint32_t dims[], int rank)
{
  if ((int)tensor->rank != rank) {
    LOG_ERROR("array of invalid size(got %d) passed: expected %u", rank, tensor->rank);
    return NULL;
  }
  for (int i = 0; i < rank; i++) {
    if (dims[i] >= (uint32_t)rank) {
      LOG_ERROR("dims out of bounds: %u >= %d", dims[i], rank);
      return NULL;
    }
  }
  // TODO: check for duplicates in dims

  struct lida_Tensor* ret = tensor_copy(tensor);
  for (int i = 0; i < rank; i++) {
    ret->dims[tensor->dims[i].index].index = dims[i];
  }
  return ret;
}

struct lida_Tensor*
lida_tensor_slice(struct lida_Tensor* tensor, const uint32_t left[], const uint32_t right[], int rank)
{
  if ((int)tensor->rank != rank) {
    LOG_ERROR("array of invalid size(got %d) passed: expected %u", rank, tensor->rank);
    return NULL;
  }
  for (int i = 0; i < rank; i++) {
    if (left[i] >= right[i]) {
      LOG_ERROR("slice in dimension [%d] has size non-positive size", i);
      return NULL;
    }
    if (right[i] > tdim(tensor, i).num) {
      LOG_ERROR("slice in dimension [%d] is out of bounds (it should be < %u)",
		i, tdim(tensor, i).num);
      return NULL;
    }
  }

  struct lida_Tensor* ret = tensor_copy(tensor);
  ret->cpu_mem = (uint8_t*)ret->cpu_mem + tensor_offset(tensor, left);
  for (int i = 0; i < rank; i++) {
    tdim(ret, i).num = right[i] - left[i];
  }
  return ret;
}

struct lida_Tensor*
lida_tensor_deep_copy(struct lida_Tensor* tensor)
{
  uint32_t dims[LIDA_MAX_DIMENSIONALITY];
  lida_tensor_get_dims(tensor, dims, NULL);
  struct lida_Tensor* ret = lida_tensor_create(dims, tensor->rank, tensor->format);

  int32_t seq = seq_full_rank(tensor);
  uint32_t mag = format_num_bytes(tensor->format);
  for (int32_t i = 0; i < seq; i++) {
    mag *= tdim(tensor, i).pitch;
  }
  if (seq == tensor->rank) {
    memcpy(ret->cpu_mem, tensor->cpu_mem, lida_tensor_size(ret) * format_num_bytes(tensor->format));
  } else {
    uint32_t indices[LIDA_MAX_DIMENSIONALITY] = {0};
    uint8_t* dst = ret->cpu_mem;
    uint8_t* src = tensor->cpu_mem;
    while (indices[tensor->rank-1] < tdim(tensor, tensor->rank-1).num) {
      uint32_t o1 = 0, o2 = 0;
      for (int32_t i = tensor->rank-1; i >= seq; i--) {
	o1 *= tensor->dims[i].pitch;
	o1 += indices[tensor->dims[i].index];
	o2 *= ret->dims[i].pitch;
	o2 += indices[i];
      }
      o1 *= mag;
      o2 *= mag;
      memcpy(&dst[o2], &src[o1], mag);

      for (int32_t i = seq; i < tensor->rank; i++) {
	indices[i]++;
	if (indices[i] == tdim(tensor, i).num && i < tensor->rank-1) {
	  indices[i] = 0;
	} else {
	  break;
	}
      }
    }
  }

  return ret;
}

struct lida_Tensor*
lida_tensor_reshape(struct lida_Tensor* tensor, const uint32_t dims[], int rank)
{
  if (rank > LIDA_MAX_DIMENSIONALITY) {
    LOG_ERROR("can't create a tensor with rank higher than %d", LIDA_MAX_DIMENSIONALITY);
    return NULL;
  }
  uint32_t s = 1;
  for (int i = 0; i < rank; i++) {
    s *= dims[i];
  }
  if (s != lida_tensor_size(tensor)) {
    LOG_ERROR("dimensionality mismatch(%u != %u)", s, lida_tensor_size(tensor));
    return NULL;
  }

  struct lida_Tensor* ret;

  if (tensor->rank != seq_full_rank(tensor)) {
    ret = lida_tensor_deep_copy(tensor);
  } else {
    ret = tensor_copy(tensor);
  }

  ret->rank = rank;
  for (int i = 0; i < rank; i++) {
    ret->dims[i] = (struct Dim) {
      .num = dims[i],
      .pitch = dims[i],
      .index = i
    };
  }
  return ret;
}

struct lida_Tensor*
lida_tensor_flip(struct lida_Tensor* tensor, const uint32_t axes[], int num_axes)
{
  if (num_axes == 0) {
    LOG_ERROR("can't pass num_axes=0");
    return NULL;
  }
  for (int i = 0; i < num_axes; i++) {
    if (axes[i] >= (uint32_t)tensor->rank) {
      LOG_ERROR("axes[%d] > tensor rank (%u > %d)", i, axes[i], tensor->rank);
      return NULL;
    }
  }
  struct lida_Tensor* ret = lida_tensor_deep_copy(tensor);
  uint8_t buff[16];
  uint32_t bytes_per_elem = format_num_bytes(tensor->format);
  uint8_t* bytes = ret->cpu_mem;

  for (int i = 0; i < num_axes; i++) {
    uint32_t ax = axes[i];

    if (ret->rank == 1) {
      // reverse an array
      for (uint32_t coord = 0; coord < tdim(ret, 0).num/2; coord++) {
	uint32_t a = coord*bytes_per_elem;
	uint32_t b = (tdim(ret, 0).num - coord - 1)*bytes_per_elem;
	memcpy(buff, &bytes[a],      bytes_per_elem);
	memcpy(&bytes[a], &bytes[b], bytes_per_elem);
	memcpy(&bytes[b], buff,      bytes_per_elem);
      }
    } else {
      uint32_t indices[LIDA_MAX_DIMENSIONALITY];
      for (int i = 0; i < ret->rank; i++) {
	indices[i] = i;
      }
      indices[ax] = 0;
      indices[0] = ax;

      struct lida_Tensor* temp = lida_tensor_transpose(ret, indices, ret->rank);
      for (uint32_t coord = 0; coord < tdim(temp, 0).num/2; coord++) {
	uint32_t indices[LIDA_MAX_DIMENSIONALITY] = {0};

	while (indices[temp->rank-1] < tdim(temp, temp->rank-1).num) {
	  // calculate offsets
	  indices[0] = coord;
	  uint32_t a = tensor_offset(temp, indices);
	  indices[0] = tdim(temp, 0).num - coord - 1;
	  uint32_t b = tensor_offset(temp, indices);
	  memcpy(buff, &bytes[a],      bytes_per_elem);
	  memcpy(&bytes[a], &bytes[b], bytes_per_elem);
	  memcpy(&bytes[b], buff,      bytes_per_elem);

	  for (int32_t i = 1; i < temp->rank; i++) {
	    indices[i]++;
	    if (indices[i] == tdim(temp, i).num && i < tensor->rank-1) {
	      indices[i] = 0;
	    } else {
	      break;
	    }
	  }
	}
      }
      lida_tensor_destroy(temp);
    }
  }
  return ret;
}

struct lida_Tensor*
lida_tensor_rot90(struct lida_Tensor* tensor, uint32_t ax1, uint32_t ax2, int n)
{
  if (ax1 >= (uint32_t)tensor->rank || ax2 >= (uint32_t)tensor->rank) {
    LOG_ERROR("axis is not smaller than tensor's rank");
    return NULL;
  }
  if (ax1 == ax2) {
    LOG_ERROR("rotation axes must not match");
    return NULL;
  }
  if (tdim(tensor, ax1).num != tdim(tensor, ax2).num) {
    LOG_ERROR("dimensions of rotation axes must match");
    return NULL;
  }

  n = python_mod(n, 4);
  if (n == 0) {
    return tensor_copy(tensor);
  }

  uint32_t bytes_per_elem = format_num_bytes(tensor->format);

  struct lida_Tensor* ret = lida_tensor_deep_copy(tensor);
  if (tensor->rank == 2) {
    uint32_t dst[2];
    uint32_t src[2];
    for (uint32_t i = 0; i < tdim(ret, ax1).num; i++) {
      for (uint32_t j = 0; j < tdim(ret, ax2).num; j++) {
	src[ax1] = i;
	src[ax2] = j;
	switch (n)
	  {
	  case 1:
	    dst[ax1] = j;
	    dst[ax2] = tdim(tensor, ax1).num - i - 1;
	    break;
	  case 2:
	    dst[ax1] = tdim(tensor, ax1).num - i - 1;
	    dst[ax2] = tdim(tensor, ax2).num - j - 1;
	    break;
	  case 3:
	    dst[ax1] = tdim(tensor, ax2).num - j - 1;
	    dst[ax2] = i;
	    break;
	  }
	memcpy(lida_tensor_get(ret, dst, 2), lida_tensor_get(tensor, src, 2), bytes_per_elem);
      }
    }
  } else {
    LOG_WARN("rotation for tensors with rank > 2 not implemented");
    return NULL;
  }

  return ret;
}

struct lida_Compute_Graph*
lida_compute_graph_create(int requires_grad)
{
  struct lida_Compute_Graph* cg = g_ml.alloc(sizeof(struct lida_Compute_Graph));
  cg->first_arena = allocate_compute_node_arena(64);
  cg->last_arena = cg->first_arena;
  cg->num_inputs = 0;
  cg->num_outputs = 0;
  cg->first_layer = NULL;
  cg->node_counter = 0;
  cg->requires_grad = requires_grad;
  return cg;
}

void
lida_compute_graph_destroy(struct lida_Compute_Graph* cg)
{
  struct Compute_Node_Arena* arena = cg->first_arena;
  while (arena) {
    for (size_t i = 0; i < arena->offset; i++) {
      struct Compute_Node* node = &arena->data[i];
      destroy_compute_node(node);
    }
    struct Compute_Node_Arena* next = arena->next;
    g_ml.dealloc(arena->data);
    arena = next;
  }
  g_ml.dealloc(cg);
}

int
lida_compute_graph_add_input(struct lida_Compute_Graph* cg, const char* name, const uint32_t dims[], int rank)
{
  if (cg->num_inputs >= (int32_t)ARR_SIZE(cg->inputs)) {
    LOG_ERROR("max number of inputs exceeded");
    return -1;
  }
  struct Compute_Node* node = allocate_compute_node(cg);
  node->type = NODE_INPUT;
  node->u.input = (struct Node_Input) {
    .name = name,
    .rank = rank,
    .tensor = NULL,
  };
  memcpy(node->u.input.dims, dims, rank * sizeof(uint32_t));

  cg->inputs[cg->num_inputs++] = node;
  cg->outputs[cg->num_outputs++] = node;

  return 0;
}

int
lida_compute_graph_add_parameter(struct lida_Compute_Graph* cg, struct lida_Tensor* parameter, int frozen)
{
  (void)frozen;
  if (cg->num_outputs >= (int32_t)ARR_SIZE(cg->outputs)) {
    LOG_ERROR("max number of outputs exceeded");
    return -1;
  }

  struct Compute_Node* node = allocate_compute_node(cg);
  node->type = NODE_PARAMETER;
  node->u.param = (struct Node_Parameter) {
    .value = parameter,
    .grad = NULL,
    // .frozen = frozen
  };
  cg->outputs[cg->num_outputs++] = node;

  return 0;
}

int
lida_compute_graph_add_gate(struct lida_Compute_Graph* cg, const struct lida_Gate* gate)
{
  if (cg->num_outputs >= (int32_t)ARR_SIZE(cg->outputs)) {
    LOG_ERROR("max number of outputs exceeded");
    return -1;
  }
  if (cg->num_outputs != gate->num_args) {
    LOG_ERROR("argument count mismatch");
    return -1;
  }

  size_t first_id = cg->node_counter - cg->num_outputs;

  struct Compute_Node* node = allocate_compute_node(cg);
  node->type = NODE_GATE;
  node->u.gate = (struct Node_Gate) {
    .gate = gate,
    .output = NULL,
    .grad = NULL,
    .first_id = first_id
  };

  cg->num_outputs = 1;
  cg->outputs[0] = node;

  if (cg->first_layer == NULL) {
    cg->first_layer = &node->u.gate;
    cg->last_layer = cg->first_layer;
  } else {
    cg->last_layer->right = &node->u.gate;
    cg->last_layer->right->left = cg->last_layer;
    cg->last_layer = cg->last_layer->right;
  }

  return 0;
}

int
lida_compute_graph_add_child(struct lida_Compute_Graph* cg, struct lida_Compute_Graph* child)
{
  if (cg->num_inputs + child->num_inputs > (int32_t)ARR_SIZE(cg->inputs)) {
    LOG_ERROR("max number of inputs exceeded");
    return -1;
  }
  for (size_t i = 0; i < child->num_inputs; i++) {
    cg->inputs[cg->num_inputs++] = child->inputs[i];
  }
  // FIXME: how do I connect? I think I need to do additional stuff
  return 0;
}

int
lida_compute_graph_set_input(struct lida_Compute_Graph* cg, const char* name, const struct lida_Tensor* tensor)
{
  for (size_t i = 0; i < cg->num_inputs; i++) {
    struct Node_Input* input = &cg->inputs[i]->u.input;
    if (strcmp(name, input->name)) {
      if (input->rank != tensor->rank) {
	LOG_ERROR("input tensor rank mismatch(%d != %d)", input->rank, tensor->rank);
	return -1;
      }
      for (int32_t j = 0; j < tensor->rank; j++) {
	if (input->dims[j] != tdim(tensor, j).num) {
	  LOG_ERROR("input tensor dimension mismatch(%u != %u)", input->dims[j], tdim(tensor, j).num);
	  return -1;
	}
      }
      input->tensor = tensor;
      return 0;
    }
  }
  LOG_ERROR("no input with name '%s' found", name);
  return -1;
}

void
lida_compute_graph_forward(struct lida_Compute_Graph* cg)
{
  for (size_t i = 0; i < cg->num_outputs; i++) {
    compute_node_clear_output(cg, cg->outputs[i]);
  }
  for (size_t i = 0; i < cg->num_outputs; i++) {
    eval_compute_node(cg, cg->outputs[i]);
  }
}

void
lida_compute_graph_backward(struct lida_Compute_Graph* cg, struct lida_Loss* losses, int count)
{
  if (count != (int)cg->num_outputs) {
    LOG_ERROR("number of losses doesn't match number of outputs in compute graph");
    return;
  }
  for (size_t i = 0; i < cg->num_outputs; i++) {
    if (cg->outputs[i]->type == NODE_INPUT) {
      LOG_ERROR("input parameter can't be output");
    } else if (cg->outputs[i]->type == NODE_PARAMETER) {
      LOG_WARN("parameter at output of compute graph");
      cg->outputs[i]->u.param.grad = losses[i].backward(&losses[i]);
    } else {
      cg->outputs[i]->u.gate.grad = losses[i].backward(&losses[i]);
    }
  }
}

const struct lida_Tensor*
lida_compute_graph_get_output(struct lida_Compute_Graph* cg, size_t index)
{
  if (index >= cg->num_outputs) {
    LOG_ERROR("index is out of bounds(%d >= %d)", index, cg->num_outputs);
    return NULL;
  }

  return tensor_const_copy(get_node_tensor(cg->outputs[index]));
}

const struct lida_Tensor*
lida_compute_graph_get_output_grad(struct lida_Compute_Graph* cg, size_t index)
{
  if (index >= cg->num_outputs) {
    LOG_ERROR("index is out of bounds(%d >= %d)", index, cg->num_outputs);
    return NULL;
  }

  struct lida_Tensor* ret;
  if (cg->outputs[index]->type == NODE_PARAMETER) {
    return cg->outputs[index]->u.param.grad;
  } else {
    return cg->outputs[index]->u.gate.grad;
  }
  return tensor_copy(ret);
}

const struct lida_Gate*
lida_gate_plus()
{
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
