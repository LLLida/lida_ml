/*
  My small machine learning framework.
 */
#include "lida_ml.h"

#include "string.h"

#define ARR_SIZE(arr) sizeof(arr) / sizeof(arr[0])

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

struct Input {
  const char* name;
  uint32_t dims[LIDA_MAX_DIMENSIONALITY];
  int32_t rank;
  struct lida_Tensor* tensor;
};

struct Compute_Node {
  struct lida_Gate* gate;
  struct Compute_Node* parents[4];
  int32_t num_args;
  struct lida_Tensor* grad;
  int frozen;
};

struct lida_Compute_Graph {
  struct Compute_Node* inputs[32];
  struct Compute_Node* outputs[32];
  int32_t num_inputs;
  int32_t num_outputs;

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
POOL_DEF(Compute_Node, 128);

static struct lida_ML g_ml;
POOL_DECL(Allocation);
POOL_DECL(lida_Tensor);
POOL_DECL(Compute_Node);

#define LOG_DEBUG(...) g_ml.log(0, __VA_ARGS__)
#define LOG_INFO(...)  g_ml.log(1, __VA_ARGS__)
#define LOG_WARN(...)  g_ml.log(2, __VA_ARGS__)
#define LOG_ERROR(...) g_ml.log(3, __VA_ARGS__)


/// static functions

ADD_POOL_DECL(Allocation)
ADD_POOL_DECL(lida_Tensor)
ADD_POOL_DECL(Compute_Node)
ALLOCATE_DECL(Allocation)
ALLOCATE_DECL(lida_Tensor)
ALLOCATE_DECL(Compute_Node)

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


/// library functions

void
lida_ml_init(const struct lida_ML* ml)
{
  memcpy(&g_ml, ml, sizeof(struct lida_ML));
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
  cg->num_inputs = 0;
  cg->num_outputs = 0;
  cg->requires_grad = requires_grad;
  return cg;
}

void
lida_compute_graph_destroy(struct lida_Compute_Graph* cg)
{
  g_ml.dealloc(cg);
}

int
lida_compute_graph_add_input(struct lida_Compute_Graph* cg, const char* name, const uint32_t dims[], int rank)
{
  if (cg->num_inputs >= ARR_SIZE(cg->inputs)) {
    LOG_ERROR("max number of inputs exceeded");
    return -1;
  }
}

int
lida_compute_graph_add_parameter(struct lida_Compute_Graph* cg, lida_Tensor* parameter, int frozen)
{

}

int
lida_compute_graph_add_child(struct lida_Compute_Graph* cg, struct lida_Compute_Graph* child)
{

}
