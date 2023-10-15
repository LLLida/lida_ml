/*
  My small machine learning framework.
 */
#include "lida_ml.h"

#include "string.h"

#define LIDA_GC_MASK 0xffff

#define ARR_SIZE(arr) sizeof(arr) / sizeof(arr[0])

struct Dim {
  uint32_t num;
  uint32_t pitch;
  uint32_t index;
  uint32_t _padding;
};

struct Allocation {
  void* ptr;
  size_t refs;
};

struct lida_Tensor {
  lida_Format format;
  uint32_t rank;
  struct Dim dims[LIDA_MAX_DIMENSIONALITY];
  void* cpu_mem;
  // alloc == NULL means cpu_mem points to external memory
  struct Allocation* alloc;
};

#define tdim(tensor, dim) ((tensor)->dims[(tensor)->dims[dim].index])

struct Allocation_Pool {
  struct Allocation data[512];
  uint32_t num_free;
  uint32_t free_index;
  struct Allocation_Pool* next;
};

struct Tensor_Pool {
  struct lida_Tensor data[1024];
  uint32_t num_free;
  uint32_t free_index;
  struct Tensor_Pool* next;
};

static struct lida_ML g_ml;
static struct Tensor_Pool* g_tpools = NULL;
static struct Allocation_Pool* g_apools = NULL;

#define LOG_DEBUG(...) g_ml.log(0, __VA_ARGS__)
#define LOG_INFO(...)  g_ml.log(1, __VA_ARGS__)
#define LOG_WARN(...)  g_ml.log(2, __VA_ARGS__)
#define LOG_ERROR(...) g_ml.log(3, __VA_ARGS__)


/// static functions

static void
add_tensor_pool()
{
  struct Tensor_Pool* pool = g_ml.alloc(sizeof(struct Tensor_Pool));
  pool->num_free = ARR_SIZE(pool->data);
  pool->free_index = 0;
  pool->next = g_tpools;
  for (size_t i = 0; i < ARR_SIZE(pool->data); i++) {
    uint32_t* index = (uint32_t*)&pool->data[i];
    *index = i+1;
  }

  g_tpools = pool;
}

static void
add_alloc_pool()
{
  struct Allocation_Pool* pool = g_ml.alloc(sizeof(struct Allocation_Pool));
  pool->num_free = ARR_SIZE(pool->data);
  pool->free_index = 0;
  pool->next = g_apools;
  for (size_t i = 0; i < ARR_SIZE(pool->data); i++) {
    uint32_t* index = (uint32_t*)&pool->data[i];
    *index = i+1;
  }

  g_apools = pool;
}

static uint32_t
format_num_bytes(lida_Format format)
{
  switch (format)
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

static struct lida_Tensor*
allocate_tensor()
{
  // find suitable pool
  struct Tensor_Pool* pool = g_tpools;
  while (pool != NULL && pool->num_free > 0) {
    pool = pool->next;
  }
  if (pool == NULL || pool->num_free == 0) {
    add_tensor_pool();
    pool = g_tpools;
  }

  // take free object
  struct lida_Tensor* ret = &pool->data[pool->free_index];
  uint32_t* index = (uint32_t*)ret;
  pool->free_index = *index;
  pool->num_free--;
  return ret;
}

static void
release_tensor(struct lida_Tensor* tensor)
{
  // find parent pool
  struct Tensor_Pool* prev = NULL;
  struct Tensor_Pool* pool = g_tpools;
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
    if (g_tpools == pool) {
      g_tpools = pool->next;
    } else {
      prev->next = pool->next;
    }
    g_ml.dealloc(pool);
  }
}

static struct Allocation*
do_allocation(uint32_t bytes)
{
  // find suitable pool
  struct Allocation_Pool* pool = g_apools;
  while (pool != NULL && pool->num_free > 0) {
    pool = pool->next;
  }
  if (pool == NULL || pool->num_free == 0) {
    add_alloc_pool();
    pool = g_apools;
  }

  // take free object
  struct Allocation* ret = &pool->data[pool->free_index];
  uint32_t* index = (uint32_t*)ret;
  pool->free_index = *index;
  pool->num_free--;

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
  struct Allocation_Pool* pool = g_apools;
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
    if (g_apools == pool) {
      g_apools = pool->next;
    } else {
      prev->next = pool->next;
    }
    g_ml.dealloc(pool);
  }
}

static uint32_t
seq_rank(const struct lida_Tensor* tensor)
{
  uint32_t r = 1;
  for (uint32_t i = 0; i < tensor->rank-1; i++) {
    if (tensor->dims[i].num == tensor->dims[i].pitch) {
      r++;
    } else {
      break;
    }
  }
  return r;
}

static uint32_t
seq_full_rank(const struct lida_Tensor* tensor)
{
  uint32_t r = 0;
  for (uint32_t i = 0; i < tensor->rank; i++) {
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
  struct lida_Tensor* ret = allocate_tensor();
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
  while (g_tpools) {
    if (g_tpools->num_free != ARR_SIZE(g_tpools->data)) {
      do_warning = 1;
    }
    struct Tensor_Pool* next = g_tpools->next;
    g_ml.dealloc(g_tpools);
    g_tpools = next;
  }
  while (g_apools) {
    struct Allocation_Pool* next = g_apools->next;
    g_ml.dealloc(g_apools);
    g_apools = next;
  }
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

  struct lida_Tensor* ret = allocate_tensor();
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
  struct lida_Tensor* ret = allocate_tensor();
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

void
lida_tensor_get_dims(const struct lida_Tensor* tensor, uint32_t* dims, int* rank)
{
  if (dims) {
    for (uint32_t i = 0; i < tensor->rank; i++) {
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
  for (uint32_t i = 0; i < tensor->rank; i++) {
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

  uint32_t seq = seq_rank(tensor);
  uint32_t mag = bytes_per_elem;
  for (uint32_t i = 0; i < seq; i++) {
    mag *= tensor->dims[i].pitch;
  }
  if (seq == tensor->rank) {
    memset(bytes, 0, mag);
    return;
  }

  while (indices[tensor->rank-1] < tensor->dims[tensor->rank-1].num) {
    uint32_t offset = 0;
    for (uint32_t i = tensor->rank-1; i >= seq; i--) {
      offset *= tensor->dims[i].pitch;
      offset += indices[i];
    }
    offset *= mag;
    memset(&bytes[offset], 0, mag);

    for (uint32_t i = seq; i < tensor->rank; i++) {
      indices[i]++;
      if (indices[i] == tensor->dims[i].num && i < tensor->rank-1) {
	indices[i] = 0;
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
  struct lida_Tensor* ret = tensor_copy(tensor);
  if (tensor->alloc)
    tensor->alloc->refs--;		// tensor_copy increments refs, we don't need that
  uint32_t bytes_per_elem = format_num_bytes(tensor->format);

  uint32_t s = bytes_per_elem * lida_tensor_size(tensor);
  for (uint32_t i = 0; i < tensor->rank; i++) {
    ret->dims[i] = (struct Dim) {
      .index	= i,
      .num	= tdim(tensor, i).num,
      .pitch	= tdim(tensor, i).num
    };
  }
  ret->alloc = do_allocation(s);
  ret->cpu_mem = ret->alloc->ptr;

  uint32_t seq = seq_full_rank(tensor);
  uint32_t mag = bytes_per_elem;
  for (uint32_t i = 0; i < seq; i++) {
    mag *= tdim(tensor, i).pitch;
  }
  if (seq == tensor->rank) {
    memcpy(ret->cpu_mem, tensor->cpu_mem, s);
  } else {
    uint32_t indices[LIDA_MAX_DIMENSIONALITY] = {0};
    uint8_t* dst = ret->cpu_mem;
    uint8_t* src = tensor->cpu_mem;
    while (indices[tensor->rank-1] < tdim(tensor, tensor->rank-1).num) {
      uint32_t o1 = 0, o2 = 0;
      for (uint32_t ii = tensor->rank; ii > seq; ii--) {
	uint32_t i = ii-1;
	o1 *= tensor->dims[i].pitch;
	o1 += indices[tensor->dims[i].index];
	o2 *= ret->dims[i].pitch;
	o2 += indices[i];
      }
      o1 *= mag;
      o2 *= mag;
      memcpy(&dst[o2], &src[o1], mag);

      for (uint32_t i = seq; i < tensor->rank; i++) {
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
