#ifndef LIDA_ML_HPP
#define LIDA_ML_HPP

#include <cmath>
#include <stdexcept>
#include <span>

#include "lida_ml.h"
#include "lida_ml_std.h"

#define LIDA_ML_NOEXCEPT noexcept

namespace lida {

  class ML_Library {

    inline static bool did_init;

    ML_Library() { did_init = false; }

  public:

    [[nodiscard]]
    static ML_Library init(const lida_ML& init_info) {
      static ML_Library lib;
      if (!did_init) {
	lida_ml_init(&init_info);
	did_init = true;
      }
      return lib;
    }

    ~ML_Library() {
      lida_ml_done();
      did_init = false;
    }
  };

  using Format = lida_Format;
  using Gate = lida_Gate;

  template<typename T>
  static Format to_format() LIDA_ML_NOEXCEPT {
    if constexpr (std::is_same_v<T, float>) {
      return LIDA_FORMAT_F32;
    } else if constexpr (std::is_same_v<T, int32_t>) {
      return LIDA_FORMAT_I32;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      return LIDA_FORMAT_U32;
    }
    // TODO: support for other formats
    return LIDA_FORMAT_MASK;
  }

  class Tensor {

    struct lida_Tensor* raw;

    template<typename T>
    static bool check_type(Format format) LIDA_ML_NOEXCEPT {
      if constexpr (std::is_same_v<T, float>) {
	if (format == LIDA_FORMAT_F32)
	  return true;
      } else if constexpr (std::is_same_v<T, int32_t>) {
	if (format == LIDA_FORMAT_I32)
	  return true;
      } else if constexpr (std::is_same_v<T, uint32_t>) {
	if (format == LIDA_FORMAT_U32)
	  return true;
      }
      return false;
    }

  public:

    Tensor(struct lida_Tensor* handle) LIDA_ML_NOEXCEPT {
      raw = handle;
    }

    Tensor(std::span<uint32_t> dims, Format format) LIDA_ML_NOEXCEPT {
      raw = lida_tensor_create(dims.data(), dims.size(), format);
    }

    template<typename T, std::size_t E>
    Tensor(std::span<T, E> external, std::span<uint32_t> dims) LIDA_ML_NOEXCEPT {
      raw = lida_tensor_create_from_memory(external.data(), external.size_bytes(), dims.data(), dims.size(), to_format<T>());
    }

    Tensor(const Tensor& other) = delete;

    Tensor(Tensor&& other) LIDA_ML_NOEXCEPT {
      raw = other.raw;
      other.raw = NULL;
    }

    ~Tensor() LIDA_ML_NOEXCEPT {
      if (raw)
	lida_tensor_destroy(raw);
    }

    Tensor& operator=(Tensor&& other) LIDA_ML_NOEXCEPT {
      if (raw)
	lida_tensor_destroy(raw);
      raw = other.raw;
      other.raw = NULL;
      return *this;
    }

    [[nodiscard]]
    struct lida_Tensor* handle() LIDA_ML_NOEXCEPT {
      return raw;
    }

    [[nodiscard]]
    const struct lida_Tensor* handle() const LIDA_ML_NOEXCEPT {
      return raw;
    }

    [[nodiscard]]
    Format format() const LIDA_ML_NOEXCEPT {
      return lida_tensor_get_format(raw);
    }

    [[nodiscard]]
    int rank() const LIDA_ML_NOEXCEPT {
      int r;
      lida_tensor_get_dims(raw, NULL, &r);
      return r;
    }

    void dims(std::span<uint32_t> dims) const {
      if (dims.size() != rank()) {
	throw std::invalid_argument("lida::Tensor::dims: dims has invalid size");
      }
      lida_tensor_get_dims(raw, dims.data(), NULL);
    }

    [[nodiscard]]
    void* get(std::span<const uint32_t> indices) LIDA_ML_NOEXCEPT {
      return lida_tensor_get(raw, indices.data(), indices.size());
    }

    [[nodiscard]]
    const void* get(std::span<const uint32_t> indices) const LIDA_ML_NOEXCEPT {
      return lida_tensor_get(raw, indices.data(), indices.size());
    }

    void fill_zeros() LIDA_ML_NOEXCEPT {
      lida_tensor_fill_zeros(raw);
    }

    template<typename T>
    void fill(T value) {
      if (!check_type<T>(format())) {
	throw std::logic_error("lida::Tensor::fill: invalid type");
      }
      lida_tensor_fill(raw, &value);
    }

    void fill_uniform(float left = 0.0, float right = 1.0) LIDA_ML_NOEXCEPT {
      lida_tensor_fill_uniform(raw, left, right);
    }

    void fill_normal(float mu = 0.0, float sigma = 1.0) LIDA_ML_NOEXCEPT {
      lida_tensor_fill_normal(raw, mu, sigma);
    }

    [[nodiscard]]
    Tensor transpose(std::span<const uint32_t> dims) LIDA_ML_NOEXCEPT {
      return lida_tensor_transpose(raw, dims.data(), dims.size());
    }

    [[nodiscard]]
    Tensor slice(std::span<const uint32_t> left, std::span<const uint32_t> right) {
      if (left.size() != right.size()) {
	throw std::invalid_argument("lida::Tensor::slice: left and right have different sizes");
      }
      return lida_tensor_slice(raw, left.data(), right.data(), left.size());
    }

    [[nodiscard]]
    Tensor deep_copy() LIDA_ML_NOEXCEPT {
      return lida_tensor_deep_copy(raw);
    }

    [[nodiscard]]
    Tensor reshape(std::span<const uint32_t> dims) LIDA_ML_NOEXCEPT {
      return lida_tensor_reshape(raw, dims.data(), dims.size());
    }

    [[nodiscard]]
    Tensor flip(std::span<const uint32_t> axes) LIDA_ML_NOEXCEPT {
      return lida_tensor_flip(raw, axes.data(), axes.size());
    }

    [[nodiscard]]
    Tensor rot90(uint32_t ax1, uint32_t ax2, int n = 1) LIDA_ML_NOEXCEPT {
      return lida_tensor_rot90(raw, ax1, ax2, n);
    }

    void add(const Tensor& other, float scalar = 1.0) {
      if (lida_tensor_add(raw, other.raw, scalar) != 0) {
	throw std::runtime_error("failed to add tensors");
      }
    }

    [[nodiscard]]
    static Tensor stack(std::span<const Tensor> tensors) LIDA_ML_NOEXCEPT {
      return lida_tensor_stack((const lida_Tensor**)tensors.data(), tensors.size());
    }

  };

  static_assert(sizeof(Tensor) == sizeof(lida_Tensor*));

  class Loss {

    lida_Loss raw;

  public:

    [[nodiscard]]
    float value() const {
      return raw.value;
    }

    [[nodiscard]]
    static auto MSE(const Tensor& pred, const Tensor& y) LIDA_ML_NOEXCEPT {
      Loss loss;
      lida_MSE_loss(&loss.raw);
      loss.raw.forward(&loss.raw, pred.handle(), y.handle());
      return loss;
    }

    [[nodiscard]]
    static auto cross_entropy(const Tensor& pred, const Tensor& y) LIDA_ML_NOEXCEPT {
      Loss loss;
      lida_Cross_Entropy_Loss(&loss.raw);
      loss.raw.forward(&loss.raw, pred.handle(), y.handle());
      return loss;
    }

  };

  class Basic_Optimizer {

    lida_Optimizer raw;

    static void step_wrapper(lida_Optimizer* self, struct lida_Tensor* param, const struct lida_Tensor* grad) LIDA_ML_NOEXCEPT {
      auto obj = (Basic_Optimizer*)self->udata;
      obj->step(*(Tensor*)&param, *(const Tensor*)&grad);
    }

  protected:
    virtual void step(Tensor& param, const Tensor& grad) = 0;

    Basic_Optimizer() LIDA_ML_NOEXCEPT {
      raw.udata = (void*)this;
      raw.step = step_wrapper;
    }

  public:
    Basic_Optimizer(const Basic_Optimizer& other) = delete;
    Basic_Optimizer(Basic_Optimizer&& other) = delete;

    [[nodiscard]]
    auto handle() LIDA_ML_NOEXCEPT {
      return &raw;
    }

  };

  class Compute_Graph {

    struct lida_Compute_Graph* raw;

    Compute_Graph(struct lida_Compute_Graph* handle) LIDA_ML_NOEXCEPT {
      raw = handle;
    }

  public:

    Compute_Graph(bool requires_grad = true) LIDA_ML_NOEXCEPT {
      raw = lida_compute_graph_create(requires_grad);
    }

    Compute_Graph(const Compute_Graph& other) = delete;

    Compute_Graph(Compute_Graph&& other) LIDA_ML_NOEXCEPT {
      raw = other.raw;
      other.raw = NULL;
    }

    ~Compute_Graph() LIDA_ML_NOEXCEPT {
      lida_compute_graph_destroy(raw);
    }

    Compute_Graph& operator=(Compute_Graph&& other) LIDA_ML_NOEXCEPT {
      if (raw)
	lida_compute_graph_destroy(raw);
      raw = other.raw;
      other.raw = NULL;
      return *this;
    }

    [[nodiscard]]
    lida_Compute_Graph* handle() LIDA_ML_NOEXCEPT {
      return raw;
    }

    Compute_Graph& add_input(const char* name, std::span<uint32_t> shape) {
      if (lida_compute_graph_add_input(raw, name, shape.data(), shape.size()) != 0) {
	throw std::runtime_error("TODO: failed");
      }
      return *this;
    }

    Compute_Graph& add_parameter(Tensor& tensor, bool frozen = false) {
      if (lida_compute_graph_add_parameter(raw, tensor.handle(), frozen) != 0) {
	throw std::runtime_error("TODO: failed");
      }
      return *this;
    }

    Compute_Graph& add_gate(const Gate* gate) {
      if (lida_compute_graph_add_gate(raw, gate) != 0) {
	throw std::runtime_error("TODO: failed");
      }
      return *this;
    }

    Compute_Graph& add_child(Compute_Graph&& child) {
      if (lida_compute_graph_add_child(raw, child.raw) != 0) {
	throw std::runtime_error("TODO: failed");
      }
      return *this;
    }

    template<typename T>
    Compute_Graph& add_layer(T& layer) {
      layer.bind(*this);
      return *this;
    }

    Compute_Graph& set_input(const char* name, const lida::Tensor& tensor) {
      if (lida_compute_graph_set_input(raw, name, tensor.handle()) != 0) {
	throw std::runtime_error("TODO: failed");
      }
      return *this;
    }

    void forward() LIDA_ML_NOEXCEPT {
      lida_compute_graph_forward(raw);
    }

    Compute_Graph& zero_grad() LIDA_ML_NOEXCEPT {
      lida_compute_graph_zero_grad(raw);
      return *this;
    }

    Compute_Graph& backward(std::span<Loss> losses) LIDA_ML_NOEXCEPT {
      lida_compute_graph_backward(raw, (lida_Loss*)losses.data(), losses.size());
      return *this;
    }

    Compute_Graph& backward(Loss& loss) LIDA_ML_NOEXCEPT {
      lida_compute_graph_backward(raw, (lida_Loss*)&loss, 1);
      return *this;
    }

    [[nodiscard]]
    lida::Tensor get_output(size_t index) {
      const struct lida_Tensor* handle = lida_compute_graph_get_output(raw, index);
      if (handle == NULL) {
	throw std::runtime_error("TODO: failed");
      }
      return (struct lida_Tensor*)handle;
    }

    Compute_Graph& optimizer_step(Basic_Optimizer& other) {
      lida_compute_graph_optimizer_step(raw, other.handle());
      return *this;
    }

  };

  class SGD_Optimizer : public Basic_Optimizer {

    float lr;

  public:

    SGD_Optimizer(float lr) {
      this->lr = lr;
    }

    void step(Tensor& param, const Tensor& grad) override {
      // param += grad * lr
      param.add(grad, -lr);
    }

  };

  inline auto plus() LIDA_ML_NOEXCEPT {
    return lida_gate_plus();
  }

  inline auto mul() LIDA_ML_NOEXCEPT {
    return lida_gate_mul();
  }

  inline auto mm() LIDA_ML_NOEXCEPT {
    return lida_gate_mm();
  }

  inline auto relu() LIDA_ML_NOEXCEPT {
    return lida_gate_relu();
  }

  inline auto sigmoid() LIDA_ML_NOEXCEPT {
    return lida_gate_sigmoid();
  }

  inline auto tanh() LIDA_ML_NOEXCEPT {
    return lida_gate_tanh();
  }

  inline void rand_seed(uint64_t seed) LIDA_ML_NOEXCEPT {
    lida_rand_seed(seed);
  }

  inline auto rand() LIDA_ML_NOEXCEPT {
    return lida_rand();
  }

  inline auto rand_uniform() LIDA_ML_NOEXCEPT {
    return lida_rand_uniform();
  }

  inline auto rand_normal() LIDA_ML_NOEXCEPT {
    return lida_rand_normal();
  }

  class Linear_Layer {
  public:
    // weight matrix
    lida::Tensor w;
    // bias vector
    lida::Tensor b;

    Linear_Layer(uint32_t num_inputs, uint32_t num_outputs)
      // we have to use this ugly construct because std::span doesn't accept r-values
      : w(std::span{std::array{num_inputs, num_outputs}.data(), 2}, LIDA_FORMAT_F32),
	b(std::span{&num_outputs, 1}, LIDA_FORMAT_F32) {
      // Xavier initialization
      float var = sqrtf(2.0 / (num_inputs + num_outputs));
      w.fill_normal(0.0, var);

      b.fill_uniform(-1.0, 1.0);
    }

    void bind(Compute_Graph& cg) {
      cg.add_parameter(w)
	.add_gate(mm())
	.add_parameter(b)
	.add_gate(plus());
    }

  };

}

#endif // LIDA_ML_HPP
