#ifndef LIDA_ML_HPP
#define LIDA_ML_HPP

#include <stdexcept>
#include <span>

#include "lida_ml.h"

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

  };

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

    Compute_Graph& set_input(const char* name, const lida::Tensor& tensor) {
      if (lida_compute_graph_set_input(raw, name, tensor.handle()) != 0) {
	throw std::runtime_error("TODO: failed");
      }
      return *this;
    }

    void forward() LIDA_ML_NOEXCEPT {
      lida_compute_graph_forward(raw);
    }

    void backward(std::span<Loss> losses) LIDA_ML_NOEXCEPT {
      lida_compute_graph_backward(raw, (lida_Loss*)losses.data(), losses.size());
    }

    void backward(Loss& loss) LIDA_ML_NOEXCEPT {
      lida_compute_graph_backward(raw, (lida_Loss*)&loss, 1);
    }

    lida::Tensor get_output(size_t index) {
      const struct lida_Tensor* handle = lida_compute_graph_get_output(raw, index);
      if (handle == NULL) {
	throw std::runtime_error("TODO: failed");
      }
      return (struct lida_Tensor*)handle;
    }

    lida::Tensor get_output_grad(size_t index) {
      const struct lida_Tensor* handle = lida_compute_graph_get_output_grad(raw, index);
      if (handle == NULL) {
	throw std::runtime_error("TODO: failed");
      }
      return (struct lida_Tensor*)handle;
    }

  };

  inline auto plus() LIDA_ML_NOEXCEPT {
    return lida_gate_plus();
  }

  inline auto mul() LIDA_ML_NOEXCEPT {
    return lida_gate_mul();
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

}

#endif // LIDA_ML_HPP
