#ifndef LIDA_ML_HPP
#define LIDA_ML_HPP

#include <stdexcept>
#include <span>

#include "lida_ml.h"

#define LIDA_ML_NOEXCEPT noexcept

namespace lida {

  class Tensor {

    struct lida_Tensor* raw;

    Tensor(struct lida_Tensor* handle) LIDA_ML_NOEXCEPT {
      raw = handle;
    }

  public:

    Tensor(std::span<uint32_t> dims, lida_Format format) LIDA_ML_NOEXCEPT {
      raw = lida_tensor_create(dims.data(), dims.size(), format);
    }

    Tensor(const Tensor& other) = delete;

    Tensor(Tensor&& other) LIDA_ML_NOEXCEPT {
      raw = other.raw;
      other.raw = NULL;
    }

    ~Tensor() LIDA_ML_NOEXCEPT {
      // TODO: free tensor
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
    void* get(std::span<uint32_t> indices) LIDA_ML_NOEXCEPT {
      return lida_tensor_get(raw, indices.data(), indices.size());
    }

    [[nodiscard]]
    const void* get(std::span<uint32_t> indices) const LIDA_ML_NOEXCEPT {
      return lida_tensor_get(raw, indices.data(), indices.size());
    }

    void fill_zeros() LIDA_ML_NOEXCEPT {
      lida_tensor_fill_zeros(raw);
    }

    [[nodiscard]]
    Tensor transpose(std::span<uint32_t> dims) LIDA_ML_NOEXCEPT {
      return lida_tensor_transpose(raw, dims.data(), dims.size());
    }

    [[nodiscard]]
    Tensor slice(std::span<uint32_t> left, std::span<uint32_t> right) {
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
    Tensor reshape(std::span<uint32_t> dims) LIDA_ML_NOEXCEPT {
      return lida_tensor_reshape(raw, dims.data(), dims.size());
    }

  };

}

#endif // LIDA_ML_HPP
