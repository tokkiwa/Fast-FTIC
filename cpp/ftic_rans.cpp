#include <torch/extension.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "rans64.h"

namespace {
constexpr int kPrecision = 16;
constexpr uint16_t kBypassPrecision = 4;
constexpr uint16_t kMaxBypassVal = (1 << kBypassPrecision) - 1;

inline void Rans64EncPutBits(Rans64State *r, uint32_t **pptr, uint32_t val,
                             uint32_t nbits) {
  assert(nbits <= 16);
  assert(val < (1u << nbits));

  uint64_t x = *r;
  uint32_t freq = 1 << (16 - nbits);
  uint64_t x_max = ((RANS64_L >> 16) << 32) * freq;
  if (x >= x_max) {
    *pptr -= 1;
    **pptr = (uint32_t)x;
    x >>= 32;
    Rans64Assert(x < x_max);
  }

  *r = (x << nbits) | val;
}

inline uint32_t Rans64DecGetBits(Rans64State *r, uint32_t **pptr,
                                 uint32_t n_bits) {
  uint64_t x = *r;
  uint32_t val = x & ((1u << n_bits) - 1);

  x = x >> n_bits;
  if (x < RANS64_L) {
    x = (x << 32) | **pptr;
    *pptr += 1;
    Rans64Assert(x >= RANS64_L);
  }

  *r = x;

  return val;
}

inline float standardized_cumulative(float x) {
  constexpr float half = 0.5f;
  constexpr float constv = -0.7071067811865475f; // -1/sqrt(2)
  return half * std::erfc(constv * x);
}

void build_cdf_for_symbol(float mean, float scale, int minmax,
                          std::vector<int32_t> &cdf_out) {
  const int S = 2 * minmax + 1;
  const float scale_min = 0.01f;
  const float s = std::max(scale, scale_min);
  const float min_prob = 1.0f / 65536.0f;

  std::vector<float> pmf(S);
  float sum = 0.0f;
  for (int k = 0; k < S; ++k) {
    const float value = std::fabs(static_cast<float>(k) - mean);
    const float upper = standardized_cumulative((0.5f - value) / s);
    const float lower = standardized_cumulative((-0.5f - value) / s);
    float p = upper - lower;
    if (p < min_prob) {
      p = min_prob;
    }
    pmf[k] = p;
    sum += p;
  }

  cdf_out.resize(S + 1);
  cdf_out[0] = 0;

  std::vector<int32_t> pmf_q(S);
  int32_t cum = 0;
  for (int k = 0; k < S; ++k) {
    int32_t v = static_cast<int32_t>(std::llround(pmf[k] / sum * 65536.0f));
    if (v < 1) {
      v = 1;
    }
    pmf_q[k] = v;
    cum += v;
  }

  int32_t diff = 65536 - cum;
  if (diff != 0) {
    std::vector<int> idx(S);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return pmf_q[a] > pmf_q[b]; });

    if (diff > 0) {
      for (int i = 0; i < diff; ++i) {
        pmf_q[idx[i % S]] += 1;
      }
    } else {
      int need = -diff;
      int i = 0;
      while (need > 0 && i < S * 2) {
        int k = idx[i % S];
        if (pmf_q[k] > 1) {
          pmf_q[k] -= 1;
          --need;
        }
        ++i;
      }
    }
  }

  cum = 0;
  for (int k = 0; k < S; ++k) {
    cum += pmf_q[k];
    cdf_out[k + 1] = cum;
  }
}

struct RansSymbol {
  uint16_t start;
  uint16_t range;
  bool bypass;
};

class BufferedRansEncoder {
 public:
  void encode_symbol_ptr(int32_t symbol, const int32_t *cdf,
                         int32_t cdf_size, int32_t offset) {
    const int32_t max_value = cdf_size - 2;
    int32_t value = symbol - offset;

    uint32_t raw_val = 0;
    if (value < 0) {
      raw_val = -2 * value - 1;
      value = max_value;
    } else if (value >= max_value) {
      raw_val = 2 * (value - max_value);
      value = max_value;
    }

    _syms.push_back({static_cast<uint16_t>(cdf[value]),
                     static_cast<uint16_t>(cdf[value + 1] - cdf[value]),
                     false});

    if (value == max_value) {
      int32_t n_bypass = 0;
      while ((raw_val >> (n_bypass * kBypassPrecision)) != 0) {
        ++n_bypass;
      }

      int32_t val = n_bypass;
      while (val >= kMaxBypassVal) {
        _syms.push_back({kMaxBypassVal, kMaxBypassVal + 1, true});
        val -= kMaxBypassVal;
      }
      _syms.push_back({static_cast<uint16_t>(val),
                       static_cast<uint16_t>(val + 1), true});

      for (int32_t j = 0; j < n_bypass; ++j) {
        const int32_t v = (raw_val >> (j * kBypassPrecision)) & kMaxBypassVal;
        _syms.push_back(
            {static_cast<uint16_t>(v), static_cast<uint16_t>(v + 1), true});
      }
    }
  }

  std::string flush() {
    Rans64State rans;
    Rans64EncInit(&rans);

    std::vector<uint32_t> output(_syms.size(), 0xCC);
    uint32_t *ptr = output.data() + output.size();

    while (!_syms.empty()) {
      const RansSymbol sym = _syms.back();
      if (!sym.bypass) {
        Rans64EncPut(&rans, &ptr, sym.start, sym.range, kPrecision);
      } else {
        Rans64EncPutBits(&rans, &ptr, sym.start, kBypassPrecision);
      }
      _syms.pop_back();
    }

    Rans64EncFlush(&rans, &ptr);

    const int nbytes =
        std::distance(ptr, output.data() + output.size()) * sizeof(uint32_t);
    return std::string(reinterpret_cast<char *>(ptr), nbytes);
  }

 private:
  std::vector<RansSymbol> _syms;
};

} // namespace

py::bytes encode_with_cdfs_tensor(const torch::Tensor &symbols,
                                  const torch::Tensor &cdfs) {
  TORCH_CHECK(symbols.device().is_cpu(), "symbols must be a CPU tensor");
  TORCH_CHECK(cdfs.device().is_cpu(), "cdfs must be a CPU tensor");
  TORCH_CHECK(symbols.scalar_type() == torch::kInt32,
              "symbols must be int32");
  TORCH_CHECK(cdfs.scalar_type() == torch::kInt32, "cdfs must be int32");
  TORCH_CHECK(cdfs.dim() == 2, "cdfs must be 2D");

  auto sym = symbols.contiguous();
  auto cdf = cdfs.contiguous();

  const int64_t n = sym.numel();
  TORCH_CHECK(cdf.size(0) == n, "cdfs first dim must match symbols");
  const int64_t cdf_size = cdf.size(1);
  TORCH_CHECK(cdf_size >= 2, "invalid cdf size");

  const auto *sym_ptr = sym.data_ptr<int32_t>();
  const auto *cdf_ptr = cdf.data_ptr<int32_t>();

  BufferedRansEncoder encoder;

  for (int64_t i = 0; i < n; ++i) {
    const int32_t *row = cdf_ptr + i * cdf_size;
    const int32_t max_value = static_cast<int32_t>(cdf_size - 2);
    int32_t value = sym_ptr[i];

    encoder.encode_symbol_ptr(value, row, static_cast<int32_t>(cdf_size), 0);
  }

  return py::bytes(encoder.flush());
}

torch::Tensor decode_with_cdfs_tensor(const std::string &encoded,
                                      const torch::Tensor &cdfs) {
  TORCH_CHECK(cdfs.device().is_cpu(), "cdfs must be a CPU tensor");
  TORCH_CHECK(cdfs.scalar_type() == torch::kInt32, "cdfs must be int32");
  TORCH_CHECK(cdfs.dim() == 2, "cdfs must be 2D");

  auto cdf = cdfs.contiguous();
  const int64_t n = cdf.size(0);
  const int64_t cdf_size = cdf.size(1);
  TORCH_CHECK(cdf_size >= 2, "invalid cdf size");

  const auto *cdf_ptr = cdf.data_ptr<int32_t>();

  auto output = torch::empty({n}, torch::TensorOptions().dtype(torch::kInt32));
  auto *out_ptr = output.data_ptr<int32_t>();

  Rans64State rans;
  uint32_t *ptr = (uint32_t *)encoded.data();
  if (!ptr) {
    throw std::runtime_error("invalid encoded buffer");
  }
  Rans64DecInit(&rans, &ptr);

  for (int64_t i = 0; i < n; ++i) {
    const int32_t *row = cdf_ptr + i * cdf_size;
    const int32_t max_value = static_cast<int32_t>(cdf_size - 2);

    const uint32_t cum_freq = Rans64DecGet(&rans, kPrecision);
    const auto it = std::lower_bound(
        row, row + cdf_size, static_cast<int32_t>(cum_freq + 1));
    const int32_t s = static_cast<int32_t>(std::distance(row, it) - 1);

    Rans64DecAdvance(&rans, &ptr, row[s], row[s + 1] - row[s], kPrecision);

    int32_t value = s;
    if (value == max_value) {
      int32_t val = Rans64DecGetBits(&rans, &ptr, kBypassPrecision);
      int32_t n_bypass = val;
      while (val == kMaxBypassVal) {
        val = Rans64DecGetBits(&rans, &ptr, kBypassPrecision);
        n_bypass += val;
      }
      int32_t raw_val = 0;
      for (int j = 0; j < n_bypass; ++j) {
        val = Rans64DecGetBits(&rans, &ptr, kBypassPrecision);
        raw_val |= val << (j * kBypassPrecision);
      }
      value = raw_val >> 1;
      if (raw_val & 1) {
        value = -value - 1;
      } else {
        value += max_value;
      }
    }

    out_ptr[i] = value;
  }

  return output;
}

py::bytes encode_with_gaussian_tensor(const torch::Tensor &symbols,
                                      const torch::Tensor &means,
                                      const torch::Tensor &scales,
                                      int64_t minmax) {
  TORCH_CHECK(symbols.device().is_cpu(), "symbols must be a CPU tensor");
  TORCH_CHECK(means.device().is_cpu(), "means must be a CPU tensor");
  TORCH_CHECK(scales.device().is_cpu(), "scales must be a CPU tensor");
  TORCH_CHECK(symbols.scalar_type() == torch::kInt32,
              "symbols must be int32");
  TORCH_CHECK(means.scalar_type() == torch::kFloat32,
              "means must be float32");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat32,
              "scales must be float32");

  auto sym = symbols.contiguous();
  auto mu = means.contiguous();
  auto sc = scales.contiguous();

  TORCH_CHECK(sym.numel() == mu.numel(), "symbols/means size mismatch");
  TORCH_CHECK(sym.numel() == sc.numel(), "symbols/scales size mismatch");

  const auto *sym_ptr = sym.data_ptr<int32_t>();
  const auto *mu_ptr = mu.data_ptr<float>();
  const auto *sc_ptr = sc.data_ptr<float>();
  const int64_t n = sym.numel();

  BufferedRansEncoder encoder;
  std::vector<int32_t> cdf;

  for (int64_t i = 0; i < n; ++i) {
    build_cdf_for_symbol(mu_ptr[i] + static_cast<float>(minmax), sc_ptr[i],
                         static_cast<int>(minmax), cdf);
    encoder.encode_symbol_ptr(sym_ptr[i], cdf.data(),
                              static_cast<int32_t>(cdf.size()), 0);
  }

  return py::bytes(encoder.flush());
}

torch::Tensor decode_with_gaussian_tensor(const std::string &encoded,
                                          const torch::Tensor &means,
                                          const torch::Tensor &scales,
                                          int64_t minmax) {
  TORCH_CHECK(means.device().is_cpu(), "means must be a CPU tensor");
  TORCH_CHECK(scales.device().is_cpu(), "scales must be a CPU tensor");
  TORCH_CHECK(means.scalar_type() == torch::kFloat32,
              "means must be float32");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat32,
              "scales must be float32");

  auto mu = means.contiguous();
  auto sc = scales.contiguous();

  TORCH_CHECK(mu.numel() == sc.numel(), "means/scales size mismatch");

  const auto *mu_ptr = mu.data_ptr<float>();
  const auto *sc_ptr = sc.data_ptr<float>();
  const int64_t n = mu.numel();

  auto output = torch::empty({n}, torch::TensorOptions().dtype(torch::kInt32));
  auto *out_ptr = output.data_ptr<int32_t>();
  std::vector<int32_t> cdf;

  const int S = 2 * static_cast<int>(minmax) + 1;
  const int max_value = S - 1;

  Rans64State rans;
  uint32_t *ptr = (uint32_t *)encoded.data();
  if (!ptr) {
    throw std::runtime_error("invalid encoded buffer");
  }
  Rans64DecInit(&rans, &ptr);

  for (int64_t i = 0; i < n; ++i) {
    build_cdf_for_symbol(mu_ptr[i] + static_cast<float>(minmax), sc_ptr[i],
                         static_cast<int>(minmax), cdf);

    const uint32_t cum_freq = Rans64DecGet(&rans, kPrecision);
    const auto it = std::lower_bound(
        cdf.begin(), cdf.end(), static_cast<int32_t>(cum_freq + 1));
    const int32_t s = static_cast<int32_t>(std::distance(cdf.begin(), it) - 1);

    Rans64DecAdvance(&rans, &ptr, cdf[s], cdf[s + 1] - cdf[s], kPrecision);

    int32_t value = s;
    if (value == max_value) {
      int32_t val = Rans64DecGetBits(&rans, &ptr, kBypassPrecision);
      int32_t n_bypass = val;
      while (val == kMaxBypassVal) {
        val = Rans64DecGetBits(&rans, &ptr, kBypassPrecision);
        n_bypass += val;
      }
      int32_t raw_val = 0;
      for (int j = 0; j < n_bypass; ++j) {
        val = Rans64DecGetBits(&rans, &ptr, kBypassPrecision);
        raw_val |= val << (j * kBypassPrecision);
      }
      value = raw_val >> 1;
      if (raw_val & 1) {
        value = -value - 1;
      } else {
        value += max_value;
      }
    }

    out_ptr[i] = value;
  }

  return output;
}

PYBIND11_MODULE(ftic_rans, m) {
  m.doc() = "FTIC rANS bindings";
  m.def("encode_with_cdfs_tensor", &encode_with_cdfs_tensor);
  m.def("decode_with_cdfs_tensor", &decode_with_cdfs_tensor);
  m.def("encode_with_gaussian_tensor", &encode_with_gaussian_tensor);
  m.def("decode_with_gaussian_tensor", &decode_with_gaussian_tensor);
}
