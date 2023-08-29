// Copyright 2023 CoDeX authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "codex/cc/range_ans.h"
#include "include/pybind11/numpy.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/stl.h"

namespace py = ::pybind11;

namespace codex {
namespace python {
namespace {

template <typename T>
T ThrowNotOk(absl::StatusOr<T>&& status_or) {
  if (!status_or.ok()) {
    throw absl::BadStatusOrAccess(std::move(status_or).status());
  }
  return *std::move(status_or);
}

template <typename T>
using ndarray_t = py::array_t<T, py::array::c_style>;

absl::StatusOr<ndarray_t<uint64_t>> RangeAnsStackMakeDecoder(
    const std::vector<ndarray_t<int32_t>>& pmfs) {
  if (pmfs.empty()) {
    return absl::InvalidArgumentError("No pmf in the list.");
  }
  int64_t size = pmfs.size() * 2;
  for (const auto& pmf : pmfs) {
    auto access = pmf.unchecked<1>();  // Throws if ndim != 1
    if (access.size() == 0) {
      return absl::InvalidArgumentError("Contains an empty pmf.");
    }
    size += RangeAnsStack::DecoderLookupSize(pmf);
  }

  // Consider simplifying this to limiting the number of pmfs in the list.
  if (std::numeric_limits<int32_t>::max() < size) {
    return absl::InvalidArgumentError("Decoder lookup is too large.");
  }

  ndarray_t<uint64_t> buffer(size);
  uint64_t* header_data = buffer.mutable_data();
  uint64_t* lookup_data = header_data + pmfs.size() * 2;

  for (const absl::Span<const int32_t> pmf : pmfs) {
    auto decoder_info = RangeAnsStack::MakeDecoder(pmf);
    if (!decoder_info.ok()) return std::move(decoder_info).status();
    auto [decoder_header, decoder] = *std::move(decoder_info);

    // This should be true because size <= int32 max.
    DCHECK_LE(lookup_data - header_data, std::numeric_limits<int32_t>::max());

    static_assert(RangeAnsStack::kMaxPrecision < 16);
    // pmf size cannot be greater than 2**precision. Otherwise MakeDecoder()
    // call above should have returned a non-ok status.
    DCHECK_LE(pmf.size(), std::numeric_limits<uint16_t>::max());

    // `last_nonzero` here is the maximum decodable value, used to signal an
    // overflow.
    int last_nonzero = pmf.size();
    while (last_nonzero > 0 && pmf[--last_nonzero] == 0) {  // No body.
    }

    header_data[0] = static_cast<uint64_t>(lookup_data - header_data) |
                     (static_cast<uint64_t>(last_nonzero) << 32);
    header_data[1] = decoder_header;
    header_data += 2;

    const int64_t lookup_size = RangeAnsStack::DecoderLookupSize(pmf);
    std::copy_n(decoder.get(), lookup_size, lookup_data);
    lookup_data += lookup_size;
  }

  // Reset the writeable flag.
  // https://github.com/pybind/pybind11/issues/481#issue-187301065
  reinterpret_cast<py::detail::PyArray_Proxy*>(buffer.ptr())->flags &=
      ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
  return std::move(buffer);
}

absl::StatusOr<ndarray_t<uint32_t>> RangeAnsStackMakeEncoder(
    const ndarray_t<uint64_t>& decoder_buffer) {
  constexpr auto half_roundup = [](int64_t x) { return x / 2 + x % 2; };

  const uint64_t* const decoder = decoder_buffer.unchecked<1>().data(0);
  const uint64_t* const decoder_lookup =
      decoder + static_cast<uint32_t>(*decoder);

  int64_t size = 0;
  for (auto* p = decoder + 1; p < decoder_lookup; p += 2) {
    size += 2 + half_roundup(RangeAnsStack::EncoderLookupSize(*p));
  }

  if (std::numeric_limits<int32_t>::max() < size) {
    return absl::InvalidArgumentError("Encoder lookup is too large");
  }

  ndarray_t<uint32_t> buffer(size);
  uint32_t* header = buffer.mutable_data();
  uint32_t* lookup = header + (decoder_lookup - decoder);

  for (const uint64_t* p = decoder; p < decoder_lookup; p += 2) {
    auto encoder_info =
        RangeAnsStack::MakeEncoder(*(p + 1), p + static_cast<uint32_t>(*p));
    if (!encoder_info.ok()) return std::move(encoder_info).status();
    auto [encoder_header, encoder] = *std::move(encoder_info);

    // (*p >> 32) is the last value of this distribution.
    const uint32_t last_nonzero = *p >> 32;

    if (std::numeric_limits<uint16_t>::max() < last_nonzero) {
      // Decoder lookup generation should have failed in this case.
      return absl::InternalError("Decoder lookup contains an error.");
    }

    // `encoder_header` contains the precision value only.
    // Another option here is to enforce the precision to be the same across all
    // the distributions. In that case the encoder header need not be stored per
    // distribution, and `header[1]` may simply contain the max value only.
    header[0] = lookup - header;
    header[1] = encoder_header | (last_nonzero << 16);
    header += 2;

    const int64_t lookup_size = RangeAnsStack::EncoderLookupSize(*(p + 1));
    std::copy_n(encoder.get(), lookup_size, reinterpret_cast<int16_t*>(lookup));
    lookup += half_roundup(lookup_size);
  }

  // Reset the writeable flag.
  // https://github.com/pybind/pybind11/issues/481#issue-187301065
  reinterpret_cast<py::detail::PyArray_Proxy*>(buffer.ptr())->flags &=
      ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
  return std::move(buffer);
}

struct PyRangeAnsStack {
  PyRangeAnsStack(uint32_t state, int reserve)
      : stack(
            [reserve](std::string* s) {
              s->reserve(reserve);
              return s;
            }(&sink),
            state) {}

  PyRangeAnsStack(const PyRangeAnsStack&) = delete;
  PyRangeAnsStack(PyRangeAnsStack&&) = delete;

  ndarray_t<int32_t> Pop(const ndarray_t<uint64_t>& decoder_buffer,
                         const ndarray_t<int32_t>& indices) {
    const uint64_t* const decoder = decoder_buffer.unchecked<1>().data(0);
    ndarray_t<int32_t> buffer(absl::MakeSpan(indices.shape(), indices.ndim()));
    int32_t* output = buffer.mutable_data();

    const int32_t* input = indices.data();
    const int64_t size = indices.size();
    CHECK_EQ(size, buffer.size());

    for (int64_t i = 0; i < size; ++i) {
      const uint64_t* header = decoder + 2 * *input++;
      *output++ =
          stack.Pop(*(header + 1), header + static_cast<uint32_t>(*header));
    }
    return buffer;
  }

  ndarray_t<int32_t> PopWithFallback(const ndarray_t<uint64_t>& decoder_buffer,
                                     const ndarray_t<int32_t>& indices) {
    const uint64_t* const decoder = decoder_buffer.unchecked<1>().data(0);
    ndarray_t<int32_t> buffer(absl::MakeSpan(indices.shape(), indices.ndim()));
    int32_t* output = buffer.mutable_data();

    const int32_t* input = indices.data();
    const int64_t size = indices.size();
    CHECK_EQ(size, buffer.size());

    for (int64_t i = 0; i < size; ++i) {
      const uint64_t* header = decoder + 2 * *input++;
      *output =
          stack.Pop(*(header + 1), header + static_cast<uint32_t>(*header));
      if (*output == (*header >> 32)) {
        // Gamma code expected in the stack.
        const uint32_t offset = stack.PopGammaCode32();
        if (offset % 2 != 0) {
          *output += offset >> 1;
        } else {
          *output = -static_cast<int32_t>(offset >> 1);
        }
      }
      ++output;
    }
    return buffer;
  }

  void Push(const ndarray_t<uint32_t>& encoder_buffer,
            const ndarray_t<int32_t>& indices,
            const ndarray_t<int32_t>& values) {
    const uint32_t* const encoder = encoder_buffer.unchecked<1>().data(0);
    const int64_t size = indices.size();
    CHECK_EQ(size, values.size());
    const int32_t* index_ptr = indices.data() + size;
    const int32_t* value_ptr = values.data() + size;

    for (int64_t i = 0; i < size; ++i) {
      const uint32_t* header = encoder + 2 * *--index_ptr;
      const uint16_t* lookup =
          reinterpret_cast<const uint16_t*>(header + *header);
      stack.Push(*(header + 1), lookup, *--value_ptr);
    }
  }

  // NOTE: This function requires that no value is -2**31, i.e., the minimum
  // value in int32_t range. This restriction is, in some aspect, unnecessary.
  // Since the fallback is required for negative values or positive values
  // greater than or equal to the last value of the PMF, there is enough room to
  // encode all the out-of-range values, as long as the last value > 0.
  //
  // The restriction comes from these two choices:
  //   - We assume that the last value is the most likely value among the
  //     out-of-range values.
  //   - 1 has the shortest gamma code.
  //   - Closer to the boundaries (0 and last_value), the out-of-range values
  //     are more likely than other out-of-range values. Roughly speaking,
  //     values of smaller magnitudes are more likely than values of larger
  //     magnitudes.
  //
  // Therefore we mapped the last value to the gamma code of 1, and naturally
  // last_value, last_value + 1, last_value + 2, ... mapped to 1, 3, 5, ...
  // and -1, -2 -3, ... mapped to 2, 4, 6. One caveat of this mapping is that
  // -2**31 maps to gamma code of 2**32, which does not fit in 32 bits.
  //
  // One easy remedy is to modify the mapping so that -1, -2, -3 map to gamma
  // code of 1, 3, 5, ... instead. This scheme assumes that -1 is the most
  // likely out-of-range value, instead of the last value.
  //
  // The current assumption is that shorter encoding for the last value
  // outweighs the need to support encoding -2**31.
  //
  // Another note. When last_value == 0, then there is no way to avoid at least
  // one un-encodable value since gamma coding cannot encode value=0.
  //
  // REQUIRES: values[i] != -2**31 for all i.
  void PushWithFallback(const ndarray_t<uint32_t>& encoder_buffer,
                        const ndarray_t<int32_t>& indices,
                        const ndarray_t<int32_t>& values) {
    const uint32_t* const encoder = encoder_buffer.unchecked<1>().data(0);
    const int64_t size = indices.size();
    CHECK_EQ(size, values.size());
    const int32_t* index_ptr = indices.data() + size;
    const int32_t* value_ptr = values.data() + size;

    for (int64_t i = 0; i < size; ++i) {
      const uint32_t* header = encoder + 2 * *--index_ptr;
      const uint16_t* lookup =
          reinterpret_cast<const uint16_t*>(header + *header);

      const int32_t last_value = *(header + 1) >> 16;
      if (const int32_t value = *--value_ptr;
          (0 <= value && value < last_value)) {
        stack.Push(*(header + 1), lookup, value);
      } else {
        DCHECK_NE(value, std::numeric_limits<int32_t>::min());
        stack.PushGammaCode32(
            (value < 0) ? (static_cast<uint32_t>(-value) * 2)
                        : (static_cast<uint32_t>(value - last_value) * 2 + 1));
        stack.Push(*(header + 1), lookup, last_value);
      }
    }
  }

  py::bytes Serialize() {
    const size_t prev_size = sink.size();
    {
      RangeAnsPushableStack copy = stack;
      std::move(copy).Serialize();
    }
    py::bytes serialized(sink);
    sink.resize(prev_size);
    return serialized;
  }

  static absl::StatusOr<std::unique_ptr<PyRangeAnsStack>> Deserialize(
      const py::bytes& source) {
    auto stack = std::make_unique<PyRangeAnsStack>(0, 0);
    stack->sink = source;
    auto stack_or = RangeAnsPushableStack::Deserialize(&stack->sink);
    if (!stack_or.ok()) return std::move(stack_or).status();
    stack->stack = *std::move(stack_or);
    return stack;
  }

  std::string sink;
  RangeAnsPushableStack stack;
};

// Module definition.
PYBIND11_MODULE(range_ans_pybind, m) {
  py::register_local_exception<absl::BadStatusOrAccess>(m, "AnsError",
                                                        PyExc_ValueError);

  m.doc() = "Codex RangeAns pybind11 interface";  // Module docstring.

  // See range_ans_pybind.pyi for type-annotated python interface.

  // rANS coder ----------------------------------------------------------------
  py::class_<PyRangeAnsStack>(m, "RangeAnsStack")
      .def(py::init<uint32_t, int>(), py::arg("initial_state") = 0,
           py::kw_only(), py::arg("reserve") = 0)
      .def_property_readonly("state",
                             [](PyRangeAnsStack& stack) {
                               stack.stack.Read16BitsIfAvailable();
                               return stack.stack.state;
                             })
      .def_static("make_decoder",
                  [](const std::vector<ndarray_t<int32_t>>& pmfs) {
                    return ThrowNotOk(RangeAnsStackMakeDecoder(pmfs));
                  })
      .def_static("make_encoder",
                  [](const ndarray_t<uint64_t>& decoder_buffer) {
                    return ThrowNotOk(RangeAnsStackMakeEncoder(decoder_buffer));
                  })

      .def("push",
           [](PyRangeAnsStack& stack, const ndarray_t<uint32_t>& encoder_buffer,
              const ndarray_t<int32_t>& indices,
              const ndarray_t<int32_t>& values) {
             if (indices.size() != values.size()) {
               throw absl::BadStatusOrAccess(absl::InvalidArgumentError(
                   "indices and values have different sizes"));
             }
             return stack.Push(encoder_buffer, indices, values);
           })
      .def("pop", &PyRangeAnsStack::Pop)

      .def("push_with_fallback",
           [](PyRangeAnsStack& stack, const ndarray_t<uint32_t>& encoder_buffer,
              const ndarray_t<int32_t>& indices,
              const ndarray_t<int32_t>& values) {
             if (indices.size() != values.size()) {
               throw absl::BadStatusOrAccess(absl::InvalidArgumentError(
                   "indices and values have different sizes"));
             }
             return stack.PushWithFallback(encoder_buffer, indices, values);
           })
      .def("pop_with_fallback", &PyRangeAnsStack::PopWithFallback)

      .def("serialize", &PyRangeAnsStack::Serialize)
      .def_static("deserialize", [](const py::bytes& source) {
        return ThrowNotOk(PyRangeAnsStack::Deserialize(source));
      });
}

}  // namespace
}  // namespace python
}  // namespace codex
