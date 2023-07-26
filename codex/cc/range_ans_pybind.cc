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
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "codex/cc/range_ans.h"
#include "include/pybind11/numpy.h"
#include "include/pybind11/pybind11.h"
#include "pybind11_abseil/absl_casters.h"
#include "pybind11_abseil/status_casters.h"

namespace py = ::pybind11;

namespace codex {
namespace python {
namespace {

template <typename T>
T ThrowNotOk(absl::StatusOr<T>&& status_or) {
  if (!status_or.ok()) {
    py::object conversion_should_raise_exception = py::cast(status_or.status());
    LOG(FATAL) << "Exception not raised";
  }
  return *std::move(status_or);
}

void ThrowNotOk(absl::Status&& status) {
  if (!status.ok()) {
    py::object conversion_should_raise_exception = py::cast(status);
    LOG(FATAL) << "Exception not raised";
  }
}

template <typename T>
using ndarray_t = py::array_t<T, py::array::c_style>;

absl::StatusOr<ndarray_t<uint64_t>> RangeAnsStackMakeDecoder(
    const std::vector<ndarray_t<int32_t>>& pmfs) {
  if (pmfs.empty()) {
    return absl::InvalidArgumentError("No distribution in the list");
  }
  int64_t size = pmfs.size() * 2;
  for (const auto& pmf : pmfs) {
    auto access = pmf.unchecked<1>();  // Throws if ndim != 1
    if (access.size() == 0) {
      return absl::InvalidArgumentError("PMF is empty");
    }
    size += RangeAnsStack::DecoderLookupSize(pmf);
  }

  ndarray_t<uint64_t> buffer(size);
  uint64_t* header_data = buffer.mutable_data();
  uint64_t* lookup_data = header_data + pmfs.size() * 2;

  for (const auto& pmf : pmfs) {
    auto decoder_info = RangeAnsStack::MakeDecoder(pmf);
    if (!decoder_info.ok()) return decoder_info.status();
    auto [decoder_header, decoder] = *std::move(decoder_info);
    header_data[0] = lookup_data - header_data;
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
  const uint64_t* const decoder_lookup = decoder + *decoder;

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
    auto encoder_info = RangeAnsStack::MakeEncoder(*(p + 1), p + *p);
    if (!encoder_info.ok()) return encoder_info.status();
    auto [encoder_header, encoder] = *std::move(encoder_info);

    header[0] = lookup - header;
    header[1] = encoder_header;
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
      *output++ = stack.Pop(*(header + 1), header + *header);
    }
    return buffer;
  }

  void Push(const ndarray_t<uint32_t>& encoder_buffer,
            const ndarray_t<int32_t>& indices,
            const ndarray_t<int32_t>& values) {
    const uint32_t* const encoder = encoder_buffer.unchecked<1>().data(0);
    const int64_t size = indices.size();
    const int32_t* index_ptr = indices.data() + size;

    if (size != values.size()) {
      ThrowNotOk(absl::InvalidArgumentError(
          "indices and values have different sizes"));
    }
    const int32_t* value_ptr = values.data() + size;

    for (int64_t i = 0; i < size; ++i) {
      const uint32_t* header = encoder + 2 * *--index_ptr;
      const uint16_t* lookup =
          reinterpret_cast<const uint16_t*>(header + *header);
      stack.Push(*++header, lookup, *--value_ptr);
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
    if (!stack_or.ok()) return stack_or.status();
    stack->stack = *std::move(stack_or);
    return stack;
  }

  std::string sink;
  RangeAnsPushableStack stack;
};

// Module definition.
PYBIND11_MODULE(range_ans_pybind, m) {
  pybind11::google::ImportStatusModule();

  m.doc() = "Codex RangeAns pybind11 interface";  // Module docstring.

  // rANS coder ----------------------------------------------------------------
  py::class_<PyRangeAnsStack>(m, "RangeAnsStack")
      .def(py::init<uint32_t, int>(), py::arg("initial_state") = 0,
           py::kw_only(), py::arg("reserve") = 0)
      .def_property_readonly("state",
                             [](PyRangeAnsStack& stack) {
                               stack.stack.Read16BitsIfAvailable();
                               return stack.stack.state;
                             })
      .def_static("make_decoder", &RangeAnsStackMakeDecoder)
      .def_static("make_encoder", &RangeAnsStackMakeEncoder)
      .def("push", &PyRangeAnsStack::Push)
      .def("pop", &PyRangeAnsStack::Pop)
      .def("serialize", &PyRangeAnsStack::Serialize)
      .def_static("deserialize", &PyRangeAnsStack::Deserialize);
}

}  // namespace
}  // namespace python
}  // namespace codex
