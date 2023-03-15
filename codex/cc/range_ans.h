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
#ifndef CODEX_CC_RANGE_ANS_H_
#define CODEX_CC_RANGE_ANS_H_

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "absl/base/internal/endian.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace codex {
// This is a fork of table-based rANS [1] in JpegXL project [2].
// The table-based "alias method" can be found in [3].
//
// Special thanks to Luca Versari.
//
// [1] Duda, "Asymmetric numeral systems: entropy coding combining speed of
// Huffman coding with compression rate of arithmetic coding", 2013.
// https://arxiv.org/abs/1311.2540
// [2] https://gitlab.com/wg1/jpeg-xl/-/blob/master/lib/jxl/ans_common.h
// [3] https://fgiesen.wordpress.com/2014/02/18/rans-with-static-probability-distributions/
struct RangeAnsStack {
  static constexpr int kMaxPrecision = 15;
  static constexpr int kLogMaxState = 32;
  static constexpr int kLogMinState = kLogMaxState - 16;

  struct DecoderHeader {
    uint16_t lower_mask;
    int16_t precision;
    uint16_t upper_mask;
    int16_t shift;
  };

  RangeAnsStack() = default;

  // `source` is produced by `RangeAnsPushableStack::Serialize()`.
  static absl::StatusOr<RangeAnsStack> Deserialize(absl::string_view source) {
    RangeAnsStack stack;
    stack.current = source.data() + source.size();
    stack.end = source.data();

    if (source.size() < 3) {
      return absl::InvalidArgumentError(
          absl::StrCat("Serialized rANS stack is too short: ", source.size()));
    }

    // Serialize() writes 3 or 4 bytes, and this is the only case when rAns
    // encoder may write odd number of bytes. Therefore if the entire code
    // length is odd/even then Serialize() wrote 3/4 bytes.
    const int n_byte =
        std::min<int>(source.size() % 2 == 0 ? 4 : 3, source.size());
    stack.current -= n_byte;

    uint32_t temp = 0;
    std::memcpy(&temp, stack.current, n_byte);
    stack.state = absl::little_endian::ToHost32(temp);
    return stack;
  }

  // rANS encodes an integer k in [0, N) for N > 0. This integer k is called a
  // character, and N is the size of this alphabet. The argument `pmf` should be
  // a span of quantized values of Pr(X = k). The span `pmf` should sum to a
  // power of two, i.e., `2^precision` for some integer `precision`.
  // In other words,
  //
  //   Pr(X = k) = pmf[k] * 2^{-precision} for k in {0, ..., N - 1}.
  //
  // REQUIRES: 0 <= pmf[k] for k in {0, ..., N - 1}.
  // REQUIRES: 0 < precision <= 15.
  //
  // Return value contains pre-computed decoder lookup table and its metadata.
  // The first item is the metadata/header inforamtion.
  // The second item is the lookup table of size std::bit_ceil(pmf.size()).
  //
  static absl::StatusOr<std::tuple<uint64_t, std::unique_ptr<const uint64_t[]>>>
  MakeDecoder(absl::Span<const int32_t> pmf);

  // Return value contains pre-computed encoder lookup table and its metadata.
  // The first item is the metadata/header inforamtion.
  // The second item is the lookup table.
  static absl::StatusOr<std::tuple<uint16_t, std::unique_ptr<const uint16_t[]>>>
  MakeEncoder(uint64_t header, const uint64_t* decoder);

  static int64_t DecoderLookupSize(absl::Span<const int32_t> pmf);
  static int64_t EncoderLookupSize(uint64_t decoder_header);

  // TODO(ssjhv): When shift <= 8, promote shift to 8, and write a special
  // case when shift=8 is hard-coded.
  //
  // Pops (or decodes) a value from the stack.
  // The argument is the return value of `MakeDecoder()`.
  int Pop(uint64_t header, const uint64_t* decoder) {
    // Read 2 bytes if `state` is below threshold.
    // This keeps `state` above 2**kLogMinState, except at the end.
    // TODO(ssjhv): This can be done at the very last, and it makes state
    // retrieval easier. Benchmark both cases.
    Read16BitsIfAvailable();

    const auto h = absl::bit_cast<DecoderHeader>(header);

    // Split the lower `precision` bits into `major` and `minor` indices.
    const uint32_t major = state & h.lower_mask;
    const uint32_t minor = state & h.upper_mask;
    state >>= h.precision;

    // entry contains lookup table for either value0 or value1.
    // When the `minor` index above is less than the cutoff, the decoded value
    // is value0, otherwise it is value1.
    //
    // value0 and value1 have offset and freq. After decoding the top value,
    // the lower `precision` bits of the new state is: offset + minor
    //
    // entry0 (lower 32 bits) contains cutoff, value1, and freq of value0.
    // entry1 (upper 32 bits) contains offset and freq of value1.
    //
    // value0 is always the same as `major`. This is implicit and this value is
    // not explicitly recorded in the decoder lookup table. Similarly, the
    // offset of value0 is always implicitly zero.
    //
    const uint64_t entry = decoder[major];
    const uint32_t entry0 = entry;
    const uint32_t entry1 = entry >> 32;
    const uint32_t cutoff = entry0 & h.upper_mask;
    const uint32_t value1 = entry0 & h.lower_mask;

    const bool is_value0 = (minor < cutoff);
    state *= (is_value0 ? entry0 : entry1) >> 16;  // freq of value0 or value1.
    state += (is_value0 ? 0 : entry1) & 0xFFFF;  // offset of value0 or value1.
    state += minor >> h.shift;

    return is_value0 ? major : value1;
  }

  void Read16BitsIfAvailable() {
    if ((state >> kLogMinState) == 0 && current != end) {
      CHECK(end + 2 <= current);
      state <<= 16;
      state += (*--current << 8);
      state += *--current;
    }
  }

  uint32_t state;
  const char* current;
  const char* end;
};

struct RangeAnsPushableStack : RangeAnsStack {
  // The ownership of `sink` is not transferred. It is the user's responsibility
  // to keep `sink` valid until `Serialize()` is called.
  //
  // The stack appends to `sink`, and the existing contents of `sink` is
  // untouched.
  explicit RangeAnsPushableStack(std::string* sink, uint32_t initial_state = 0)
      : sink(sink) {
    current = end = sink->data() + sink->size();
    state = initial_state;
  }

  // Pushes (or encodes) a value in to the stack using the encoder information.
  // The behavior is undefined when `pmf[value]` is out-of-range or zero.
  void Push(uint16_t header, const uint16_t* encoder, int value) {
    const int precision = header;

    const uint32_t size = encoder[2 * value + 1];
    const uint16_t* lookup = &encoder[2 * value] + encoder[2 * value];

    // If the state is going to overflow, write the lower 16 bits to the sink.
    if (size <= (state >> (kLogMaxState - precision))) {
      const char* limit = sink->data() + sink->size();
      DCHECK(sink->data() <= end && end <= current && current <= limit);
      if (limit - current < 2) {
        constexpr int kAppendSize = 64;
        static_assert(2 <= kAppendSize);

        const ptrdiff_t current_diff = current - sink->data();
        const ptrdiff_t end_diff = end - sink->data();
        // sink->resize(current_diff + kAppendSize);
        sink->append(kAppendSize, 0);
        current = sink->data() + current_diff;
        end = sink->data() + end_diff;
      }
      absl::little_endian::Store16(const_cast<char*>(current),
                                   static_cast<uint16_t>(state));
      current += 2;
      state >>= 16;
    }

    // Consider using multiplicative inverses instead of division.
    const uint32_t quotient = state / size;
    const uint32_t remainder = state - quotient * size;
    state = (quotient << precision) + lookup[remainder];
  }

  // The users must call Serialize() after encoding all characters. After
  // encoding the last character, the stack may have internal state that is
  // not dumped out to `sink` yet. Without Serialize(), the produced stream
  // cannot be decoded correctly.
  std::string& Serialize() && {
    // Write 3 or 4 bytes depending on the state value.
    const int n_byte = (state >> 24) == 0 ? 3 : 4;

    const ptrdiff_t current_diff = current - sink->data();
    sink->resize(current_diff + n_byte);

    const uint32_t word = absl::little_endian::FromHost32(state);
    std::memcpy(sink->data() + current_diff, &word, n_byte);
    return *sink;
  }

  // The main difference from the constructor version is that `Deserialize()`
  // reads the content of `sink` and attempts to recover the last state.
  static absl::StatusOr<RangeAnsPushableStack> Deserialize(std::string* sink) {
    RangeAnsPushableStack stack(sink);
    auto status_or = RangeAnsStack::Deserialize(*sink);
    if (!status_or.ok()) return status_or.status();
    static_cast<RangeAnsStack&>(stack) = *std::move(status_or);
    return stack;
  }

  std::string* sink;  // Not owned.
};

}  // namespace codex

#endif  // CODEX_CC_RANGE_ANS_H_
