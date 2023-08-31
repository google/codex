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
#include "codex/cc/range_ans.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"

namespace codex {

absl::StatusOr<std::tuple<uint64_t, std::unique_ptr<const uint64_t[]>>>
RangeAnsStack::MakeDecoder(absl::Span<const int32_t> pmf) {
  const int32_t sum = std::accumulate(pmf.begin(), pmf.end(), 0);
  if (sum <= 1 || (1 << kMaxPrecision) < sum ||
      !absl::has_single_bit(static_cast<uint32_t>(sum))) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The sum of PMF should be a power of two (1 < sum <= 32768): ", sum));
  }
  ABSL_ASSUME(sum != 0);
  const int precision = absl::bit_width(static_cast<uint32_t>(sum)) - 1;
  CHECK_GT(precision, 0);

  const int shift =
      absl::bit_width(pmf.size()) - (absl::has_single_bit(pmf.size()) ? 1 : 0);
  CHECK_EQ(1u << shift, absl::bit_ceil(pmf.size()));
  if (precision < shift) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Alphabet size is too large for precision: precision=", precision,
        ", and 2**precision < alphabet.size=", pmf.size()));
  }

  auto lookup = std::make_unique<uint64_t[]>(1 << shift);

  const auto make_split_entry = [shift](uint64_t value1, uint64_t freq1,
                                        uint64_t offset1, uint64_t cutoff,
                                        uint64_t freq0) {
    return value1 | (cutoff << shift) | (freq0 << 16) | (offset1 << 32) |
           (freq1 << 48);
  };
  constexpr auto make_whole_entry = [](uint64_t value, uint64_t freq) {
    // When cutoff = 0, offset1 = 0, freq1 = freq0
    return value | (freq << 16) | (freq << 48);
  };

  const uint32_t capacity = 1u << (precision - shift);
  std::vector<int> overfull;
  std::vector<int> underfull;
  size_t i = 0;
  for (; i < pmf.size(); ++i) {
    if (pmf[i] < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("PMF contains negative value: pmf[", i, "]=", pmf[i]));
    }
    const uint32_t freq = pmf[i];
    lookup[i] = freq;

    if (freq < capacity) {
      underfull.push_back(i);
    } else if (freq > capacity) {
      overfull.push_back(i);
    } else {
      lookup[i] = make_whole_entry(i, freq);
    }
  }
  for (; i < (1u << shift); ++i) {
    lookup[i] = 0;
    underfull.push_back(i);
  }

  const int pmf_size = pmf.size();

  while (!underfull.empty()) {
    CHECK(!overfull.empty());
    const int i_over = overfull.back();
    const int i_under = underfull.back();
    underfull.pop_back();

    uint64_t& c_over = lookup[i_over];
    uint64_t& c_under = lookup[i_under];

    CHECK_LT(c_under, capacity);
    const int32_t room = capacity - c_under;

    const int32_t f_under = (i_under < pmf_size) ? pmf[i_under] : 0;
    const int32_t f_over = pmf[i_over];

    CHECK(i_under < pmf_size || c_under == 0);
    CHECK_GT(c_over, capacity);
    c_under =
        make_split_entry(i_over, f_over, c_over - capacity, c_under, f_under);

    c_over -= room;
    if (c_over <= capacity) {
      overfull.pop_back();
    }
    if (c_over < capacity) {
      underfull.push_back(i_over);
    } else if (c_over == capacity) {
      c_over = make_whole_entry(i_over, f_over);
    }
  }
  CHECK(overfull.empty());

  DecoderHeader header;
  header.lower_mask = (1 << shift) - 1;
  header.upper_mask = ((1 << precision) - 1) ^ header.lower_mask;
  header.precision = precision;
  header.shift = shift;

  return std::make_tuple(absl::bit_cast<uint64_t>(header), std::move(lookup));
}

namespace {
#if defined(__GNUC__) && defined(__BMI2__) && defined(__x86_64__)
uint64_t GetLowBits(uint64_t src, int nbits) { return _bzhi_u64(src, nbits); }
#else
uint64_t GetLowBits(uint64_t src, int nbits) {
  const uint64_t ones = ~uint64_t{0};
  return (nbits == 0) ? 0 : (src & (ones >> (sizeof(ones) * 8 - nbits)));
}
#endif
}  // namespace

absl::StatusOr<std::tuple<uint16_t, std::unique_ptr<const uint16_t[]>>>
RangeAnsStack::MakeEncoder(uint64_t header, const uint64_t* decoder) {
  const auto decoder_header = absl::bit_cast<DecoderHeader>(header);
  const int shift = decoder_header.shift;
  const int precision = decoder_header.precision;
  if (!(0 <= shift && shift <= precision && precision <= kMaxPrecision)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid decoder precision and/or shift: precision=", precision,
        ", shift=", shift));
  }
  const int capacity = 1 << (precision - shift);

  std::vector<int32_t> pmf;
  std::vector<int32_t> cdf;
  cdf.push_back(0);
  for (int i = 0; i < (1 << shift); ++i) {
    // Use unsigned type intermediately to avoid sign extension.
    pmf.push_back(static_cast<uint32_t>(decoder[i]) >> 16);
    cdf.push_back(cdf[cdf.size() - 1] + pmf[pmf.size() - 1]);
  }
  if (cdf[cdf.size() - 1] != (1 << precision)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "CDF from decoder should end at 2**precision: cdf[-1]=", cdf.back(),
        ", precision=", precision));
  }

  const int64_t size = (1 << precision) + 2 * pmf.size();
  if (std::numeric_limits<uint16_t>::max() < size) {
    return absl::InvalidArgumentError(
        absl::StrCat("Encoder lookup is too large: precision=", precision,
                     ", decoder.size=", 1 << shift));
  }
  auto buffer = std::make_unique<uint16_t[]>(size);
  auto jump = buffer.get();
  auto lookup = buffer.get() + 2 * pmf.size();

  for (size_t i = 0; i < pmf.size(); ++i) {
    jump[2 * i + 0] = cdf[i] + 2 * (pmf.size() - i);
    jump[2 * i + 1] = pmf[i];
  }

  for (int i = 0; i < (1 << shift); ++i) {
    const uint64_t e = decoder[i];
    int32_t j = 0;
    {
      const int32_t cutoff = (e & 0xFFFF) >> shift;
      const int32_t lower = cdf[i];
      for (; j < cutoff; ++j) {
        lookup[lower + j] = (j << shift) + i;
      }
    }

    {
      const int32_t offset = (e >> 32) & 0xFFFF;
      const int32_t lower = cdf[GetLowBits(e, shift)] + offset;
      for (; j < capacity; ++j) {
        lookup[lower + j] = (j << shift) + i;
      }
    }
  }

  return std::make_tuple(precision, std::move(buffer));
}

int64_t RangeAnsStack::DecoderLookupSize(absl::Span<const int32_t> pmf) {
  return absl::bit_ceil(pmf.size());
}

int64_t RangeAnsStack::EncoderLookupSize(uint64_t decoder_header) {
  const auto header = absl::bit_cast<DecoderHeader>(decoder_header);
  return (1 << header.precision) + 2 * (1 << header.shift);
}

// Decodes or encodes a gamma code when value < 2**16.
//
// This bit stack reads and writes 16 bits at a time from the bitstream.
// Therefore when the value being pushed and popped is at most 16 bits, the
// operations may be carried out with at most one read or write.
//
// NOTE: Gamma code is used here as a fallback scheme. This feature should not
// complicate the main stack design or implementation.
//
// Gamma coding splits a strictly positive number n into its MSB and the rest of
// the bit pattern. For example, when n=42, its bit pattern 0b101010 is
// represented as
//
//   0b101010 = 0b100000 + 0b01010
//
// and these two numbers are pushed into the bit stack. During decoding, the
// decoder first pops 0b100000. From the number of trailing zeros, the decoder
// knows 5 bits should be decoded the next, and pops 0b01010 from the stack.
//
uint32_t RangeAnsStack::PopGammaCode16() {
  Read16BitsIfAvailable();
  const int tzcnt = absl::countr_zero(state);
  DCHECK_LT(tzcnt, 16);
  state >>= (tzcnt + 1);

  Read16BitsIfAvailable();
  const uint32_t value = (1u << tzcnt) + GetLowBits(state, tzcnt);
  state >>= tzcnt;
  return value;
}

void RangeAnsPushableStack::PushGammaCode16(uint16_t value16) {
  DCHECK_GT(value16, 0);
  const uint32_t value = value16;
  ABSL_ASSUME(value != 0);
  const int n = absl::bit_width(value);
  DCHECK_GT(n, 0);

  if (n > 1) {
    Flush16BitsIfAboutToOverflow(1, n - 1);
    state = (state << (n - 1)) | GetLowBits(value, n - 1);
  }

  Flush16BitsIfAboutToOverflow(1, n);
  state = (state << n) | (1u << (n - 1));
}

// Gamma coding for 32-bit values is more complicated. As usual, the input value
// is split into its MSB and the rest of the bit pattern. Then their upper 16
// bits are encoded first, then their lower 16 bits are encoded second.
//
// For example, suppose the input value were 0x42e7a1, which would split into
// its MSB isolation 0x400000 and the rest 0x02e7a1. Four numbers are pushed
// into the stack:
//
//   1. 0x02  (hi-word of 0x02e7a1, 5 bits)
//   2. 0x40  (hi-word of 0x400000, 6 bits)
//   3. 0xe7a1 (lo-word of 0x02e7a1, 16 bits)
//   4. 0x0000 (lo-word of 0x400000, 16 bits)
//
// Note that fixed 16 bits are used to encode lo-words. During the decode, the
// decoder observes 0x0000, 16 serial zeros. This tells the decoder that the
// next value to be popped from the stack is 16-bit wide, and there is one more
// 16-bit gamma code to be decoded after that.
//
uint32_t RangeAnsStack::PopGammaCode32() {
  Read16BitsIfAvailable();
  if (static_cast<uint16_t>(state) != 0) {
    // NOTE: PopGammaCode16() has another Read16BitsIfAvailable() call at the
    // beginning, which might cause a small overhead.
    return PopGammaCode16();
  }

  state >>= 16;
  Read16BitsIfAvailable();
  const uint32_t lo = static_cast<uint16_t>(state);
  state >>= 16;

  return lo | (PopGammaCode16() << 16);
}

void RangeAnsPushableStack::PushGammaCode32(uint32_t value) {
  DCHECK_GT(value, 0);
  if ((value >> 16) == 0) {
    PushGammaCode16(value);
    return;
  }

  PushGammaCode16(value >> 16);

  Flush16BitsIfAboutToOverflow(1, 16);
  state = (state << 16) | static_cast<uint16_t>(value);

  Flush16BitsIfAboutToOverflow(1, 16);
  state <<= 16;  // Push 16 unset bits.
}

}  // namespace codex
