// Copyright 2024 CoDeX authors.
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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/base/casts.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"

namespace codex {
namespace {

using ::testing::Each;
using ::testing::Eq;
using ::testing::Pointwise;
using ::testing::WhenSorted;
using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

void RandomCdf(absl::BitGen* gen, absl::Span<int32_t> cdf) {
  if (cdf.size() <= 2) {
    return;
  }

  const int32_t mid = (cdf.size() - 1) / 2;
  const int32_t min_value = cdf.front() + mid;
  const int32_t max_value = cdf.back() - (cdf.size() - 1 - mid);
  cdf[mid] = absl::Uniform(*gen, min_value, max_value + 1);

  RandomCdf(gen, absl::Span<int32_t>(cdf.data(), mid + 1));
  RandomCdf(gen, absl::Span<int32_t>(cdf.data() + mid, cdf.size() - mid));
}

TEST(RangeAnsStackTest, EncodeDecode) {
  absl::BitGen gen;
  const int precision = absl::Uniform(gen, 6, 15);
  const int max_value = absl::Uniform(gen, 10, 1 << precision) - 1;

  std::vector<int32_t> pmf(max_value + 1, 0);
  std::vector<int> data(1 << precision);

  for (int i = 0; i < data.size(); ++i) {
    data[i] = absl::LogUniform(gen, 0, max_value);
    ++pmf[data[i]];
  }

  auto [decoder_header, decoder] = RangeAnsStack::MakeDecoder(pmf).value();
  auto [encoder_header, encoder] =
      RangeAnsStack::MakeEncoder(decoder_header, decoder.get()).value();

  {
    // Decoder table sanity check.
    std::vector<int> temp;
    std::transform(decoder.get(),
                   decoder.get() + RangeAnsStack::DecoderLookupSize(pmf),
                   std::back_inserter(temp),
                   [](uint64_t x) { return static_cast<uint32_t>(x) >> 16; });

    ASSERT_THAT(absl::MakeSpan(temp).subspan(0, pmf.size()),
                Pointwise(Eq(), pmf));
    ASSERT_THAT(absl::MakeSpan(temp).subspan(pmf.size()), Each(Eq(0)));
  }
  {
    // Encoder table sanity check.
    auto lookup = absl::MakeSpan(
        encoder.get(), RangeAnsStack::EncoderLookupSize(decoder_header));
    lookup.remove_prefix(2 * RangeAnsStack::DecoderLookupSize(pmf));
    ASSERT_THAT(lookup[0], 0);
    std::vector<uint16_t> range(1 << precision, 0);
    std::iota(range.begin(), range.end(), 0);
    ASSERT_THAT(lookup, WhenSorted(range));
  }

  std::string output;
  RangeAnsPushableStack pushable_stack(&output);

  double expected = 0.0;
  for (int point : data) {
    pushable_stack.Push(encoder_header, encoder.get(), point);
    expected += precision - std::log2(pmf[point]);
  }
  std::move(pushable_stack).Serialize();

  if (1.05 * expected < output.size() * 8) {
    LOG(ERROR) << "Expected length: " << expected
               << " vs. actual: " << output.size() * 8 << " (over 105%)";
  } else {
    LOG(INFO) << "Expected length: " << expected
              << " vs. actual: " << output.size() * 8;
  }

  auto stack = RangeAnsStack::Deserialize(output).value();

  std::vector<int> decoded(data.size());
  for (int i = data.size() - 1; i >= 0; --i) {
    decoded[i] = stack.Pop(decoder_header, decoder.get());
  }

  EXPECT_THAT(decoded, Pointwise(Eq(), data));
}

TEST(RangeAnsStackTest, EncodeDecodeInterleavedTwoStreams) {
  absl::BitGen gen;
  const int precision = absl::Uniform(gen, 6, 15);
  const int max_value0 = absl::Uniform(gen, 10, 20);
  const int max_value1 = absl::Uniform(gen, 20, 1 << precision);

  std::vector<int> data(1000 * 1000);
  std::vector<int32_t> pmf0(max_value0 + 1, 1);
  std::vector<int32_t> pmf1(max_value1 + 1, 1);

  for (int i = 0; i < data.size(); i += 2) {
    const int value0 = absl::Zipf(gen, max_value0);
    const int value1 = absl::LogUniform(gen, 0, max_value1);
    data[i] = value0;
    data[i + 1] = value1;
    if (pmf0.size() + (i / 2) < (1 << precision)) {
      ++pmf0[value0];
    }
    if (pmf1.size() + (i / 2) < (1 << precision)) {
      ++pmf1[value1];
    }
  }

  auto [decoder0_header, decoder0] = RangeAnsStack::MakeDecoder(pmf0).value();
  auto [decoder1_header, decoder1] = RangeAnsStack::MakeDecoder(pmf1).value();
  auto [encoder0_header, encoder0] =
      RangeAnsStack::MakeEncoder(decoder0_header, decoder0.get()).value();
  auto [encoder1_header, encoder1] =
      RangeAnsStack::MakeEncoder(decoder1_header, decoder1.get()).value();

  std::string output;
  auto pushable_stack = RangeAnsPushableStack(&output);

  double expected = 0.0;
  for (int i = 0; i < data.size(); i += 2) {
    pushable_stack.Push(encoder0_header, encoder0.get(), data[i + 0]);
    expected += precision - std::log2(pmf0[data[i + 0]]);
    pushable_stack.Push(encoder1_header, encoder1.get(), data[i + 1]);
    expected += precision - std::log2(pmf1[data[i + 1]]);
  }
  std::move(pushable_stack).Serialize();

  if (1.05 * expected < output.size() * 8) {
    LOG(ERROR) << "Expected length: " << expected
               << " vs. actual: " << output.size() * 8 << " (over 105%)";
  } else {
    LOG(INFO) << "Expected length: " << expected
              << " vs. actual: " << output.size() * 8;
  }

  auto stack = RangeAnsPushableStack::Deserialize(&output).value();
  std::vector<int16_t> decoded(data.size());

  ASSERT_EQ(data.size() % 2, 0);
  for (int64_t i = data.size(); i > 0; i -= 2) {
    decoded[i - 1] = stack.Pop(decoder1_header, decoder1.get());
    decoded[i - 2] = stack.Pop(decoder0_header, decoder0.get());
  }

  EXPECT_THAT(decoded, Pointwise(Eq(), data));
}

// Tests mixed stream of rANS and Elias gamma codes.
TEST(RangeAnsStackTest, EncodeDecodeMixedGamma) {
  absl::BitGen gen;
  const int precision = absl::Uniform(gen, 6, 15);
  const int max_value = absl::Uniform(gen, 10, 1 << precision) - 1;
  const double out_of_range_probability = 0.05;

  std::vector<int32_t> pmf(max_value + 1, 1);
  std::vector<uint32_t> data(100 * 1000);

  for (int i = 0; i < data.size(); ++i) {
    data[i] = absl::LogUniform(gen, 0, max_value);
    if (pmf.size() + i < (1 << precision)) {
      ++pmf[data[i]];
    }
  }

  auto [decoder_header, decoder] = RangeAnsStack::MakeDecoder(pmf).value();
  auto [encoder_header, encoder] =
      RangeAnsStack::MakeEncoder(decoder_header, decoder.get()).value();

  // Add a small number of (possibly) out-of-range values.
  for (int i = 0; i < data.size() * out_of_range_probability; ++i) {
    // Assignment truncates the bit patterns from gen().
    data[absl::Uniform(gen, 0u, data.size())] = gen();
  }

  // Always add a few small number of values on the border.
  for (int i = 0; i < 5; ++i) {
    data[absl::Uniform(gen, 0u, data.size())] = max_value;
  }

  std::string output;
  RangeAnsPushableStack pushable_stack(&output);

  double expected = 0.0;
  for (int i = data.size() - 1; i >= 0; --i) {
    if (uint32_t x = data[i]; x < max_value) {
      pushable_stack.Push(encoder_header, encoder.get(), data[i]);
      expected += precision - std::log2(pmf[data[i]]);
    } else {
      // When the values are max_value or greater,
      // encode the offset as a Gamma code.
      pushable_stack.PushGammaCode32(x - max_value + 1);
      expected += 2 * absl::bit_width(x - max_value + 1) - 1;

      pushable_stack.Push(encoder_header, encoder.get(), max_value);
      expected += precision - std::log2(pmf[max_value]);
    }
  }
  std::move(pushable_stack).Serialize();

  if (1.05 * expected < output.size() * 8) {
    LOG(ERROR) << "Expected length: " << expected
               << " vs. actual: " << output.size() * 8 << " (over 105%)";
  } else {
    LOG(INFO) << "Expected length: " << expected
              << " vs. actual: " << output.size() * 8;
  }

  auto stack = RangeAnsStack::Deserialize(output).value();

  std::vector<uint32_t> decoded(data.size());
  for (int i = 0; i < data.size(); ++i) {
    decoded[i] = stack.Pop(decoder_header, decoder.get());
    if (decoded[i] == max_value) {
      decoded[i] += stack.PopGammaCode32() - 1;
    }
  }

  EXPECT_THAT(decoded, Pointwise(Eq(), data));
}

// Entire stream encoded/decoded with Gamma coding.
TEST(RangeAnsStackTest, EncodeDecodeGamma) {
  std::vector<uint32_t> data(4096);

  absl::BitGen gen;
  for (int i = 0; i < data.size(); ++i) {
    data[i] = std::max(1u, static_cast<uint32_t>(gen()));
  }

  // Always add a few smallest numbers.
  for (int i = 0; i < 5; ++i) {
    data[absl::Uniform(gen, 0u, data.size())] = 1;
  }

  // Always add a few largest numbers.
  for (int i = 0; i < 5; ++i) {
    data[absl::Uniform(gen, 0u, data.size())] =
        std::numeric_limits<uint32_t>::max();
  }

  std::string output;
  RangeAnsPushableStack pushable_stack(&output);

  double expected = 0.0;
  for (int i = data.size() - 1; i >= 0; --i) {
    pushable_stack.PushGammaCode32(data[i]);
    expected += 2 * absl::bit_width(static_cast<uint32_t>(data[i])) - 1;
  }
  std::move(pushable_stack).Serialize();

  if (1.05 * expected < output.size() * 8) {
    LOG(ERROR) << "Expected length: " << expected
               << " vs. actual: " << output.size() * 8 << " (over 105%)";
  } else {
    LOG(INFO) << "Expected length: " << expected
              << " vs. actual: " << output.size() * 8;
  }

  auto stack = RangeAnsStack::Deserialize(output).value();
  std::vector<int32_t> decoded(data.size());
  for (int i = 0; i < data.size(); ++i) {
    decoded[i] = stack.PopGammaCode32();
  }

  EXPECT_THAT(decoded, Pointwise(Eq(), data));
}

TEST(RangeAnsStackTest, ErrorCondition) {
  constexpr auto kInvalidArgument = absl::StatusCode::kInvalidArgument;

  // Decoder construction errors.
  {
    std::vector<int32_t> pmf;
    EXPECT_THAT(RangeAnsStack::MakeDecoder(pmf), StatusIs(kInvalidArgument));

    pmf = {255};  // sum(pmf) is not a power of two.
    EXPECT_THAT(RangeAnsStack::MakeDecoder(pmf), StatusIs(kInvalidArgument));

    pmf = {1};  // precision == 0 (too small).
    EXPECT_THAT(RangeAnsStack::MakeDecoder(pmf), StatusIs(kInvalidArgument));

    pmf = {1 << 17};  // precision == 17 (too large).
    EXPECT_THAT(RangeAnsStack::MakeDecoder(pmf), StatusIs(kInvalidArgument));

    pmf = {std::numeric_limits<int32_t>::min()};  // negative pmf sum.
    EXPECT_THAT(RangeAnsStack::MakeDecoder(pmf), StatusIs(kInvalidArgument));

    // Ok cases.
    pmf = {2};
    EXPECT_THAT(RangeAnsStack::MakeDecoder(pmf), IsOk());
    pmf = {256};
    EXPECT_THAT(RangeAnsStack::MakeDecoder(pmf), IsOk());
    pmf = {100, 156};
    EXPECT_THAT(RangeAnsStack::MakeDecoder(pmf), IsOk());

    // Error: pmf sum is positive but contains a negative value.
    pmf = {100, -1, 157};
    EXPECT_THAT(RangeAnsStack::MakeDecoder(pmf), StatusIs(kInvalidArgument));

    // Error: precision < log2ceil(pmf.size)
    pmf = {1, 1, 1, 1, 0, 1, 1, 1, 1};
    EXPECT_THAT(RangeAnsStack::MakeDecoder(pmf), StatusIs(kInvalidArgument));
  }

  // Encoder construction errors.
  {
    // Alphabet size = 6.
    std::vector<int32_t> pmf = {18, 18, 18, 71, 0, 131};
    auto [decoder_header, decoder] = RangeAnsStack::MakeDecoder(pmf).value();

    const int precision = (decoder_header >> 16) & 0xFFFF;
    const int shift = decoder_header >> 48;
    ASSERT_THAT(precision, 8);
    ASSERT_THAT(shift, 3);

    {
      // Invalid shift value.
      auto corrupt =
          absl::bit_cast<RangeAnsStack::DecoderHeader>(decoder_header);
      corrupt.shift = -1;
      EXPECT_THAT(RangeAnsStack::MakeEncoder(absl::bit_cast<uint64_t>(corrupt),
                                             decoder.get()),
                  StatusIs(kInvalidArgument));
    }

    {
      // Invalid precision value.
      auto corrupt =
          absl::bit_cast<RangeAnsStack::DecoderHeader>(decoder_header);
      corrupt.precision = 17;
      EXPECT_THAT(RangeAnsStack::MakeEncoder(absl::bit_cast<uint64_t>(corrupt),
                                             decoder.get()),
                  StatusIs(kInvalidArgument));

      corrupt.precision = -1;
      EXPECT_THAT(RangeAnsStack::MakeEncoder(absl::bit_cast<uint64_t>(corrupt),
                                             decoder.get()),
                  StatusIs(kInvalidArgument));
    }

    {
      // Contaminate decoder lookup table.
      const uint64_t saved = decoder[0];
      const_cast<uint64_t&>(decoder[0]) = 0;
      EXPECT_THAT(RangeAnsStack::MakeEncoder(decoder_header, decoder.get()),
                  StatusIs(kInvalidArgument));
      const_cast<uint64_t&>(decoder[0]) = saved;
    }

    // Ok case.
    EXPECT_THAT(RangeAnsStack::MakeEncoder(decoder_header, decoder.get()),
                IsOk());
  }

  // Encoder lookup displacement exceeds uint16_t.
  {
    std::vector pmf(1 << 15, 1);
    auto [decoder_header, decoder] = RangeAnsStack::MakeDecoder(pmf).value();
    EXPECT_THAT(RangeAnsStack::MakeEncoder(decoder_header, decoder.get()),
                StatusIs(kInvalidArgument));
  }

  // Deserialization errors.
  {
    std::string empty;
    EXPECT_THAT(RangeAnsPushableStack::Deserialize(&empty),
                StatusIs(kInvalidArgument));
  }
}

}  // namespace
}  // namespace codex
