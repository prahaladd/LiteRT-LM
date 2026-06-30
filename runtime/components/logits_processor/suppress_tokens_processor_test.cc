// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/components/logits_processor/suppress_tokens_processor.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_layout.h"  // from @litert
#include "runtime/components/logits_processor/suppress_tokens_config.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {
namespace {

using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::ValuesIn;
using ::testing::status::StatusIs;

enum class InputType {
  kFloat,
  kHalf,
  kTensorBufferFloat,
  kTensorBufferHalf,
};

struct TestParam {
  InputType input_type;
};

std::vector<TestParam> GetTestParams() {
  return {{InputType::kFloat},
          {InputType::kHalf},
          {InputType::kTensorBufferFloat},
          {InputType::kTensorBufferHalf}};
}

class SuppressTokensProcessorParamTest : public TestWithParam<TestParam> {
 protected:
  std::vector<float> ProcessProcessorLogits(
      SuppressTokensProcessor& processor, const std::vector<float>& logits,
      absl::Span<const ::litert::Layout::Dim> dims) {
    switch (GetParam().input_type) {
      case InputType::kFloat: {
        std::vector<float> logits_copy = logits;
        EXPECT_OK(processor.ProcessLogits(absl::MakeSpan(logits_copy), dims));
        return logits_copy;
      }
      case InputType::kHalf: {
        std::vector<tflite::half> logits_half(logits.size());
        for (size_t i = 0; i < logits.size(); ++i) {
          logits_half[i] = static_cast<tflite::half>(logits[i]);
        }
        EXPECT_OK(processor.ProcessLogits(absl::MakeSpan(logits_half), dims));
        std::vector<float> result(logits.size());
        for (size_t i = 0; i < result.size(); ++i) {
          result[i] = static_cast<float>(logits_half[i]);
        }
        return result;
      }
      case InputType::kTensorBufferFloat: {
        std::vector<float> logits_copy = logits;
        ::litert::Dimensions tb_dims(dims.begin(), dims.end());
        auto tb_logits = CopyToTensorBuffer<float>(absl::MakeSpan(logits_copy),
                                                   std::move(tb_dims));
        EXPECT_TRUE(tb_logits.HasValue());
        EXPECT_OK(processor.ProcessLogits(*tb_logits));
        auto modified = CopyFromTensorBuffer<float>(*tb_logits);
        EXPECT_TRUE(modified.HasValue());
        if (modified.HasValue()) {
          return *modified;
        }
        return logits;
      }
      case InputType::kTensorBufferHalf: {
        std::vector<tflite::half> logits_half(logits.size());
        for (size_t i = 0; i < logits.size(); ++i) {
          logits_half[i] = static_cast<tflite::half>(logits[i]);
        }
        ::litert::Dimensions tb_dims(dims.begin(), dims.end());
        auto tb_logits = CopyToTensorBuffer<tflite::half>(
            absl::MakeSpan(logits_half), std::move(tb_dims));
        EXPECT_TRUE(tb_logits.HasValue());
        EXPECT_OK(processor.ProcessLogits(*tb_logits));
        auto modified = CopyFromTensorBuffer<tflite::half>(*tb_logits);
        EXPECT_TRUE(modified.HasValue());
        std::vector<float> result(logits.size());
        if (modified.HasValue()) {
          for (size_t i = 0; i < result.size(); ++i) {
            result[i] = static_cast<float>((*modified)[i]);
          }
        }
        return result;
      }
    }
  }
};

TEST_P(SuppressTokensProcessorParamTest, SuppressesTokensProperly) {
  const int vocab_size = 10;
  SuppressTokensConfig config({2, 5, 8});
  SuppressTokensProcessor processor(/*batch_size=*/1, vocab_size, config);

  std::vector<float> initial_logits(vocab_size);
  for (int i = 0; i < vocab_size; ++i) {
    initial_logits[i] = 1.0f + 0.5f * i;
  }
  std::vector<::litert::Layout::Dim> dims = {1, 1, vocab_size};

  std::vector<float> processed_logits =
      ProcessProcessorLogits(processor, initial_logits, dims);

  const float expected_suppressed_val =
      (GetParam().input_type == InputType::kHalf ||
       GetParam().input_type == InputType::kTensorBufferHalf)
          ? static_cast<float>(tflite::half::min())
          : -std::numeric_limits<float>::infinity();

  for (int i = 0; i < vocab_size; ++i) {
    if (i == 2 || i == 5 || i == 8) {
      EXPECT_FLOAT_EQ(processed_logits[i], expected_suppressed_val);
    } else {
      EXPECT_FLOAT_EQ(processed_logits[i], initial_logits[i]);
    }
  }
}

TEST_P(SuppressTokensProcessorParamTest, MultipleBatches) {
  const int batch_size = 2;
  const int vocab_size = 10;
  SuppressTokensConfig config({1, 4});
  SuppressTokensProcessor processor(batch_size, vocab_size, config);

  std::vector<float> initial_logits(vocab_size * batch_size);
  for (int i = 0; i < vocab_size * batch_size; ++i) {
    initial_logits[i] = 1.0f + 0.5f * i;
  }
  std::vector<::litert::Layout::Dim> dims = {batch_size, 1, vocab_size};

  std::vector<float> processed_logits =
      ProcessProcessorLogits(processor, initial_logits, dims);

  const float expected_suppressed_val =
      (GetParam().input_type == InputType::kHalf ||
       GetParam().input_type == InputType::kTensorBufferHalf)
          ? static_cast<float>(tflite::half::min())
          : -std::numeric_limits<float>::infinity();

  for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    for (int i = 0; i < vocab_size; ++i) {
      if (i == 1 || i == 4) {
        EXPECT_FLOAT_EQ(processed_logits[batch_idx * vocab_size + i],
                        expected_suppressed_val);
      } else {
        EXPECT_FLOAT_EQ(processed_logits[batch_idx * vocab_size + i],
                        initial_logits[batch_idx * vocab_size + i]);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(SuppressTokensProcessorParamTest,
                         SuppressTokensProcessorParamTest,
                         ValuesIn(GetTestParams()),
                         [](const TestParamInfo<TestParam>& info) {
                           switch (info.param.input_type) {
                             case InputType::kFloat:
                               return "Float";
                             case InputType::kHalf:
                               return "Half";
                             case InputType::kTensorBufferFloat:
                               return "TensorBufferFloat";
                             case InputType::kTensorBufferHalf:
                               return "TensorBufferHalf";
                           }
                         });

TEST(SuppressTokensProcessorTest, HandlesDisabledConfig) {
  SuppressTokensConfig config = SuppressTokensConfig::Default();
  SuppressTokensProcessor processor(/*batch_size=*/1, /*vocab_size=*/5, config);

  std::vector<float> logits = {2.0f, 1.2f, -1.2f, 0.5f, 0.0f};
  std::vector<::litert::Layout::Dim> dims = {1, 1, 5};

  EXPECT_OK(
      processor.ProcessLogits(absl::MakeSpan(logits), absl::MakeSpan(dims)));

  EXPECT_FLOAT_EQ(logits[0], 2.0f);
  EXPECT_FLOAT_EQ(logits[1], 1.2f);
  EXPECT_FLOAT_EQ(logits[2], -1.2f);
  EXPECT_FLOAT_EQ(logits[3], 0.5f);
  EXPECT_FLOAT_EQ(logits[4], 0.0f);
}

TEST(SuppressTokensProcessorTest, IgnoresOutOfRangeTokenIds) {
  SuppressTokensConfig config({-1, 10, 5, 2});
  SuppressTokensProcessor processor(/*batch_size=*/1, /*vocab_size=*/5, config);

  std::vector<float> logits = {2.0f, 1.2f, -1.2f, 0.5f, 0.0f};
  std::vector<::litert::Layout::Dim> dims = {1, 1, 5};

  EXPECT_OK(
      processor.ProcessLogits(absl::MakeSpan(logits), absl::MakeSpan(dims)));

  EXPECT_FLOAT_EQ(logits[0], 2.0f);
  EXPECT_FLOAT_EQ(logits[1], 1.2f);
  EXPECT_EQ(logits[2], -std::numeric_limits<float>::infinity());
  EXPECT_FLOAT_EQ(logits[3], 0.5f);
  EXPECT_FLOAT_EQ(logits[4], 0.0f);
}

TEST(SuppressTokensProcessorTest, ReturnsErrorForInvalidLogitsDims) {
  SuppressTokensConfig config({1, 2});
  SuppressTokensProcessor processor(/*batch_size=*/2, /*vocab_size=*/5, config);

  std::vector<float> logits_float(10, 1.0f);
  std::vector<tflite::half> logits_half(10, static_cast<tflite::half>(1.0f));

  std::vector<std::vector<::litert::Layout::Dim>> invalid_dims = {
      {2, 5},     // Invalid rank (2 dims instead of 3)
      {1, 1, 5},  // Invalid batch size (1 instead of 2)
      {2, 2, 5},  // Invalid sequence length (2 instead of 1)
      {2, 1, 4},  // Invalid vocab size (4 instead of 5)
  };

  for (const auto& dims : invalid_dims) {
    EXPECT_THAT(processor.ProcessLogits(absl::MakeSpan(logits_float),
                                        absl::MakeSpan(dims)),
                StatusIs(absl::StatusCode::kInvalidArgument));
    EXPECT_THAT(processor.ProcessLogits(absl::MakeSpan(logits_half),
                                        absl::MakeSpan(dims)),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(SuppressTokensProcessorTest, ReturnsErrorForInvalidLogitsSize) {
  SuppressTokensConfig config({1, 2});
  SuppressTokensProcessor processor(/*batch_size=*/2, /*vocab_size=*/5, config);

  std::vector<::litert::Layout::Dim> dims = {2, 1, 5};
  std::vector<float> logits(11, 1.0f);  // Size 11 instead of 10

  EXPECT_THAT(
      processor.ProcessLogits(absl::MakeSpan(logits), absl::MakeSpan(dims)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Logits span size incorrectly mapped."));
}

TEST(SuppressTokensProcessorTest,
     ReturnsErrorForInvalidTensorBufferElementType) {
  SuppressTokensConfig config({1, 2});
  SuppressTokensProcessor processor(/*batch_size=*/2, /*vocab_size=*/5, config);

  std::vector<::litert::Layout::Dim> logits_dims = {2, 1, 5};

  std::vector<int32_t> logits_int32(10, 1);
  auto tb_logits_int32 = CopyToTensorBuffer<int32_t>(
      absl::MakeSpan(logits_int32),
      ::litert::Dimensions(logits_dims.begin(), logits_dims.end()));
  ASSERT_TRUE(tb_logits_int32.HasValue());
  EXPECT_THAT(processor.ProcessLogits(*tb_logits_int32),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Unsupported logits tensor format element type."));
}

}  // namespace
}  // namespace litert::lm
