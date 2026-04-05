#pragma once

#include <opencv2/imgproc.hpp>

#include "clustering/preprocessors/data_preprocessor.hpp"

namespace kmeans::clustering {

class FullDataPreprocessor final : public DataPreprocessor {
  public:
    FullDataPreprocessor() = default;
    ~FullDataPreprocessor() = default;

    [[nodiscard]] cv::Mat prepare(const cv::Mat& frame) override final;
    void reset() override final {}
};

} // namespace kmeans::clustering
