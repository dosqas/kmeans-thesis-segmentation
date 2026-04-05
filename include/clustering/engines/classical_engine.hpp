#pragma once

#include "clustering/engines/kmeans_engine.hpp"
#include <opencv2/core.hpp>
#include <vector>

namespace kmeans::clustering {

    class ClassicalEngine final : public KMeansEngine {
    public:
        ClassicalEngine() = default;
        ~ClassicalEngine() = default;

        [[nodiscard]] std::vector<cv::Vec<float, 5>> run(
            const cv::Mat& samples,
            const std::vector<cv::Vec<float, 5>>& initialCenters,
            int k) override final;
    };

} // namespace kmeans::clustering