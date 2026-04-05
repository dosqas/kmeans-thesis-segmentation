#pragma once

#include "clustering/initializers/initializer.hpp"
#include <random>

namespace kmeans::clustering {

    class RandomInitializer final : public Initializer {
    public:
        RandomInitializer() = default;
        ~RandomInitializer() = default;

        [[nodiscard]] std::vector<cv::Vec<float, 5>> initialize(const cv::Mat& samples, int k) const override final;
    };

} // namespace kmeans::clustering
