#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace kmeans::clustering {

    /**
     * @brief Strategy Design Pattern interface for Centroid Initializers.
     * 
     * Enables swapping between different initialization algorithms like 
     * Random Initialization and K-Means++ at runtime.
     */
    class Initializer {
    public:
        virtual ~Initializer() = default;

        /** @brief Calculates initial cluster centers from the provided samples. */
        [[nodiscard]] virtual std::vector<cv::Vec<float, 5>> initialize(const cv::Mat& samples, int k) const = 0;
    };

} // namespace kmeans::clustering
