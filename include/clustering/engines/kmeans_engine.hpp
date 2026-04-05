#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace kmeans::clustering {

    /**
     * @brief Strategy Design Pattern interface for Execution Engines.
     * 
     * Allows seamless transition between classical CPU execution and 
     * external quantum simulation environments without affecting the consumer.
     */
    class KMeansEngine {
    public:
        virtual ~KMeansEngine() = default;

        /** @brief Executes the clustering algorithm using the specific engine implementation. */
        [[nodiscard]] virtual std::vector<cv::Vec<float, 5>> run(
            const cv::Mat& samples,
            const std::vector<cv::Vec<float, 5>>& initialCenters,
            int k) = 0;
    };

} // namespace kmeans::clustering
