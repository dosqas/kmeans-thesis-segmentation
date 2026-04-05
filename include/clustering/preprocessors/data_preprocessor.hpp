#pragma once

#include <opencv2/core.hpp>

namespace kmeans::clustering {

    /**
     * @brief Strategy Design Pattern interface for Data Preprocessors.
     * 
     * Allows dynamic swapping between full data processing and coreset 
     * reduction techniques at runtime.
     */
    class DataPreprocessor {
    public:
        virtual ~DataPreprocessor() = default;

        /** @brief Prepares the raw image frame into clustering samples. */
        [[nodiscard]] virtual cv::Mat prepare(const cv::Mat& frame) = 0;
        
        /** @brief Resets any accumulated memory or state caches. */
        virtual void reset() {}
    };

} // namespace kmeans::clustering
