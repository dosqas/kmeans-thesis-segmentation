#pragma once

#include "clustering/preprocessors/data_preprocessor.hpp"
#include "core/rcc.hpp"
#include <vector>

namespace kmeans::clustering {

    class RccDataPreprocessor final : public DataPreprocessor {
    private:
        int m_frameCount = 0;
        core::Coreset m_currentCoreset;
        const int m_rebuildInterval = 60;

    public:
        RccDataPreprocessor() = default;
        ~RccDataPreprocessor() = default;

        [[nodiscard]] cv::Mat prepare(const cv::Mat& frame) override final;
        
        void reset() override final;
    };

} // namespace kmeans::clustering
