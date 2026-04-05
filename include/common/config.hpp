#pragma once
#include "common/enums.hpp"

namespace kmeans::common {
    struct SegmentationConfig {
        DataStrategy strategy = DataStrategy::RCC_TREES;
        InitializationType init = InitializationType::KMEANS_PLUSPLUS;
        AlgorithmType algorithm = AlgorithmType::KMEANS_REGULAR;
        
        int k = 3;
        int learningInterval = 15;
    };
}
