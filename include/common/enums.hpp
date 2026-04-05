#pragma once

namespace kmeans::common {

enum class DataStrategy { FULL_DATA, RCC_TREES };

enum class InitializationType { RANDOM, KMEANS_PLUSPLUS };

enum class AlgorithmType { KMEANS_REGULAR, KMEANS_QUANTUM };

} // namespace kmeans::common
