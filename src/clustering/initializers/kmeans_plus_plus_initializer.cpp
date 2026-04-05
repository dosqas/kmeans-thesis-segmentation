#include "clustering/initializers/kmeans_plus_plus_initializer.hpp"

#include <limits>
#include <random>

namespace kmeans::clustering {

std::vector<cv::Vec<float, 5>> KMeansPlusPlusInitializer::initialize(const cv::Mat& samples, int k) const {
    std::vector<cv::Vec<float, 5>> centers;
    centers.reserve(k);
    int numPoints = samples.rows;

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> dis(0, numPoints - 1);

    // First center random
    int firstIdx = dis(gen);
    const float* firstPtr = samples.ptr<float>(firstIdx);
    centers.emplace_back(firstPtr[0], firstPtr[1], firstPtr[2], firstPtr[3], firstPtr[4]);

    std::vector<float> distances(numPoints, std::numeric_limits<float>::max());

    for (int i = 1; i < k; ++i) {
        float sumDistSq = 0.0f;

        for (int p = 0; p < numPoints; ++p) {
            const float* pPtr = samples.ptr<float>(p);
            float currDistSq = 0.0f;
            const auto& lastCenter = centers.back();

            for (int d = 0; d < 5; ++d) {
                float diff = pPtr[d] - lastCenter[d];
                currDistSq += diff * diff;
            }

            if (currDistSq < distances[p]) {
                distances[p] = currDistSq;
            }
            sumDistSq += distances[p];
        }

        std::uniform_real_distribution<float> fdis(0.0f, sumDistSq);
        float target = fdis(gen);
        float cumulative = 0.0f;
        int selectedIdx = numPoints - 1;

        for (int p = 0; p < numPoints; ++p) {
            cumulative += distances[p];
            if (cumulative >= target) {
                selectedIdx = p;
                break;
            }
        }

        const float* selPtr = samples.ptr<float>(selectedIdx);
        centers.emplace_back(selPtr[0], selPtr[1], selPtr[2], selPtr[3], selPtr[4]);
    }

    return centers;
}

} // namespace kmeans::clustering
