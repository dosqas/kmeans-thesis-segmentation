#include "clustering/engines/classical_engine.hpp"
#include <limits>

namespace kmeans {

    std::vector<cv::Vec<float, 5>> ClassicalEngine::run(
        const cv::Mat& samples,
        const std::vector<cv::Vec<float, 5>>& initialCenters,
        int k)
    {
        std::vector<cv::Vec<float, 5>> centers = initialCenters;
        int numPoints = samples.rows;
        std::vector<int> labels(numPoints, -1);

        for (int iter = 0; iter < 20; ++iter) {
            bool changed = false;

            // --- STEP 1: ASSIGNMENT ---
            for (int i = 0; i < numPoints; ++i) {
                float minDistSq = std::numeric_limits<float>::max();
                int bestK = 0;

                // FIX: Get pointer to the start of the row and map it to a Vec5f
                const float* rowPtr = samples.ptr<float>(i);
                cv::Vec<float, 5> point(rowPtr[0], rowPtr[1], rowPtr[2], rowPtr[3], rowPtr[4]);

                for (int j = 0; j < k; ++j) {
                    // Manually calculate squared distance to avoid any cv::norm mismatch
                    float d2 = 0;
                    for (int d = 0; d < 5; ++d) {
                        float diff = point[d] - centers[j][d];
                        d2 += diff * diff;
                    }

                    if (d2 < minDistSq) {
                        minDistSq = d2;
                        bestK = j;
                    }
                }

                if (labels[i] != bestK) {
                    labels[i] = bestK;
                    changed = true;
                }
            }

            if (!changed) break;

            // --- STEP 2: UPDATE ---
            std::vector<cv::Vec<float, 5>> newSums(k, cv::Vec<float, 5>(0, 0, 0, 0, 0));
            std::vector<int> counts(k, 0);

            for (int i = 0; i < numPoints; ++i) {
                int label = labels[i];
                const float* rowPtr = samples.ptr<float>(i);

                for (int d = 0; d < 5; ++d) {
                    newSums[label][d] += rowPtr[d];
                }
                counts[label]++;
            }

            for (int j = 0; j < k; ++j) {
                if (counts[j] > 0) {
                    centers[j] = newSums[j] / (float)counts[j];
                } else {
                    // Empty cluster detected!
                    // This happens when sudden color changes leave a center too far from any new data points.
                    // We reinitialize it to a random data point to pull it back into the active color space.
                    int randIdx = std::rand() % numPoints;
                    const float* randPtr = samples.ptr<float>(randIdx);
                    centers[j] = cv::Vec<float, 5>(randPtr[0], randPtr[1], randPtr[2], randPtr[3], randPtr[4]);
                    changed = true; // Force another iteration to integrate the resurrected cluster
                }
            }
        }

        return centers;
    }

} // namespace kmeans