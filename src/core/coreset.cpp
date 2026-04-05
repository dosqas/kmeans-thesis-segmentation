#include "core/coreset.hpp"

#include <iostream>
#include <random>

#include <opencv2/opencv.hpp>

#include "common/constants.hpp"

namespace kmeans::core {

Coreset buildCoresetFromFrame(const cv::Mat& frame) {
    Coreset coreset;

    int rows = frame.rows;
    int cols = frame.cols;
    int total_pixels = rows * cols;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> row_dist(0, rows - 1);
    std::uniform_int_distribution<> col_dist(0, cols - 1);

    coreset.points.reserve(constants::SAMPLE_COUNT);

    for (int i = 0; i < constants::SAMPLE_COUNT; ++i) {
        int r = row_dist(gen);
        int c = col_dist(gen);
        cv::Vec3b pixel = frame.at<cv::Vec3b>(r, c);

        CoresetPoint pt;
        pt.bgr = cv::Vec3f(pixel[0], pixel[1], pixel[2]);
        pt.weight = static_cast<float>(total_pixels) / static_cast<float>(constants::SAMPLE_COUNT);
        pt.x = static_cast<float>(c) / static_cast<float>(cols);
        pt.y = static_cast<float>(r) / static_cast<float>(rows);

        coreset.points.push_back(pt);
    }

    return coreset;
}

Coreset mergeCoresets(const Coreset& A, const Coreset& B) {
    Coreset merged;

    merged.points.reserve(A.points.size() + B.points.size());
    merged.points.insert(merged.points.end(), A.points.begin(), A.points.end());
    merged.points.insert(merged.points.end(), B.points.begin(), B.points.end());

    // If the merged coreset exceeds the sample size, randomly downsample it
    if (merged.points.size() > constants::SAMPLE_COUNT) {
        std::shuffle(merged.points.begin(), merged.points.end(), std::mt19937{std::random_device{}()});
        merged.points.resize(constants::SAMPLE_COUNT);
    }

    return merged;
}

} // namespace kmeans::core