#include <iostream>
#include <vector>
#include <string>

#include "core/coreset.hpp"
#include "common/config.hpp"
#include "io/application.hpp"
#include <opencv2/core/utils/logger.hpp>

using namespace kmeans;

// Entry point
int main() 
{
    // Suppress verbose OpenCV backend detection logs (e.g. GSTREAMER FAILED, TBB FAILED)
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    std::cout << "Starting K-Means Clustering for Thesis - ImGui Application..." << std::endl;
    
    io::Application app;
    app.run();
    
    return 0;
}
