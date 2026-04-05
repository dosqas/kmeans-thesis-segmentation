#include "io/application.hpp"
#include <iostream>

#include "core/coreset.hpp"
#include "common/config.hpp"
#include <opencv2/core/utils/logger.hpp>

using namespace kmeans;

int main(int argc, char** argv) {
    try {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    std::cout << "Starting K-Means Clustering for Thesis - ImGui Application..." << std::endl;
    
        io::Application app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal UI Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
