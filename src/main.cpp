//#include <iostream>
//#include <vector>
//#include <string>
//
//#include "core/coreset.hpp"
//#include "common/config.hpp"
//#include "io/application.hpp"
//#include <opencv2/core/utils/logger.hpp>
//
//using namespace kmeans;
//
//// Entry point
//int main() 
//{
//    // Suppress verbose OpenCV backend detection logs (e.g. GSTREAMER FAILED, TBB FAILED)
//    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
//
//    std::cout << "Starting K-Means Clustering for Thesis - ImGui Application..." << std::endl;
//    
//    io::Application app;
//    app.run();
//    
//    return 0;
//}

#include <qiskit/qiskit.h>
#include <qiskit_ibm_runtime.h>

#include <iostream>

int main() {
  // Test 1: Core SDK
  qiskit::Circuit circuit("Thesis_Circuit");
  circuit.h(0);  // Hadamard gate
  std::cout << "Quantum Circuit initialized on D: drive!" << std::endl;

  // Test 2: Runtime Service
  try {
    qiskit::service::QiskitRuntimeService service;
    std::cout << "IBM Runtime Service linked successfully!" << std::endl;
  } catch (...) {
    // We don't care if it fails to connect yet, just that it CALLS the function
    std::cout << "Runtime Service call successful (Linker confirmed)."
              << std::endl;
  }

  return 0;
}