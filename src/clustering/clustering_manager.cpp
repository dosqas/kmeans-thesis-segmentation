#include "clustering/clustering_manager.hpp"
#include "common/enums.hpp"
#include "common/config.hpp"
#include "common/constants.hpp"
#include "opencv2/core.hpp"

#include "clustering/flat_data_preprocessor.hpp"
#include "clustering/rcc_data_preprocessor.hpp"
#include "clustering/random_initializer.hpp"
#include "clustering/kmeans_plus_plus_initializer.hpp"
#include "clustering/classical_engine.hpp"

namespace kmeans {
namespace clustering {

    ClusteringManager::ClusteringManager() {
        updateStategyImplementations();
    }

    void ClusteringManager::updateStategyImplementations() {
        // Data Preparation Strategy
        if (m_config.strategy == DataStrategy::RCC_TREES) {
            if (!dynamic_cast<RccDataPreprocessor*>(m_dataPreprocessor.get())) {
                m_dataPreprocessor = std::make_unique<RccDataPreprocessor>();
            }
        } else {
            if (!dynamic_cast<FlatDataPreprocessor*>(m_dataPreprocessor.get())) {
                m_dataPreprocessor = std::make_unique<FlatDataPreprocessor>();
            }
        }

        // Initializer
        if (m_config.init == InitializationType::KMEANS_PLUSPLUS) {
            if (!dynamic_cast<KMeansPlusPlusInitializer*>(m_initializer.get())) {
                m_initializer = std::make_unique<KMeansPlusPlusInitializer>();
            }
        } else {
            if (!dynamic_cast<RandomInitializer*>(m_initializer.get())) {
                m_initializer = std::make_unique<RandomInitializer>();
            }
        }

        // Execution Engine
        // if (m_config.algorithm == AlgorithmType::QUANTUM) { ... } else {
            if (!dynamic_cast<ClassicalEngine*>(m_clusteringEngine.get())) {
                m_clusteringEngine = std::make_unique<ClassicalEngine>();
            }
        // }
    }

    cv::Mat ClusteringManager::segmentFrame(const cv::Mat& frame) {
        std::vector<cv::Vec<float, 5>> centers = computeCenters(frame);

        if (!m_cudaContext || m_cudaContext->getWidth() != frame.cols || m_cudaContext->getK() != m_config.k) {
            m_cudaContext = std::make_unique<CudaAssignmentContext>(frame.cols, frame.rows, m_config.k);
        }

        cv::Mat result(frame.rows, frame.cols, CV_8UC3);

        m_cudaContext->run(
            frame,
            centers,
            result
        );

        return result;
    }

    std::vector<cv::Vec<float, 5>> ClusteringManager::computeCenters(const cv::Mat& frame) {
        updateStategyImplementations(); // Allow hot-swapping if config changed

        bool shouldUpdate = (m_frameCount % m_config.learningInterval == 0) || !m_hasPrevious;
        m_frameCount++;

        if (!shouldUpdate && m_hasPrevious) {
            return m_previousCenters;
        }

        cv::Mat samples = m_dataPreprocessor->prepare(frame);

        std::vector<cv::Vec<float, 5>> initialCenters;
        if (m_hasPrevious && m_previousCenters.size() == m_config.k) {
            initialCenters = m_previousCenters;
        } else {
            initialCenters = m_initializer->initialize(samples, m_config.k);
        }

        std::vector<cv::Vec<float, 5>> finalCenters = m_clusteringEngine->run(samples, initialCenters, m_config.k);

        m_previousCenters = finalCenters;
        m_hasPrevious = true;

        return m_previousCenters;
    }

}
}