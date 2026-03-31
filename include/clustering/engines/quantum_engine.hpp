#pragma once
#include "clustering/engines/kmeans_engine.hpp"
#include <zmq.hpp>
#include <memory>

namespace kmeans {

    class QuantumEngine : public KMeansEngine {
    private:
        std::unique_ptr<zmq::context_t> m_context;
        std::unique_ptr<zmq::socket_t> m_socket;
        bool m_connected;

    public:
        QuantumEngine();
        ~QuantumEngine() override;

        std::vector<cv::Vec<float, 5>> run(
            const cv::Mat& samples,
            const std::vector<cv::Vec<float, 5>>& initialCenters,
            int k) override;
    };
}
