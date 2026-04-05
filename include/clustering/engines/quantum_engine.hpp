#pragma once

#include "clustering/engines/kmeans_engine.hpp"
#include <zmq.hpp>
#include <memory>

namespace kmeans::clustering {

    class QuantumEngine final : public KMeansEngine {
    private:
        zmq::context_t m_context;
        zmq::socket_t m_socket;
        bool m_connected = false;

    public:
        QuantumEngine();
        ~QuantumEngine() = default;

        [[nodiscard]] std::vector<cv::Vec<float, 5>> run(
            const cv::Mat& samples,
            const std::vector<cv::Vec<float, 5>>& initialCenters,
            int k) override final;
    };

} // namespace kmeans::clustering
