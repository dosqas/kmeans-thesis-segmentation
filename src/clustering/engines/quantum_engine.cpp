#include "clustering/engines/quantum_engine.hpp"

#include <iostream>
#include <stdexcept>

#include "common/constants.hpp"

namespace kmeans::clustering {

QuantumEngine::QuantumEngine() : m_context(1), m_socket(m_context, zmq::socket_type::req) {
    m_socket.set(zmq::sockopt::rcvtimeo, constants::IPC_TIMEOUT_MS);
    m_socket.set(zmq::sockopt::sndtimeo, constants::IPC_TIMEOUT_MS);

    try {
        m_socket.connect(constants::IPC_SOCKET);
        m_connected = true;
    } catch (const zmq::error_t& e) {
        std::cerr << "QuantumEngine ZMQ connect error: " << e.what() << std::endl;
        m_connected = false;
    }
}

std::vector<cv::Vec<float, 5>> QuantumEngine::run(const cv::Mat& samples,
                                                  const std::vector<cv::Vec<float, 5>>& initialCenters, int k) {
    if (!m_connected) {
        std::cerr << "Warning: ZMQ Socket disconnected, falling back" << std::endl;
        return initialCenters;
    }

    int n = samples.rows;
    size_t samples_bytes = n * 5 * sizeof(float);
    size_t centers_bytes = k * 5 * sizeof(float);
    size_t header_bytes = 2 * sizeof(int);
    size_t total_size = header_bytes + samples_bytes + centers_bytes;

    zmq::message_t request(total_size);
    char* ptr = static_cast<char*>(request.data());

    // Header
    std::memcpy(ptr, &n, sizeof(int));
    ptr += sizeof(int);
    std::memcpy(ptr, &k, sizeof(int));
    ptr += sizeof(int);

    // Samples
    for (int i = 0; i < n; ++i) {
        const float* row = samples.ptr<float>(i);
        std::memcpy(ptr, row, 5 * sizeof(float));
        ptr += 5 * sizeof(float);
    }

    // Centers
    for (int i = 0; i < k; ++i) {
        std::memcpy(ptr, initialCenters[i].val, 5 * sizeof(float));
        ptr += 5 * sizeof(float);
    }

    try {
        if (!m_socket.send(request, zmq::send_flags::none)) {
            return initialCenters;
        }

        zmq::message_t reply;
        if (!m_socket.recv(reply, zmq::recv_flags::none)) {
            return initialCenters;
        }

        if (reply.size() == 3 && std::memcmp(reply.data(), "ERR", 3) == 0) {
            return initialCenters; // Python engine error
        }

        if (reply.size() != centers_bytes) {
            return initialCenters;
        }

        std::vector<cv::Vec<float, 5>> new_centers(k);
        const float* rep_ptr = static_cast<const float*>(reply.data());
        for (int i = 0; i < k; ++i) {
            for (int d = 0; d < 5; ++d) {
                new_centers[i][d] = rep_ptr[i * 5 + d];
            }
        }
        return new_centers;

    } catch (const zmq::error_t& e) {
        std::cerr << "ZMQ recv exception: " << e.what() << std::endl;
        return initialCenters;
    }
}

} // namespace kmeans::clustering
