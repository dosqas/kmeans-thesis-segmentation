#include "clustering/engines/quantum_engine.hpp"
#include <iostream>

namespace kmeans {

    QuantumEngine::QuantumEngine() : m_connected(false) {
        try {
            m_context = std::make_unique<zmq::context_t>(1);
            m_socket = std::make_unique<zmq::socket_t>(*m_context, zmq::socket_type::req);
            
            // Long timeouts so Python Quantum Simulator has ample time to boot and process batches
            m_socket->set(zmq::sockopt::rcvtimeo, 10000); 
            m_socket->set(zmq::sockopt::sndtimeo, 10000);
            
            // Connect to the local Python Qiskit ZeroMQ Server
            m_socket->connect("tcp://127.0.0.1:5555");
            m_connected = true;
            std::cout << "[QuantumEngine] Connected to Python IPC Server on tcp://127.0.0.1:5555" << std::endl;
        } catch (const zmq::error_t& e) {
            std::cerr << "QuantumEngine IPC ZMQ Error: " << e.what() << std::endl;
        }
    }

    QuantumEngine::~QuantumEngine() {
        if (m_socket) {
            m_socket->close();
        }
        if (m_context) {
            m_context->close();
        }
    }

    std::vector<cv::Vec<float, 5>> QuantumEngine::run(
        const cv::Mat& samples,
        const std::vector<cv::Vec<float, 5>>& initialCenters,
        int k) 
    {
        if (!m_connected) return initialCenters;

        int32_t n = samples.rows;
        int32_t k32 = k;
        
        // 1. Pack data: [N, K, Samples..., Centers...]
        size_t bytesN = sizeof(int32_t);
        size_t bytesK = sizeof(int32_t);
        size_t bytesSamples = n * 5 * sizeof(float);
        size_t bytesCenters = k * 5 * sizeof(float);
        
        size_t totalSize = bytesN + bytesK + bytesSamples + bytesCenters;
        zmq::message_t req(totalSize);
        
        char* ptr = (char*)req.data();
        memcpy(ptr, &n, bytesN); ptr += bytesN;
        memcpy(ptr, &k32, bytesK); ptr += bytesK;
        
        if(samples.isContinuous()) {
            memcpy(ptr, samples.ptr<float>(0), bytesSamples);
            ptr += bytesSamples;
        } else {
            // Fallback for non-continuous Mats
            for(int i = 0; i < n; i++) {
                memcpy(ptr, samples.ptr<float>(i), 5 * sizeof(float));
                ptr += 5 * sizeof(float);
            }
        }
        
        memcpy(ptr, initialCenters.data(), bytesCenters);

        // 2. Send Data
        try {
            auto res = m_socket->send(req, zmq::send_flags::none);
            if (!res) {
                // Return fallback old centers if the network timeout popped
                return initialCenters; 
            }

            // 3. Receive new centers
            zmq::message_t reply;
            auto rec_res = m_socket->recv(reply, zmq::recv_flags::none);
            
            if (rec_res && reply.size() == bytesCenters) {
                std::vector<cv::Vec<float, 5>> newCenters(k);
                memcpy(newCenters.data(), reply.data(), bytesCenters);
                return newCenters;
            }
        } catch (const zmq::error_t& e) {
            std::cerr << "ZMQ Exception during transport: " << e.what() << " - Reconnecting..." << std::endl;
            m_socket = std::make_unique<zmq::socket_t>(*m_context, zmq::socket_type::req);
            m_socket->set(zmq::sockopt::rcvtimeo, 10000); 
            m_socket->set(zmq::sockopt::sndtimeo, 10000);
            m_socket->set(zmq::sockopt::linger, 0);
            m_socket->connect("tcp://127.0.0.1:5555");
            return initialCenters;
        } catch (const std::exception& e) {
            std::cerr << "Standard Exception during transport: " << e.what() << std::endl;
            return initialCenters;
        } catch (...) {
            std::cerr << "Unknown exception during transport" << std::endl;
            return initialCenters;
        }

        return initialCenters;
    }
}
