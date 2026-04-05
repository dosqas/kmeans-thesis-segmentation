#pragma once

namespace kmeans::constants {
constexpr int VIDEO_WIDTH = 640;
constexpr int VIDEO_HEIGHT = 480;
constexpr float COLOR_SCALE = 1.0f;
constexpr float SPATIAL_SCALE = 0.5f;

constexpr int K_MIN = 2;
constexpr int K_MAX = 12;
constexpr int LEARN_INTERVAL_MIN = 1;
constexpr int LEARN_INTERVAL_MAX = 60;
constexpr int SAMPLE_COUNT = 2000;

// IPC Settings
constexpr const char* IPC_SOCKET = "tcp://127.0.0.1:5555";
constexpr int IPC_TIMEOUT_MS = 2000;
} // namespace kmeans::constants