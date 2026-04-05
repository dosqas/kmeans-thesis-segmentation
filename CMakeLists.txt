cmake_minimum_required(VERSION 3.18)
project(KMeansThesisSegmentation LANGUAGES CXX CUDA)

# Dependencies
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(cppzmq CONFIG REQUIRED)
find_package(glfw3 REQUIRED)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/vendor/imgui
    ${CMAKE_SOURCE_DIR}/src/vendor/imgui/backends
)

# Core Logic Library
add_library(kmeans_core
    src/backend/cuda_kernels.cu
    src/clustering/engines/classical_engine.cpp
    src/clustering/engines/quantum_engine.cpp
    src/clustering/clustering_manager.cpp
    src/clustering/clustering_factory.cpp
    src/clustering/preprocessors/full_data_preprocessor.cpp
    src/clustering/preprocessors/rcc_data_preprocessor.cpp
    src/clustering/initializers/random_initializer.cpp
    src/clustering/initializers/kmeans_plus_plus_initializer.cpp
    src/core/coreset.cpp
    src/core/rcc.cpp
    src/common/utils.cpp
    src/io/application.cpp
)

set_source_files_properties(src/backend/cuda_kernels.cu PROPERTIES LANGUAGE CUDA)
set_target_properties(kmeans_core PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CUDA_SEPARABLE_COMPILATION ON       # Enable separable compilation for CUDA (different from .cpp files)
    CUDA_STANDARD 17                    # Use C++14 standard for CUDA files
    CUDA_STANDARD_REQUIRED ON           # Enforce the use of the specified C++ standard
    CUDA_ARCHITECTURES "native"         # Specify target architecture (native for the current machine)
)

target_link_libraries(kmeans_core PUBLIC ${OpenCV_LIBS} glfw cppzmq)

# ImGui Library
add_library(imgui STATIC
    src/vendor/imgui/imgui.cpp
    src/vendor/imgui/imgui_draw.cpp
    src/vendor/imgui/imgui_tables.cpp
    src/vendor/imgui/imgui_widgets.cpp
    src/vendor/imgui/backends/imgui_impl_glfw.cpp
    src/vendor/imgui/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC src/vendor/imgui src/vendor/imgui/backends)
target_link_libraries(imgui PUBLIC glfw)
if (WIN32)
    target_link_libraries(imgui PUBLIC opengl32)
endif()

# Bind ImGui exactly into the core logic library so application.cpp can inherit the headers
target_link_libraries(kmeans_core PUBLIC imgui)

# Main Executable
add_executable(kmeans_thesis_segmentation src/main.cpp)

# Set CUDA properties for the main executable
set_target_properties(kmeans_thesis_segmentation PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "native"
)

target_link_libraries(kmeans_thesis_segmentation PRIVATE kmeans_core imgui)
