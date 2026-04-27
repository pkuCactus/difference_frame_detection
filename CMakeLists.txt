cmake_minimum_required(VERSION 3.14)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FetchDeps)

project(difference_detection VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTS "Build unit tests" ON)
option(RK3566_PLATFORM "Build for RK3566 platform with RKNN SDK" OFF)

if(RK3566_PLATFORM)
    add_definitions(-DRK3566_PLATFORM)
    message(STATUS "Building for RK3566 platform with RKNN SDK")
else()
    message(STATUS "Building with stub mode (non-RK3566 platform)")
endif()

# 初始化核心依赖
EnsureSubmodule(third-party/opencv)
EnsureSubmodule(third-party/yaml-cpp)

# 引入 yaml-cpp
add_subdirectory(third-party/yaml-cpp)

find_package(CURL REQUIRED)

set(SOURCES
    src/common/config.cpp
    src/common/logger.cpp
    src/common/performance_stats.cpp
    src/core/state_machine.cpp
    src/core/pipeline.cpp
    src/rtsp/rtsp_client.cpp
    src/rtsp/rtsp_validator.cpp
    src/decoder/frame_decoder.cpp
    src/camera/capability_checker.cpp
    src/camera/detection_reader.cpp
    src/detection/rknn_detector.cpp
    src/detection/rknn_adapter.cpp
    src/detection/yolo_postprocess.cpp
    src/tracking/byte_tracker.cpp
    src/analysis/frame_diff.cpp
    src/analysis/similarity.cpp
    src/analysis/event_analyzer.cpp
    src/utils/frame_queue.cpp
)

add_executable(difference_detection src/main.cpp ${SOURCES})

target_include_directories(difference_detection PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
)

target_link_libraries(difference_detection PRIVATE
    ${OpenCV_LIBS}
    ${CURL_LIBRARIES}
    yaml-cpp
)

if(RK3566_PLATFORM)
    find_library(RKNN_LIBRARY rknn_api HINTS /usr/lib /lib)
    if(RKNN_LIBRARY)
        target_link_libraries(difference_detection PRIVATE ${RKNN_LIBRARY})
        message(STATUS "Found RKNN library: ${RKNN_LIBRARY}")
    else()
        message(WARNING "RKNN library not found, building without real RKNN support")
    endif()
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()