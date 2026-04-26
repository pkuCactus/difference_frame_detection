cmake_minimum_required(VERSION 3.14)
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

find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/home/jianzhong/miniconda3/lib/cmake/yaml-cpp")
find_package(yaml-cpp REQUIRED)

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
    /home/jianzhong/miniconda3/include
)

target_link_libraries(difference_detection PRIVATE
    ${OpenCV_LIBS}
    ${CURL_LIBRARIES}
    /home/jianzhong/miniconda3/lib/libyaml-cpp.so
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