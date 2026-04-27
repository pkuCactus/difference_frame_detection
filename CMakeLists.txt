cmake_minimum_required(VERSION 3.14)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FetchDeps)

option(RK3566_PLATFORM "Build for RK3566 platform with RKNN SDK" OFF)

if(RK3566_PLATFORM)
    if(NOT CMAKE_TOOLCHAIN_FILE)
        include(DownloadNDK)
    endif()
    message(STATUS "Building for RK3566 platform with RKNN SDK")
else()
    message(STATUS "Building with stub mode (non-RK3566 platform)")
endif()

project(difference_detection VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTS "Build unit tests" ON)

if(RK3566_PLATFORM)
    add_definitions(-DRK3566_PLATFORM)
endif()

# 初始化核心依赖
EnsureSubmodule(third-party/opencv)
EnsureSubmodule(third-party/yaml-cpp)

# 引入 yaml-cpp
add_subdirectory(third-party/yaml-cpp)

include(BuildOpenCV)
include(BuildCURL)

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
add_dependencies(difference_detection opencv_external curl_external)

target_include_directories(difference_detection PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
)

target_link_libraries(difference_detection PRIVATE
    ${OpenCV_LIBS}
    ${CURL_LIBRARIES}
    yaml-cpp
    z
    pthread
)

if(RK3566_PLATFORM)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third-party/rknn-sdk")
        message(STATUS "Attempting to initialize rknn-sdk submodule...")
        execute_process(
            COMMAND git submodule update --init -- third-party/rknn-sdk
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            RESULT_VARIABLE result
            OUTPUT_VARIABLE output
            ERROR_VARIABLE error
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE
        )
        if(NOT result EQUAL 0)
            message(WARNING "Failed to initialize rknn-sdk submodule:\n${error}")
        endif()
    endif()

    set(RKNN_SDK_PATH "${CMAKE_SOURCE_DIR}/third-party/rknn-sdk" CACHE PATH "Path to RKNN SDK")

    find_path(RKNN_INCLUDE_DIR rknn_api.h
        HINTS "${RKNN_SDK_PATH}/rknn_runtime/include"
              "${RKNN_SDK_PATH}/runtime/include"
              "${RKNN_SDK_PATH}/include"
    )

    find_library(RKNN_LIBRARY rknn_api
        HINTS "${RKNN_SDK_PATH}/rknn_runtime/lib/linux/arm64"
              "${RKNN_SDK_PATH}/runtime/lib/linux/arm64"
              "${RKNN_SDK_PATH}/lib"
              /usr/lib /lib
    )

    if(RKNN_INCLUDE_DIR)
        target_include_directories(difference_detection PRIVATE ${RKNN_INCLUDE_DIR})
        message(STATUS "Found RKNN include dir: ${RKNN_INCLUDE_DIR}")
    endif()

    if(RKNN_LIBRARY)
        target_link_libraries(difference_detection PRIVATE ${RKNN_LIBRARY})
        message(STATUS "Found RKNN library: ${RKNN_LIBRARY}")
    else()
        message(WARNING "RKNN library not found, building without real RKNN support")
    endif()
endif()

if(BUILD_TESTS)
    enable_testing()
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third-party/googletest/CMakeLists.txt")
        message(STATUS "googletest submodule not present, attempting to initialize...")
        execute_process(
            COMMAND git submodule update --init -- third-party/googletest
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            RESULT_VARIABLE result
        )
    endif()
    if(EXISTS "${CMAKE_SOURCE_DIR}/third-party/googletest/CMakeLists.txt")
        add_subdirectory(third-party/googletest)
    else()
        find_package(GTest REQUIRED)
    endif()
    add_subdirectory(tests)
endif()