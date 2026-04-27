cmake_minimum_required(VERSION 3.14)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FetchDeps)

option(RK3566_PLATFORM "Build for RK3566 platform with RKNN SDK (local build on device)" OFF)
option(ANDROID_BUILD "Build for Android platform using NDK (cross-compile)" OFF)

# Android 交叉编译时加载 NDK
if(ANDROID_BUILD)
    if(NOT CMAKE_TOOLCHAIN_FILE)
        include(DownloadNDK)
    endif()
    message(STATUS "Building for Android platform (cross-compile)")
endif()

# RK3566 本地编译（在设备上执行，非交叉编译）
if(RK3566_PLATFORM)
    if(CMAKE_TOOLCHAIN_FILE)
        message(WARNING "RK3566_PLATFORM should be used for LOCAL build on the device, not cross-compile. "
                        "If you want Android cross-compile, use ANDROID_BUILD=ON instead. "
                        "To force local build, set -DCMAKE_TOOLCHAIN_FILE=\"\"")
    endif()
    message(STATUS "Building for RK3566 platform with RKNN SDK (local build)")
else()
    message(STATUS "Building with stub mode (non-RK3566 platform)")
endif()

project(difference_detection VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTS "Build unit tests" ON)

# Android 交叉编译时禁用测试（系统 GTest 不兼容）
if(CMAKE_TOOLCHAIN_FILE AND BUILD_TESTS)
    message(STATUS "Disabling tests for cross-compile build")
    set(BUILD_TESTS OFF CACHE BOOL "Tests disabled for cross-compile" FORCE)
endif()

if(RK3566_PLATFORM)
    add_definitions(-DRK3566_PLATFORM)
endif()

# 初始化核心依赖
EnsureSubmodule(third-party/opencv)
EnsureSubmodule(third-party/yaml-cpp)

# 引入 yaml-cpp
add_subdirectory(third-party/yaml-cpp)

include(BuildOpenCV)

# CURL：本地编译用系统库，交叉编译时自己编译
if(CMAKE_TOOLCHAIN_FILE)
    # 交叉编译：需要自己编译 curl
    include(BuildCURL)
    set(CURL_EXTERNAL_DEPS curl_external)
else()
    # 本地编译：使用系统 curl
    find_package(CURL REQUIRED)
    set(CURL_EXTERNAL_DEPS "")
endif()

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
add_dependencies(difference_detection opencv_external ${CURL_EXTERNAL_DEPS})

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
        HINTS "${RKNN_SDK_PATH}/rknpu2/runtime/Linux/librknn_api/include"
              "${RKNN_SDK_PATH}/rknn_runtime/include"
              "${RKNN_SDK_PATH}/runtime/include"
              "${RKNN_SDK_PATH}/include"
    )

    find_library(RKNN_LIBRARY rknnrt
        HINTS "${RKNN_SDK_PATH}/rknpu2/runtime/Linux/librknn_api/aarch64"
              "${RKNN_SDK_PATH}/rknpu2/runtime/Linux/librknn_api/armhf"
              "${RKNN_SDK_PATH}/rknn_runtime/lib/linux/arm64"
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