cmake_minimum_required(VERSION 3.14)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FetchDeps)

option(RK3566_PLATFORM "Build for RK3566 platform with RKNN SDK (local build on device)" OFF)
option(ANDROID_BUILD "Build for Android platform using NDK (cross-compile)" OFF)

# RK3566 平台使用交叉编译，依赖 Android NDK
if(RK3566_PLATFORM)
    include(DownloadNDK)
    set(CMAKE_TOOLCHAIN_FILE "${ANDROID_NDK_DIR}/build/cmake/android.toolchain.cmake" CACHE FILEPATH "" FORCE)
    message(STATUS "Configured for RK3566 platform with NDK: ${ANDROID_NDK_DIR}")
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

# nlohmann/json（header-only 库）
include(BuildJson)

# 本地编译使用系统库，交叉编译时自己构建
if(CMAKE_TOOLCHAIN_FILE)
    include(BuildFFmpeg)
    include(BuildOpenCV)
    set(OPENCV_FFMPEG_DEPS opencv_external ffmpeg_external)
else()
    # 本地编译：使用系统 OpenCV 和 FFmpeg
    find_package(OpenCV REQUIRED)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale libswresample)
    set(FFMPEG_INCLUDE_DIRS ${FFMPEG_INCLUDE_DIRS})
    set(FFMPEG_LIBRARIES ${FFMPEG_LIBRARIES})
    set(OPENCV_FFMPEG_DEPS "")
endif()

# CURL：本地编译用系统库，交叉编译时自己编译
if(RK3566_PLATFORM)
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
    src/common/visualization.cpp
    src/common/image_loader.cpp
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
if(OPENCV_FFMPEG_DEPS)
    add_dependencies(difference_detection ${OPENCV_FFMPEG_DEPS})
endif()
if(CURL_EXTERNAL_DEPS)
    add_dependencies(difference_detection ${CURL_EXTERNAL_DEPS})
endif()

target_include_directories(difference_detection PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
)

set(THREAD_LIB "")
if(NOT CMAKE_TOOLCHAIN_FILE)
    set(THREAD_LIB pthread)
endif()

set(ANDROID_SYS_LIBS "")
if(ANDROID)
    set(ANDROID_SYS_LIBS log mediandk)
endif()

if(ANDROID)
    target_link_options(difference_detection PRIVATE -Wl,--allow-shlib-undefined)
endif()

set(ANDROID_CXX_LIB "")
if(ANDROID)
    set(ANDROID_CXX_LIB c++_shared)
endif()

target_link_libraries(difference_detection PRIVATE
    ${OpenCV_LIBS}
    ${FFMPEG_LIBRARIES}
    ${CURL_LIBRARIES}
    yaml-cpp
    nlohmann_json::nlohmann_json
    z
    ${THREAD_LIB}
    ${ANDROID_SYS_LIBS}
    ${ANDROID_CXX_LIB}
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
        HINTS "${RKNN_SDK_PATH}/rknpu2/runtime/Android/librknn_api/include"
        NO_CMAKE_FIND_ROOT_PATH
    )

    find_library(RKNN_LIBRARY rknnrt
        HINTS "${RKNN_SDK_PATH}/rknpu2/runtime/Android/librknn_api/${ANDROID_ABI}"
        NO_CMAKE_FIND_ROOT_PATH
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

# 自动打包部署文件
set(DEPLOY_DIR "${CMAKE_BINARY_DIR}/deploy")

if(CMAKE_TOOLCHAIN_FILE)
    add_custom_command(TARGET difference_detection POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${DEPLOY_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:difference_detection> ${DEPLOY_DIR}/
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FFMPEG_LIBRARIES} ${DEPLOY_DIR}/
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENCV_INSTALL_DIR}/sdk/native/libs/${ANDROID_ABI}/libopencv_videoio_ffmpeg.so ${DEPLOY_DIR}/
        COMMENT "Deploying binaries and libraries to ${DEPLOY_DIR}"
    )
else()
    add_custom_command(TARGET difference_detection POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${DEPLOY_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:difference_detection> ${DEPLOY_DIR}/
        COMMENT "Deploying binary to ${DEPLOY_DIR}"
    )
endif()

if(ANDROID AND ANDROID_NDK_DIR)
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(ANDROID_CXX_TRIPLE "aarch64-linux-android")
    elseif(ANDROID_ABI STREQUAL "armeabi-v7a")
        set(ANDROID_CXX_TRIPLE "arm-linux-androideabi")
    elseif(ANDROID_ABI STREQUAL "x86")
        set(ANDROID_CXX_TRIPLE "i686-linux-android")
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(ANDROID_CXX_TRIPLE "x86_64-linux-android")
    else()
        set(ANDROID_CXX_TRIPLE "${ANDROID_ABI}")
    endif()
    set(ANDROID_CXX_SHARED "${ANDROID_NDK_DIR}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/${ANDROID_CXX_TRIPLE}/libc++_shared.so")
    if(EXISTS ${ANDROID_CXX_SHARED})
        add_custom_command(TARGET difference_detection POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ANDROID_CXX_SHARED} ${DEPLOY_DIR}/
        )
    endif()
endif()

if(RKNN_LIBRARY)
    add_custom_command(TARGET difference_detection POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${RKNN_LIBRARY} ${DEPLOY_DIR}/
    )
endif()