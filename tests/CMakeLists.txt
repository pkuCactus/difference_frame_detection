cmake_minimum_required(VERSION 3.14)

find_package(GTest REQUIRED)

set(TEST_SOURCES
    test_config.cpp
    test_types.cpp
    test_logger.cpp
    test_state_machine.cpp
    test_detector.cpp
    test_tracker.cpp
    test_frame_diff.cpp
    test_similarity.cpp
    test_performance_stats.cpp
    test_frame_queue.cpp
    test_rknn_adapter.cpp
    test_rtsp_validator.cpp
    test_event_analyzer.cpp
    test_visualization.cpp
    test_image_loader.cpp
    ${CMAKE_SOURCE_DIR}/src/common/config.cpp
    ${CMAKE_SOURCE_DIR}/src/common/logger.cpp
    ${CMAKE_SOURCE_DIR}/src/common/performance_stats.cpp
    ${CMAKE_SOURCE_DIR}/src/core/state_machine.cpp
    ${CMAKE_SOURCE_DIR}/src/rtsp/rtsp_client.cpp
    ${CMAKE_SOURCE_DIR}/src/rtsp/rtsp_validator.cpp
    ${CMAKE_SOURCE_DIR}/src/decoder/frame_decoder.cpp
    ${CMAKE_SOURCE_DIR}/src/camera/capability_checker.cpp
    ${CMAKE_SOURCE_DIR}/src/camera/detection_reader.cpp
    ${CMAKE_SOURCE_DIR}/src/detection/detector.cpp
    ${CMAKE_SOURCE_DIR}/src/tracking/byte_tracker.cpp
    ${CMAKE_SOURCE_DIR}/src/analysis/frame_diff.cpp
    ${CMAKE_SOURCE_DIR}/src/analysis/similarity.cpp
    ${CMAKE_SOURCE_DIR}/src/analysis/event_analyzer.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/frame_queue.cpp
    ${CMAKE_SOURCE_DIR}/src/common/visualization.cpp
    ${CMAKE_SOURCE_DIR}/src/common/image_loader.cpp
)

if(NOT CMAKE_TOOLCHAIN_FILE)
    list(APPEND TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/rknn_api_stub.cpp)
endif()

add_executable(difference_detection_tests ${TEST_SOURCES})

if(NOT CMAKE_TOOLCHAIN_FILE)
    find_package(OpenCV REQUIRED)
endif()
if(OPENCV_FFMPEG_DEPS)
    add_dependencies(difference_detection_tests ${OPENCV_FFMPEG_DEPS})
endif()
if(CURL_EXTERNAL_DEPS)
    add_dependencies(difference_detection_tests ${CURL_EXTERNAL_DEPS})
endif()

target_include_directories(difference_detection_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
    ${RKNN_INCLUDE_DIR}
)

target_link_libraries(difference_detection_tests PRIVATE
    ${OpenCV_LIBS}
    ${CURL_LIBRARIES}
    yaml-cpp
    z
    pthread
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME difference_detection_tests COMMAND difference_detection_tests)