cmake_minimum_required(VERSION 3.14)

find_package(GTest REQUIRED)
find_package(CURL REQUIRED)

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
    ${CMAKE_SOURCE_DIR}/src/common/config.cpp
    ${CMAKE_SOURCE_DIR}/src/common/logger.cpp
    ${CMAKE_SOURCE_DIR}/src/common/performance_stats.cpp
    ${CMAKE_SOURCE_DIR}/src/core/state_machine.cpp
    ${CMAKE_SOURCE_DIR}/src/rtsp/rtsp_client.cpp
    ${CMAKE_SOURCE_DIR}/src/rtsp/rtsp_validator.cpp
    ${CMAKE_SOURCE_DIR}/src/decoder/frame_decoder.cpp
    ${CMAKE_SOURCE_DIR}/src/camera/capability_checker.cpp
    ${CMAKE_SOURCE_DIR}/src/camera/detection_reader.cpp
    ${CMAKE_SOURCE_DIR}/src/detection/rknn_detector.cpp
    ${CMAKE_SOURCE_DIR}/src/detection/rknn_adapter.cpp
    ${CMAKE_SOURCE_DIR}/src/detection/yolo_postprocess.cpp
    ${CMAKE_SOURCE_DIR}/src/tracking/byte_tracker.cpp
    ${CMAKE_SOURCE_DIR}/src/analysis/frame_diff.cpp
    ${CMAKE_SOURCE_DIR}/src/analysis/similarity.cpp
    ${CMAKE_SOURCE_DIR}/src/analysis/event_analyzer.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/frame_queue.cpp
)

add_executable(difference_detection_tests ${TEST_SOURCES})

target_include_directories(difference_detection_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
    /home/jianzhong/miniconda3/include
)

target_link_libraries(difference_detection_tests PRIVATE
    ${OpenCV_LIBS}
    ${CURL_LIBRARIES}
    /home/jianzhong/miniconda3/lib/libyaml-cpp.so
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME difference_detection_tests COMMAND difference_detection_tests)