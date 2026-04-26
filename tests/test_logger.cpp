#include <gtest/gtest.h>
#include "common/logger.h"
#include <fstream>
#include <string>

using namespace diff_det;

TEST(LoggerTest, Init) {
    Logger& logger = Logger::GetInstance();
    logger.Init("/tmp/test_logger.Log", "DEBUG");
    
    LOG_DEBUG("Test debug message");
    LOG_INFO("Test info message");
    LOG_WARN("Test warn message");
    LOG_ERROR("Test error message");
}

TEST(LoggerTest, LevelFilter) {
    Logger& logger = Logger::GetInstance();
    logger.Init("/tmp/test_logger_level.Log", "INFO");
    
    LOG_DEBUG("This should not appear");
    LOG_INFO("This should appear");
}

TEST(LoggerTest, SetLevel) {
    Logger& logger = Logger::GetInstance();
    logger.Init("/tmp/test_logger_setlevel.Log", "ERROR");
    
    LOG_INFO("This should not appear");
    
    logger.SetLevel("INFO");
    LOG_INFO("This should appear now");
}