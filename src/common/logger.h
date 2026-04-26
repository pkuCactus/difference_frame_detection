#pragma once

#include <string>
#include <fstream>
#include <mutex>
#include <ctime>
#include <sstream>

namespace diff_det {

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

class Logger {
public:
    static Logger& getInstance();
    
    void init(const std::string& filePath, const std::string& level);
    void log(LogLevel level, const std::string& file, int line, const std::string& message);
    void setLevel(const std::string& level);
    
private:
    Logger() = default;
    ~Logger();
    
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    std::string levelToString(LogLevel level);
    LogLevel stringToLevel(const std::string& level);
    std::string getCurrentTime();
    
    std::ofstream fileStream_;
    std::mutex mutex_;
    LogLevel currentLevel_ = LogLevel::INFO;
    bool initialized_ = false;
};

#define LOG_DEBUG(msg) diff_det::Logger::getInstance().log(diff_det::LogLevel::DEBUG, __FILE__, __LINE__, msg)
#define LOG_INFO(msg) diff_det::Logger::getInstance().log(diff_det::LogLevel::INFO, __FILE__, __LINE__, msg)
#define LOG_WARN(msg) diff_det::Logger::getInstance().log(diff_det::LogLevel::WARNING, __FILE__, __LINE__, msg)
#define LOG_ERROR(msg) diff_det::Logger::getInstance().log(diff_det::LogLevel::ERROR, __FILE__, __LINE__, msg)

}