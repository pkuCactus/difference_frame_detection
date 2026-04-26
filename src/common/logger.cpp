#include "common/logger.h"
#include <iostream>
#include <iomanip>

namespace diff_det {

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

Logger::~Logger() {
    if (fileStream_.is_open()) {
        fileStream_.close();
    }
}

void Logger::init(const std::string& filePath, const std::string& level) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (fileStream_.is_open()) {
        fileStream_.close();
    }
    
    fileStream_.open(filePath, std::ios::out | std::ios::app);
    if (!fileStream_.is_open()) {
        std::cerr << "Failed to open log file: " << filePath << std::endl;
        return;
    }
    
    currentLevel_ = stringToLevel(level);
    initialized_ = true;
}

void Logger::log(LogLevel level, const std::string& file, int line, const std::string& message) {
    if (!initialized_) {
        return;
    }
    
    if (level < currentLevel_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::string fileName = file;
    size_t pos = fileName.find_last_of("/\\");
    if (pos != std::string::npos) {
        fileName = fileName.substr(pos + 1);
    }
    
    fileStream_ << "[" << getCurrentTime() << "] "
                << "[" << levelToString(level) << "] "
                << "[" << fileName << ":" << line << "] "
                << message << std::endl;
    
    fileStream_.flush();
}

void Logger::setLevel(const std::string& level) {
    std::lock_guard<std::mutex> lock(mutex_);
    currentLevel_ = stringToLevel(level);
}

std::string Logger::levelToString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

LogLevel Logger::stringToLevel(const std::string& level) {
    if (level == "DEBUG") return LogLevel::DEBUG;
    if (level == "INFO") return LogLevel::INFO;
    if (level == "WARNING") return LogLevel::WARNING;
    if (level == "ERROR") return LogLevel::ERROR;
    return LogLevel::INFO;
}

std::string Logger::getCurrentTime() {
    std::time_t now = std::time(nullptr);
    std::tm* tm = std::localtime(&now);
    
    std::ostringstream oss;
    oss << std::put_time(tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

}