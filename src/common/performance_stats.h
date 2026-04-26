#pragma once

#include <string>
#include <map>
#include <vector>
#include <chrono>
#include <mutex>
#include <sstream>
#include <iomanip>
#include <limits>

namespace diff_det {

class PerformanceStats {
public:
    PerformanceStats();
    
    void startTimer(const std::string& name);
    void endTimer(const std::string& name);
    
    void incrementCounter(const std::string& name, int value = 1);
    void setCounter(const std::string& name, int value);
    
    double getAverageTime(const std::string& name);
    int getCounter(const std::string& name);
    
    std::string getSummary();
    void reset();
    
private:
    std::mutex mutex_;
    
    std::map<std::string, std::chrono::high_resolution_clock::time_point> startTimes_;
    std::map<std::string, std::vector<double>> timeRecords_;
    std::map<std::string, int> counters_;
    
    static constexpr int MAX_RECORDS = 1000;
};

class ScopedTimer {
public:
    ScopedTimer(PerformanceStats& stats, const std::string& name);
    ~ScopedTimer();
    
private:
    PerformanceStats& stats_;
    std::string name_;
};

}