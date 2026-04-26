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
    
    void StartTimer(const std::string& name);
    void EndTimer(const std::string& name);
    
    void IncrementCounter(const std::string& name, int32_t value = 1);
    void SetCounter(const std::string& name, int32_t value);
    
    double GetAverageTime(const std::string& name);
    int32_t GetCounter(const std::string& name);
    
    std::string GetSummary();
    void Reset();
    
private:
    std::mutex mutex_;
    
    std::map<std::string, std::chrono::high_resolution_clock::time_point> startTimes_;
    std::map<std::string, std::vector<double>> timeRecords_;
    std::map<std::string, int32_t> counters_;
    
    static constexpr int32_t MAX_RECORDS = 1000;
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