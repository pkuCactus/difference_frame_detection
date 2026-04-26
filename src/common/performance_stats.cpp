#include "common/performance_stats.h"
#include "common/logger.h"

namespace diff_det {

PerformanceStats::PerformanceStats() {
}

void PerformanceStats::startTimer(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    startTimes_[name] = std::chrono::high_resolution_clock::now();
}

void PerformanceStats::endTimer(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = startTimes_.find(name);
    if (it == startTimes_.end()) {
        return;
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - it->second);
    double ms = duration.count();
    
    timeRecords_[name].push_back(ms);
    
    if (timeRecords_[name].size() > MAX_RECORDS) {
        timeRecords_[name].erase(timeRecords_[name].begin());
    }
    
    startTimes_.erase(it);
}

void PerformanceStats::incrementCounter(const std::string& name, int value) {
    std::lock_guard<std::mutex> lock(mutex_);
    counters_[name] += value;
}

void PerformanceStats::setCounter(const std::string& name, int value) {
    std::lock_guard<std::mutex> lock(mutex_);
    counters_[name] = value;
}

double PerformanceStats::getAverageTime(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = timeRecords_.find(name);
    if (it == timeRecords_.end() || it->second.empty()) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (double t : it->second) {
        sum += t;
    }
    
    return sum / it->second.size();
}

int PerformanceStats::getCounter(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = counters_.find(name);
    if (it == counters_.end()) {
        return 0;
    }
    
    return it->second;
}

std::string PerformanceStats::getSummary() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ostringstream oss;
    oss << "=== Performance Summary ===\n";
    
    oss << "\nTimers:\n";
    for (auto& [name, records] : timeRecords_) {
        if (records.empty()) continue;
        
        double sum = 0.0;
        double maxT = 0.0;
        double minT = std::numeric_limits<double>::max();
        
        for (double t : records) {
            sum += t;
            maxT = std::max(maxT, t);
            minT = std::min(minT, t);
        }
        
        double avg = sum / records.size();
        
        oss << "  " << name << ": avg=" << std::fixed << std::setprecision(2) << avg 
            << "ms, min=" << minT << "ms, max=" << maxT << "ms, samples=" << records.size() << "\n";
    }
    
    oss << "\nCounters:\n";
    for (auto& [name, value] : counters_) {
        oss << "  " << name << ": " << value << "\n";
    }
    
    return oss.str();
}

void PerformanceStats::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    startTimes_.clear();
    timeRecords_.clear();
    counters_.clear();
}

ScopedTimer::ScopedTimer(PerformanceStats& stats, const std::string& name)
    : stats_(stats), name_(name) {
    stats_.startTimer(name_);
}

ScopedTimer::~ScopedTimer() {
    stats_.endTimer(name_);
}

}