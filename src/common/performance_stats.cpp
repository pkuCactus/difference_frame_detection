#include "common/performance_stats.h"
#include "common/logger.h"

namespace diff_det {

PerformanceStats::PerformanceStats() {
}

void PerformanceStats::StartTimer(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    startTimes_[name] = std::chrono::high_resolution_clock::now();
}

void PerformanceStats::EndTimer(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = startTimes_.find(name);
    if (it == startTimes_.end()) {
        return;
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - it->second);
    double ms = static_cast<double>(duration.count());
    
    timeRecords_[name].push_back(ms);
    
    if (timeRecords_[name].size() > static_cast<size_t>(MAX_RECORDS)) {
        timeRecords_[name].erase(timeRecords_[name].begin());
    }
    
    startTimes_.erase(it);
}

void PerformanceStats::IncrementCounter(const std::string& name, int32_t value) {
    std::lock_guard<std::mutex> lock(mutex_);
    counters_[name] += value;
}

void PerformanceStats::SetCounter(const std::string& name, int32_t value) {
    std::lock_guard<std::mutex> lock(mutex_);
    counters_[name] = value;
}

double PerformanceStats::GetAverageTime(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = timeRecords_.find(name);
    if (it == timeRecords_.end() || it->second.empty()) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (double t : it->second) {
        sum += t;
    }
    
    return sum / static_cast<double>(it->second.size());
}

int32_t PerformanceStats::GetCounter(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = counters_.find(name);
    if (it == counters_.end()) {
        return 0;
    }
    
    return it->second;
}

std::string PerformanceStats::GetSummary() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ostringstream oss;
    oss << "=== Performance Summary ===\n";
    
    oss << "\nTimers:\n";
    for (auto& [name, records] : timeRecords_) {
        if (records.empty()) continue;
        
        double sum = 0.0;
        double maxTime = 0.0;
        double minTime = std::numeric_limits<double>::max();
        
        for (double t : records) {
            sum += t;
            maxTime = std::max(maxTime, t);
            minTime = std::min(minTime, t);
        }
        
        double avg = sum / static_cast<double>(records.size());
        
        oss << "  " << name << ": avg=" << std::fixed << std::setprecision(2) << avg 
            << "ms, min=" << minTime << "ms, max=" << maxTime << "ms, samples=" << records.size() << "\n";
    }
    
    oss << "\nCounters:\n";
    for (auto& [name, value] : counters_) {
        oss << "  " << name << ": " << value << "\n";
    }
    
    return oss.str();
}

void PerformanceStats::Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    startTimes_.clear();
    timeRecords_.clear();
    counters_.clear();
}

ScopedTimer::ScopedTimer(PerformanceStats& stats, const std::string& name)
    : stats_(stats), name_(name) {
    stats_.StartTimer(name_);
}

ScopedTimer::~ScopedTimer() {
    stats_.EndTimer(name_);
}

}