#include <gtest/gtest.h>
#include "common/performance_stats.h"
#include <thread>

using namespace diff_det;

TEST(PerformanceStatsTest, Init) {
    PerformanceStats stats;
    
    EXPECT_EQ(stats.GetCounter("test"), 0);
    EXPECT_EQ(stats.GetAverageTime("test"), 0.0);
}

TEST(PerformanceStatsTest, Counter) {
    PerformanceStats stats;
    
    stats.IncrementCounter("test_counter");
    EXPECT_EQ(stats.GetCounter("test_counter"), 1);
    
    stats.IncrementCounter("test_counter", 5);
    EXPECT_EQ(stats.GetCounter("test_counter"), 6);
    
    stats.SetCounter("test_counter", 100);
    EXPECT_EQ(stats.GetCounter("test_counter"), 100);
}

TEST(PerformanceStatsTest, Timer) {
    PerformanceStats stats;
    
    stats.StartTimer("test_timer");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    stats.EndTimer("test_timer");
    
    double avgTime = stats.GetAverageTime("test_timer");
    EXPECT_GE(avgTime, 40.0);
    EXPECT_LT(avgTime, 100.0);
}

TEST(PerformanceStatsTest, ScopedTimer) {
    PerformanceStats stats;
    
    {
        ScopedTimer timer(stats, "scoped_timer");
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    
    double avgTime = stats.GetAverageTime("scoped_timer");
    EXPECT_GE(avgTime, 20.0);
}

TEST(PerformanceStatsTest, Summary) {
    PerformanceStats stats;
    
    stats.IncrementCounter("counter1", 10);
    stats.IncrementCounter("counter2", 20);
    
    stats.StartTimer("timer1");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    stats.EndTimer("timer1");
    
    std::string summary = stats.GetSummary();
    EXPECT_TRUE(summary.find("counter1") != std::string::npos);
    EXPECT_TRUE(summary.find("timer1") != std::string::npos);
}

TEST(PerformanceStatsTest, Reset) {
    PerformanceStats stats;
    
    stats.IncrementCounter("test", 100);
    stats.StartTimer("test_timer");
    stats.EndTimer("test_timer");
    
    stats.Reset();
    
    EXPECT_EQ(stats.GetCounter("test"), 0);
    EXPECT_EQ(stats.GetAverageTime("test_timer"), 0.0);
}