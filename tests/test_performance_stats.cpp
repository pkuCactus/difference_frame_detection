#include <gtest/gtest.h>
#include "common/performance_stats.h"
#include <thread>

using namespace diff_det;

TEST(PerformanceStatsTest, Init) {
    PerformanceStats stats;
    
    EXPECT_EQ(stats.getCounter("test"), 0);
    EXPECT_EQ(stats.getAverageTime("test"), 0.0);
}

TEST(PerformanceStatsTest, Counter) {
    PerformanceStats stats;
    
    stats.incrementCounter("test_counter");
    EXPECT_EQ(stats.getCounter("test_counter"), 1);
    
    stats.incrementCounter("test_counter", 5);
    EXPECT_EQ(stats.getCounter("test_counter"), 6);
    
    stats.setCounter("test_counter", 100);
    EXPECT_EQ(stats.getCounter("test_counter"), 100);
}

TEST(PerformanceStatsTest, Timer) {
    PerformanceStats stats;
    
    stats.startTimer("test_timer");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    stats.endTimer("test_timer");
    
    double avgTime = stats.getAverageTime("test_timer");
    EXPECT_GE(avgTime, 40.0);
    EXPECT_LT(avgTime, 100.0);
}

TEST(PerformanceStatsTest, ScopedTimer) {
    PerformanceStats stats;
    
    {
        ScopedTimer timer(stats, "scoped_timer");
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    
    double avgTime = stats.getAverageTime("scoped_timer");
    EXPECT_GE(avgTime, 20.0);
}

TEST(PerformanceStatsTest, Summary) {
    PerformanceStats stats;
    
    stats.incrementCounter("counter1", 10);
    stats.incrementCounter("counter2", 20);
    
    stats.startTimer("timer1");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    stats.endTimer("timer1");
    
    std::string summary = stats.getSummary();
    EXPECT_TRUE(summary.find("counter1") != std::string::npos);
    EXPECT_TRUE(summary.find("timer1") != std::string::npos);
}

TEST(PerformanceStatsTest, Reset) {
    PerformanceStats stats;
    
    stats.incrementCounter("test", 100);
    stats.startTimer("test_timer");
    stats.endTimer("test_timer");
    
    stats.reset();
    
    EXPECT_EQ(stats.getCounter("test"), 0);
    EXPECT_EQ(stats.getAverageTime("test_timer"), 0.0);
}