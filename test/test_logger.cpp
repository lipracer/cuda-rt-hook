
#include <algorithm>
#include <numeric>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "logger.h"

static void simpleTask() {
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 10000; ++i) {
        double s = 0.0;
        LOG(WARN) << "task loop index:" << i << "---------"
                  << "#########" << (s + i);
        std::vector<size_t> nums(10000);
        std::iota(nums.begin(), nums.end(), 0);
        std::reverse(nums.begin(), nums.end());
        std::sort(nums.begin(), nums.end());
    }
    auto end = std::chrono::high_resolution_clock::now();
    LOG(ERROR) << "task run:"
               << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                        start)
                      .count()
               << "ms";
    LOG(ERROR) << "logger run:"
               << std::chrono::duration_cast<std::chrono::milliseconds>(
                      logger::LogWrapper::totalDur)
                      .count()
               << "ms";
}

TEST(LoggerTest, mt) {
    // LOG(WARN) << "starting";
    std::vector<std::thread> ths;
    for (size_t i = 0; i < 10; ++i) {
        ths.emplace_back(simpleTask);
    }
    for (auto& th : ths) {
        th.join();
    }
}

TEST(LoggerTest, performance) {
    for (size_t i = 0; i < 10000; ++i) {
        simpleTask();
    }
}
