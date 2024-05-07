
#include <stdlib.h>

#include <algorithm>
#include <deque>
#include <fstream>
#include <numeric>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "logger/StringRef.h"
#include "logger/logger.h"
#include "backtrace.h"


TEST(Backtrace, default) {
    EXPECT_FALSE(enable_log_backtrace("xpuMemcpy"));
    EXPECT_FALSE(enable_log_backtrace("xpuSetDevice"));
}

TEST(Backtrace, env) {
    setenv("HOOK_ENABLE_TRACE", "xpuMemcpy=1,xpuSetDevice=0", 1);
    EXPECT_TRUE(enable_log_backtrace("xpuMemcpy"));
    EXPECT_FALSE(enable_log_backtrace("xpuSetDevice"));
}
