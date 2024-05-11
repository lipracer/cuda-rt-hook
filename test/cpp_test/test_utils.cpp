
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

TEST(EnvUtil, env_util) {
    setenv("xpuWait", "0", 1);
    int intv = hook::get_env_value<int>("xpuWait");
    EXPECT_EQ(intv, 0);

    setenv("hook_ctr", "xpuWait=0", 1);
    std::pair<std::string, int> pairv =
        hook::get_env_value<std::pair<std::string, int>>("hook_ctr");
    EXPECT_EQ(pairv.first, "xpuWait");
    EXPECT_EQ(pairv.second, 0);

    setenv("hook_ctr", "xpuWait=0,xpuSetDiviceId=1", 1);
    std::vector<std::pair<std::string, int>> vec =
        hook::get_env_value<std::vector<std::pair<std::string, int>>>(
            "hook_ctr");

    EXPECT_EQ(vec[0].first, "xpuWait");
    EXPECT_EQ(vec[0].second, 0);

    EXPECT_EQ(vec[1].first, "xpuSetDiviceId");
    EXPECT_EQ(vec[1].second, 1);
}

TEST(Backtrace, default) {
    EXPECT_FALSE(enable_log_backtrace("xpuMemcpy"));
    EXPECT_FALSE(enable_log_backtrace("xpuSetDevice"));
}

TEST(Backtrace, env) {
    setenv("HOOK_ENABLE_TRACE", "xpuMemcpy=1,xpuSetDevice=0", 1);
    EXPECT_TRUE(enable_log_backtrace("xpuMemcpy"));
    EXPECT_FALSE(enable_log_backtrace("xpuSetDevice"));
}

TEST(StringRef, member) {
    EXPECT_EQ("123", adt::StringRef("123"));
    EXPECT_EQ(adt::StringRef("123"), "123");

    EXPECT_EQ(adt::StringRef("abcdefg").contain("ab"), true);
    EXPECT_EQ(adt::StringRef("abcdefg").find("hj"), std::string::npos);

    EXPECT_EQ(adt::StringRef("").contain("ab"), false);
    EXPECT_EQ(adt::StringRef("abcdnnn").contain("nnnn"), false);
    EXPECT_EQ(adt::StringRef("abcdnnn").contain("acd"), false);

    EXPECT_EQ(adt::StringRef("abcdnnn"), "abcdnnn");
    EXPECT_NE(adt::StringRef("abcdnnn"), "sabcdnnn");
    EXPECT_TRUE(adt::StringRef("abcd").contain("bc"));
}


