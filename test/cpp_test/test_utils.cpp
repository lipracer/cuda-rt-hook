
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
    setenv("xpu_wait", "0", 1);
    int intv = hook::get_env_value<int>("xpu_wait");
    EXPECT_EQ(intv, 0);

    setenv("hook_ctr", "xpu_wait=0", 1);
    std::pair<std::string, int> pairv =
        hook::get_env_value<std::pair<std::string, int>>("hook_ctr");
    EXPECT_EQ(pairv.first, "xpu_wait");
    EXPECT_EQ(pairv.second, 0);

    setenv("hook_ctr", "xpu_wait=0,xpu_set_device=1", 1);
    std::vector<std::pair<std::string, int>> vec =
        hook::get_env_value<std::vector<std::pair<std::string, int>>>(
            "hook_ctr");

    EXPECT_EQ(vec[0].first, "xpu_wait");
    EXPECT_EQ(vec[0].second, 0);

    EXPECT_EQ(vec[1].first, "xpu_set_device");
    EXPECT_EQ(vec[1].second, 1);

    setenv("hook_ctr", "xpu_wait=0x3,xpu_set_device=0b101", 1);
    vec = hook::get_env_value<std::vector<std::pair<std::string, int>>>(
        "hook_ctr");
    EXPECT_EQ(vec[0].first, "xpu_wait");
    EXPECT_EQ(vec[0].second, 0x3);

    EXPECT_EQ(vec[1].first, "xpu_set_device");
    EXPECT_EQ(vec[1].second, 0b101);

    // error case
    setenv("hook_ctr", "xpu_wait=NOTANUBER", 1);
    pairv = hook::get_env_value<std::pair<std::string, int>>("hook_ctr");
    EXPECT_EQ(pairv.first, "xpu_wait");
    EXPECT_EQ(pairv.second, 0);
}

TEST(Backtrace, env) {
    setenv("HOOK_ENABLE_TRACE", "xpu_memcpy=1,xpu_set_device=0", 1);
    EXPECT_TRUE(enable_log_backtrace("xpu_memcpy"));
    EXPECT_FALSE(enable_log_backtrace("xpu_set_device"));
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

    EXPECT_TRUE(adt::StringRef("abcd").startsWith("ab"));
    EXPECT_TRUE(adt::StringRef("abcd").startsWith("abc"));
    EXPECT_TRUE(adt::StringRef("abcd").startsWith("abcd"));

    EXPECT_TRUE(adt::StringRef("abcd").endsWith("d"));
    EXPECT_TRUE(adt::StringRef("abcd").endsWith("cd"));
    EXPECT_TRUE(adt::StringRef("abcd").endsWith("abcd"));

    EXPECT_EQ(adt::StringRef("abcd").drop_front(2), adt::StringRef("cd"));
    EXPECT_EQ(adt::StringRef("abcd").drop_back(2), adt::StringRef("ab"));
}


