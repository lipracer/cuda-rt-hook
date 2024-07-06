
#include <algorithm>
#include <future>
#include <numeric>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "logger/logger.h"

using namespace logger;

#define LOGGER_CHECK_TEST(f, ...)      \
    void Test##f() { f(__VA_ARGS__); } \
    TEST(DeathTest, logger_##f) { EXPECT_DEATH(Test##f(), ".*"); }

LOGGER_CHECK_TEST(CHECK, false, "unexpected!")

LOGGER_CHECK_TEST(CHECK_LT, 2, 1)
LOGGER_CHECK_TEST(CHECK_LE, 3, 2)
LOGGER_CHECK_TEST(CHECK_GT, 2, 3)
LOGGER_CHECK_TEST(CHECK_GE, 4, 7)

template <char... strs>
struct CombinString {};

// template <size_t N, size_t... idx>
// auto makeCombinString(std::index_sequence<idx...>) {
//     return CombinString<h0.str_[idx]...>();
// }

TEST(logger, string_literal) {
    std::cout << strlen("123") << std::endl;
    auto string_literal = makeStringLiteral("123");
    std::cout << string_literal.size() << std::endl;

    auto rhs_string_literal = makeStringLiteral("4567");
    auto sum_string = string_literal + rhs_string_literal;
    std::cout << sum_string.size() << std::endl;
    std::cout << sum_string.str_ << std::endl;
    constexpr size_t size = sum_string.size();
    std::make_index_sequence<size>();

    auto n_str_literatl = makeStringLiteral<15>();
    std::cout << n_str_literatl.size() << std::endl;
    std::cout << n_str_literatl.str_ << std::endl;

    constexpr auto h0 = GetFixedLogerHeader(__FILE__, __LINE__);
    constexpr auto h1 = MGetFixedLogerHeader("PYTHON", __FILE__, __LINE__);

    std::cout << h0 << std::endl;
    std::cout << h1 << std::endl;

    // auto cStr = makeCombinString(std::make_index_sequence<h0.size()>(), h0);

    CombinString<h0.str_[0], h0.str_[1]> cStr;
    std::cout << typeid(cStr).name() << std::endl;
    auto lhs_str_ty = STR_TO_TYPE("123456");
    auto rhs_str_ty = STR_TO_TYPE("123456");
    static_assert(
        std::is_same<decltype(lhs_str_ty), decltype(rhs_str_ty)>::value, "");
    // std::cout << std::string(STR_TO_TYPE("123456")) << std::endl;
}

TEST(logger, sync_logger) {
    std::future<int> futureResult = std::async([]() {
        for (size_t i = 0; i < 1000; ++i) {
            LOG(WARN) << i;
        }
        return 0;
    });
}

TEST(logger, overflow) {
    std::string line(128, 'A');
    std::string large_msg;
    for (size_t i = 0; i < 30; ++i) {
        large_msg = large_msg + line + '\n';
    }
    LOG(WARN) << large_msg;
}

TEST(LoggerBase, loglevel) {
    LogStream::instance().setLevel(LogLevel::warning);
    LOG(INFO) << "123";
    LOG(WARN) << "456";
    LOG(ERROR) << "789";

    LogStream::instance().setLevel(LogLevel::info);
    LOG(INFO) << "123";
    LOG(WARN) << "456";
    LOG(ERROR) << "789";
}

TEST(LoggerBase, Mloglevel) {
    LogStream::instance().setModuleLevel(LogModule::profile, LogLevel::info);
    MLOG(PROFILE, INFO) << "123";
    MLOG(PROFILE, WARN) << "123";
    MLOG(PYTHON, INFO) << "123";
    MLOG(PYTHON, WARN) << "123";

    LogStream::instance().setModuleLevel(LogModule::profile, LogLevel::warning);

    LogStream::instance().setModuleLevel(LogModule::python, LogLevel::info);
    MLOG(PROFILE, INFO) << "123";
    MLOG(PROFILE, WARN) << "123";
    MLOG(PYTHON, INFO) << "123";
    MLOG(PYTHON, WARN) << "123";
}

TEST(LoggerBase, env_loglevel) {
    LogLevel level = LogLevel::info;
    std::array<LogLevel, static_cast<size_t>(LogModule::last) + 1> module_set =
        {LogLevel::info};
    EXPECT_EQ(level, LogLevel::info);
    EXPECT_EQ(module_set[0], LogLevel::info);
    EXPECT_EQ(module_set[1], LogLevel::info);
    EXPECT_EQ(module_set[2], LogLevel::info);
    EXPECT_EQ(module_set[3], LogLevel::info);
    EXPECT_EQ(module_set[4], LogLevel::info);
    setenv("LOG_LEVEL", "WARN,PROFILE=INFO,TRACE=WARN", 1);
    setLoggerLevel(module_set, level);
    EXPECT_EQ(level, LogLevel::warning);
    EXPECT_EQ(module_set[static_cast<size_t>(PROFILE)], LogLevel::info);
    EXPECT_EQ(module_set[static_cast<size_t>(TRACE)], LogLevel::warning);
}

void goDie() { LOG(FATAL) << "go die"; }

TEST(LoggerBase, fatal) {
    // write to buffer
    for (size_t i = 0; i < 100; ++i) {
        LOG(WARN) << "123";
    }
    // test wait flush buffer
    ASSERT_DEATH(goDie(), "");
}
