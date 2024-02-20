
#include <algorithm>
#include <future>
#include <numeric>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "logger/logger.h"

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
    auto string_literal = logger::makeStringLiteral("123");
    std::cout << string_literal.size() << std::endl;

    auto rhs_string_literal = logger::makeStringLiteral("4567");
    auto sum_string = string_literal + rhs_string_literal;
    std::cout << sum_string.size() << std::endl;
    std::cout << sum_string.str_ << std::endl;
    constexpr size_t size = sum_string.size();
    std::make_index_sequence<size>();

    auto n_str_literatl = logger::makeStringLiteral<15>();
    std::cout << n_str_literatl.size() << std::endl;
    std::cout << n_str_literatl.str_ << std::endl;

    constexpr auto h0 = GetFixedLogerHeader(__FILE__, __LINE__);
    constexpr auto h1 = MGetFixedLogerHeader("PYTHON", __FILE__, __LINE__);

    std::cout << h0 << std::endl;
    std::cout << h1 << std::endl;

    // auto cStr = makeCombinString(std::make_index_sequence<h0.size()>(), h0);

    CombinString<h0.str_[0], h0.str_[1]> cStr;
    std::cout << typeid(cStr).name() << std::endl;
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
    for(size_t i = 0; i < 30; ++i) {
    large_msg = large_msg + line + '\n';
    }
    LOG(WARN) << large_msg;
}
