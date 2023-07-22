#pragma once

#include <cstdlib>
#include <iostream>
#include "fmt/core.h"

namespace logger {

enum class LogLevel { info = 0, debug, warning, error };

struct StringLiteralBase {
    constexpr StringLiteralBase(size_t N) { MN = MN > N ? MN : N; }
    static size_t MN;
};

template <size_t N>
struct StringLiteral : public StringLiteralBase {
    constexpr StringLiteral(const char (&str)[N]) : StringLiteralBase(N) {
        for (size_t i = 0; i < N; ++i) {
            str_[i] = str[i];
        }
    }

    char str_[N];

    constexpr auto simpleFile() const {
        for (int i = N - 1; i > 0; --i) {
            if (str_[i] == '/') {
                return str_ + i + 1;
            }
        }
        return str_;
    }
};

template <size_t N>
constexpr std::ostream& operator<<(std::ostream& os,
                                   const StringLiteral<N>& str) {
    os << str.str_;
    return os;
}

class LogStream {
   public:
    static LogStream& instance();

    LogStream() {
        auto strLevel = std::getenv("LOG_LEVEL");
        if (strLevel) {
            level_ = static_cast<LogLevel>(atoi(strLevel));
        }
    }

    void flush() { std::cout << std::endl; }

    LogLevel getLevel() const { return level_; }

    const char* getStrLevel(LogLevel level) {
        constexpr const char* str[] = {"INFO", "DEBUG", "WARN", "ERROR"};
        return str[static_cast<int>(level)];
    }

   private:
    LogLevel level_ = LogLevel::warning;
};

#define LOG_CONDITATION(level)    \
    (static_cast<int>((level)) >= \
     static_cast<int>(logger::LogStream::instance().getLevel()))

template <typename T>
static LogStream& operator<<(LogStream& s, const T& t) {
    std::cout << t;
    return s;
}

struct LogWrapper {
    explicit LogWrapper(LogLevel level) : level_(level) {
        LogStream::instance()
            << "[" << LogStream::instance().getStrLevel(level) << "]";
    }
    explicit LogWrapper(int level) : level_(static_cast<LogLevel>(level)) {}
    ~LogWrapper() {
        if (LOG_CONDITATION(level_)) {
            LogStream::instance().flush();
        }
    }
    LogLevel level_;
};

template <typename T>
static const LogWrapper& operator<<(const LogWrapper& s, const T& t) {
    if (LOG_CONDITATION(s.level_)) {
        LogStream::instance() << t;
    }
    return s;
}

class StreamPlaceHolder {};

static void operator<(const StreamPlaceHolder&, const LogWrapper&) {}


}  // namespace logger

#define INFO logger::LogLevel::info
#define DEBUG logger::LogLevel::debug
#define WARN logger::LogLevel::warning
#define ERROR logger::LogLevel::error

#define LOG(level)                                                       \
    !LOG_CONDITATION(level)                                              \
        ? void(0)                                                        \
        : logger::StreamPlaceHolder() <                                  \
              logger::LogWrapper(level)                                  \
                  << "[" << logger::StringLiteral(__FILE__).simpleFile() \
                  << ":" << __LINE__ << "]"

#define CHECK(p, ...)                                              \
    do {                                                           \
        if (!(p)) {                                                \
            std::cerr << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error(fmt::format(__VA_ARGS__));    \
        }                                                          \
    } while (0)

#define CHECK_LT(l, r, ...) CHECK(((l) < (r)), __VA_ARGS__)
#define CHECK_LE(l, r, ...) CHECK(((l) <= (r)), __VA_ARGS__)
#define CHECK_GT(l, r, ...) CHECK(((l) > (r)), __VA_ARGS__)
#define CHECK_GE(l, r, ...) CHECK(((l) >= (r)), __VA_ARGS__)
#define CHECK_EQ(l, r, ...) CHECK(((l) == (r)), __VA_ARGS__)