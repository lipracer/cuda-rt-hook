#pragma once

#include <cstdlib>
#include <iostream>

#define CHECK(p, msg)                            \
    do {                                         \
        if (!!!p) throw std::runtime_error(msg); \
    } while (0);

namespace logger {

enum class LogLevel { info = 0, debug, warning, error };

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
    explicit LogWrapper(LogLevel level) : level_(level) {}
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

#define LOG(level)                                                    \
    !LOG_CONDITATION(level)                                           \
        ? void(0)                                                     \
        : logger::StreamPlaceHolder() < logger::LogWrapper(level)     \
                                            << "[" << __FILE__ << ":" \
                                            << __LINE__ << "]"