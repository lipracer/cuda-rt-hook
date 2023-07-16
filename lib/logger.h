#pragma once

#include <cstdlib>
#include <iostream>

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

static constexpr const char* __SimpleFileName__(const char* file) {
    const char* first = file;
    const char* second = file;
    while (*second) {
        if (*second == '/') {
            first = second;
        }
        ++second;
    }
    return first + 1;
}

struct format_string {
    template <size_t N>
    constexpr format_string(const char (&fmt_str)[N])
        : fmt_str_(fmt_str), size_(N) {
    }
    template <typename... Args>
    std::string format_to(Args&&... args) const {
        return "";
    }

    const char* fmt_str_;
    const size_t size_;
};

template <typename... Args>
inline std::string vformat(const format_string& fmt_str, Args&&... args) {
    return fmt_str.format_to(std::forward<Args>(args)...);
}

template <size_t N, typename... Args>
inline std::string format(const char(&str)[N], Args&&... args) {
    return vformat(str, std::forward<Args>(args)...);
}

}  // namespace logger

#define LOG(level)                                                            \
    !LOG_CONDITATION(level) ? void(0)                                         \
                            : logger::StreamPlaceHolder() <                   \
                                  logger::LogWrapper(level)                   \
                                      << "["                                  \
                                      << logger::__SimpleFileName__(__FILE__) \
                                      << ":" << __LINE__ << "]"

#define CHECK(p, ...)                                              \
    do {                                                           \
        if (!!!p) {                                                \
            std::cerr << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error(logger::format(__VA_ARGS__)); \
        }                                                          \
    } while (0);
