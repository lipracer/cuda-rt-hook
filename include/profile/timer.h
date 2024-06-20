#pragma once

#include <stddef.h>

#include <chrono>

#include "logger/logger.h"

namespace hook {
class Timer {
   public:
    Timer(const char* name = nullptr,
          std::chrono::time_point<std::chrono::steady_clock> tp =
              std::chrono::steady_clock::now())
        : name_(name), _start_time(tp) {}
    ~Timer() {
        MLOG(PROFILE, INFO)
            << (name_ ? name_ : "") << " cost:" << Passed() * 1000 << " ms";
    }

    template <typename T>
    bool Timeout(double count) const {
        return Passed<T>() >= count;
    }

    double Passed() const { return Passed<std::chrono::duration<double>>(); }

    double PassedSec() const { return Passed<std::chrono::seconds>(); }

    double PassedMicro() const { return Passed<std::chrono::microseconds>(); }

    double PassedNano() const { return Passed<std::chrono::nanoseconds>(); }

    template <typename T>
    double Passed() const {
        return Passed<T>(std::chrono::steady_clock::now());
    }

    template <typename T>
    double Passed(std::chrono::time_point<std::chrono::steady_clock> tp) const {
        const auto elapsed = std::chrono::duration_cast<T>(tp - _start_time);
        return elapsed.count();
    }

    uint64_t TimePointMicro() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   _start_time.time_since_epoch())
            .count();
    }

    void Reset() { _start_time = std::chrono::steady_clock::now(); }

   private:
    std::chrono::time_point<std::chrono::steady_clock> _start_time;
    const char* name_{nullptr};
};

}  // namespace hook