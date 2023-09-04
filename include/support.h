#pragma once
#include <stdlib.h>

#include <type_traits>
#include <utility>

namespace support {

#ifdef __GNUG__
#include <cxxabi.h>

#include <cstdlib>
#include <memory>
#include <string>

static std::string demangle(const char* name) {
    int status = -4;  // some arbitrary value to eliminate the compiler warning

    // enable c++11 by passing the flag -std=c++11 to g++
    std::unique_ptr<char, void (*)(void*)> res{
        abi::__cxa_demangle(name, NULL, NULL, &status), std::free};

    return (status == 0) ? res.get() : name;
}

#else

// does nothing if not g++
static std::string demangle(const char* name) { return name; }

#endif

namespace CaptureKind {
struct Value {};
struct Reference {};
struct DeepCopy {};
};  // namespace CaptureKind

class Any {
    enum CaptureKind_ {
        kValue = 0,
        kReference = 1,
        kDeepCopy = 2,
    };

   public:
    Any() {}

    template <typename T>
    Any(T&& t, CaptureKind::Value) : kind_(kValue) {
        using ValueT = typename std::remove_reference<T>::type;
        buf_ = malloc(sizeof(T));
        new (buf_) ValueT(std::forward<T>(t));
    }

    template <typename T>
    Any(T&& t, CaptureKind::Reference) : kind_(kReference) {
        buf_ = &t;
    }

    template <typename T>
    Any(T&& t, size_t size, CaptureKind::DeepCopy) : kind_(kDeepCopy) {
        buf_ = malloc(size);
        auto buf = reinterpret_cast<char*>(buf_);
        memcpy(buf, std::forward<T>(t), size);
    }

    template <typename T>
    T as() {
        using ValueT = typename std::remove_reference<T>::type;
        if (kind_ == kDeepCopy) {
            return reinterpret_cast<T>(buf_);
        }
        if (kind_ == kReference) {
            return *reinterpret_cast<ValueT*>(buf_);
        }
        return *reinterpret_cast<ValueT*>(buf_);
    }

   private:
    void* buf_{nullptr};
    CaptureKind_ kind_{kValue};
};

template <typename T>
class Functor;

template <typename R, typename... Args>
class Functor<R(Args...)> {
   public:
    using FuncT = R(Args...);
    using FuncPtrT = R (*)(Args...);

    explicit Functor(FuncPtrT ptr) : funcPtr_(ptr) {}

    template <typename T>
    void captureByValue(size_t argIndex, T&& arg) {
        args_[argIndex] = Any(std::forward<T>(arg), CaptureKind::Value());
    }

    template <typename T>
    void captureByReference(size_t argIndex, T&& arg) {
        args_[argIndex] = Any(std::forward<T>(arg), CaptureKind::Reference());
    }

    template <typename T>
    void captureByDeepCopy(size_t argIndex, T&& arg, size_t size) {
        static_assert(
            std::is_pointer<typename std::remove_reference<T>::type>::value,
            "deep copy must pointer type.");
        args_[argIndex] =
            Any(std::forward<T>(arg), size, CaptureKind::DeepCopy());
    }

    template <size_t... idx>
    R invokeArgs(std::index_sequence<idx...> = {}) {
        return funcPtr_(args_[idx].template as<Args>()...);
    }

    R operator()() {
        return invokeArgs(std::make_index_sequence<sizeof...(Args)>());
    }

   private:
    FuncPtrT funcPtr_;
    Any args_[sizeof...(Args)];
};
}  // namespace support