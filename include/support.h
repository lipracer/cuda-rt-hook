#pragma once
#include <stdlib.h>
#include <type_traits>
#include <utility>

#ifdef __GNUG__
#include <cxxabi.h>

#include <cstdlib>
#include <memory>
#include <string>
static std::string __support__demangle(const char* name) {
    int status = -4;  // some arbitrary value to eliminate the compiler warning
    // enable c++11 by passing the flag -std=c++11 to g++
    std::unique_ptr<char, void (*)(void*)> res{
        abi::__cxa_demangle(name, NULL, NULL, &status), std::free};

    return (status == 0) ? res.get() : name;
}
#else
// does nothing if not g++
static std::string __support__demangle(const char* name) { return name; }

#endif

namespace support {

namespace CaptureKind {
struct Value {};
struct Reference {};
struct DeepCopy {};
};  // namespace CaptureKind

class SimpleAllocater {
   public:
    void* alloc(size_t size) {
        auto ptr = malloc(size);
        return ptr;
    }

    void dealloc(void* ptr) { free(ptr); }
};

template <typename AllocT = SimpleAllocater>
class __Any {
    enum CaptureKind_ {
        kValue = 0,
        kReference = 1,
        kDeepCopy = 2,
    };

   public:
    __Any() {}
    __Any(const __Any&) = delete;
    __Any& operator=(const __Any& other) = delete;

    __Any(__Any&& other) {
        std::swap(buf_, other.buf_);
        std::swap(kind_, other.kind_);
        releaseBuf_ = other.releaseBuf_;
    }

    __Any& operator=(__Any&& other) {
        if (this == &other) {
            return *this;
        }
        std::swap(buf_, other.buf_);
        std::swap(kind_, other.kind_);
        releaseBuf_ = other.releaseBuf_;
        return *this;
    }

    template <typename T>
    __Any(T&& t, CaptureKind::Value) : kind_(kValue), releaseBuf_(true) {
        using ValueT = typename std::remove_reference<T>::type;
        buf_ = AllocT().alloc(sizeof(T));
        new (buf_) ValueT(std::forward<T>(t));
    }

    template <typename T>
    __Any(T&& t, CaptureKind::Reference) : kind_(kReference) {
        buf_ = &t;
    }

    template <typename T>
    __Any(T&& t, size_t size, CaptureKind::DeepCopy)
        : kind_(kDeepCopy), releaseBuf_(true) {
        buf_ = AllocT().alloc(size);
        auto buf = reinterpret_cast<char*>(buf_);
        memcpy(buf, std::forward<T>(t), size);
    }

    ~__Any() {
        if (releaseBuf_) {
            AllocT().dealloc(buf_);
        }
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
    bool releaseBuf_{false};
};

using Any = __Any<>;

template <typename T>
struct Invoker;

template <typename R, typename... Args>
struct Invoker<R(Args...)> {
    template <size_t... idx>
    static R __InvokeImplement(void* funcPtr, Any* args,
                               std::index_sequence<idx...> = {}) {
        return reinterpret_cast<R (*)(Args...)>(funcPtr)(
            args[idx].template as<Args>()...);
    }

    static R InvokeImplement(void* funcPtr, Any* args) {
        return __InvokeImplement(funcPtr, args,
                                 std::make_index_sequence<sizeof...(Args)>());
    }
};

template <typename... Args>
struct Invoker<void(Args...)> {
    template <size_t... idx>
    static void __InvokeImplement(void* funcPtr, Any* args,
                                  std::index_sequence<idx...> = {}) {
        reinterpret_cast<void (*)(Args...)>(funcPtr)(
            args[idx].template as<Args>()...);
    }

    static void InvokeImplement(void* funcPtr, Any* args) {
        return __InvokeImplement(funcPtr, args,
                                 std::make_index_sequence<sizeof...(Args)>());
    }
};

template <typename U, typename AllocT = SimpleAllocater>
class Functor {
   public:
    template <typename R, typename... Args>
    explicit Functor(R (*ptr)(Args...))
        : funcPtr_(reinterpret_cast<void*>(ptr)),
          args_(reinterpret_cast<Any*>(
              AllocT().alloc(sizeof(Any) * sizeof...(Args)))),
          invoker_(&Invoker<R(Args...)>::InvokeImplement) {
        for (size_t i = 0; i < sizeof...(Args); ++i) {
            (void)new (reinterpret_cast<void*>(args_ + i)) Any();
        }
    }

    ~Functor() { AllocT().dealloc(args_); }

    template <typename T>
    typename std::enable_if_t<
        !std::is_reference<std::remove_reference_t<T>>::value>
    capture(size_t argIndex, T&& arg) {
        args_[argIndex] = Any(std::forward<T>(arg), CaptureKind::Value());
    }

    template <typename T>
    typename std::enable_if_t<
        std::is_reference<std::remove_reference_t<T>>::value>
    capture(size_t argIndex, T&& arg) {
        args_[argIndex] = Any(std::forward<T>(arg), CaptureKind::Reference());
    }

    template <typename T>
    typename std::enable_if_t<
        std::is_rvalue_reference<std::remove_reference_t<T>>::value>
    capture(size_t argIndex, T&& arg) {
        args_[argIndex] = Any(std::forward<T>(arg), CaptureKind::Value());
    }

    template <typename T>
    void captureByDeepCopy(size_t argIndex, T&& arg, size_t size) {
        static_assert(
            std::is_pointer<typename std::remove_reference<T>::type>::value,
            "deep copy must pointer type.");
        args_[argIndex] =
            Any(std::forward<T>(arg), size, CaptureKind::DeepCopy());
    }

    U operator()(typename std::enable_if<!std::is_same<U, void>::value,
                                         void>::type* = nullptr) {
        return invoker_(funcPtr_, args_);
    }

   private:
    void* funcPtr_;
    Any* args_;
    U (*invoker_)(void*, Any*);
};

}  // namespace support