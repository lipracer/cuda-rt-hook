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

    __Any(__Any&& other) { *this = std::move(other); }

    __Any& operator=(__Any&& other) {
        if (this == &other) {
            return *this;
        }
        buf_ = other.buf_;
        releaseBuf_ = other.releaseBuf_;
        castImpl_ = other.castImpl_;
        return *this;
    }

    template <typename T>
    __Any(T&& t, CaptureKind::Value) : releaseBuf_(true) {
        using ValueT = typename std::remove_reference<T>::type;
        buf_ = AllocT().alloc(sizeof(T));
        new (buf_) ValueT(std::forward<T>(t));
        castImpl_ = reinterpret_cast<void*>(
            &__Any<AllocT>::as_bv<std::remove_reference_t<T>>);
    }

    template <typename T>
    __Any(T&& t, CaptureKind::Reference) {
        buf_ = &t;
        castImpl_ = reinterpret_cast<void*>(
            &__Any<AllocT>::as_br<std::remove_reference_t<T>>);
    }

    template <typename T>
    __Any(T&& t, size_t size, CaptureKind::DeepCopy) : releaseBuf_(true) {
        buf_ = AllocT().alloc(size);
        auto buf = reinterpret_cast<char*>(buf_);
        memcpy(buf, std::forward<T>(t), size);
        castImpl_ = reinterpret_cast<void*>(
            &__Any<AllocT>::as_dc<std::remove_reference_t<T>>);
    }

    ~__Any() {
        if (releaseBuf_) {
            AllocT().dealloc(buf_);
        }
    }

    template <typename T>
    T as() {
        return reinterpret_cast<T (*)(void*)>(castImpl_)(buf_);
    }

    template <typename T>
    static T as_dc(void* buf) {
        return reinterpret_cast<T>(buf);
    }

    template <typename T>
    static T as_bv(void* buf) {
        using ValueT = typename std::remove_reference<T>::type;
        return *reinterpret_cast<ValueT*>(buf);
    }

    template <typename T>
    static T as_br(void* buf) {
        using ValueT = typename std::remove_reference<T>::type;
        return *reinterpret_cast<ValueT*>(buf);
    }

   private:
    void* buf_{nullptr};
    void* castImpl_{nullptr};
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

    Functor(Functor&& other) {
        *this = std::move(other);
    }
    Functor& operator=(Functor&& other) {
        if (this == &other) {
            return *this;
        }
        std::swap(funcPtr_, other.funcPtr_);
        std::swap(args_, other.args_);
        std::swap(invoker_, other.invoker_);
        return *this;
    }

    // TODO support
    Functor(const Functor& other) = delete;
    Functor& operator=(const Functor& other) = delete;

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
   using InvokePtrT = U (*)(void*, Any*);
    void* funcPtr_{nullptr};
    Any* args_{nullptr};
    InvokePtrT invoker_{nullptr};
};

}  // namespace support