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

class SimpleAllocater {
   public:
    static void* alloc(size_t size) {
        auto ptr = malloc(size);
        return ptr;
    }

    static void dealloc(void* ptr) { free(ptr); }

    static void empty_dealloc(void* ptr) {}
};

template <typename AllocT = SimpleAllocater>
class __Any {
   public:
    struct by_value_tag {};
    struct by_reference_tag {};
    struct by_deepcopy_tag {};

   public:
    __Any() {}
    __Any(const __Any&) = delete;
    __Any& operator=(const __Any& other) = delete;

    __Any(__Any&& other) { *this = std::move(other); }

    __Any& operator=(__Any&& other) {
        if (this == &other) {
            return *this;
        }
        // buf_ = std::exchange(other.buf_, nullptr);
        buf_ = other.buf_;
        dealloctor_ = std::exchange(other.dealloctor_, &AllocT::empty_dealloc);
        castImpl_ = other.castImpl_;
        return *this;
    }

    template <typename T>
    __Any(T&& t, by_value_tag = {})
        : dealloctor_(
              &__Any::destruct<typename std::remove_reference<T>::type>) {
        using ValueT = typename std::remove_reference<T>::type;
        buf_ = AllocT::alloc(sizeof(T));
        new (buf_) ValueT(std::forward<T>(t));
        castImpl_ = reinterpret_cast<void*>(
            &__Any<AllocT>::as_bv<std::remove_reference_t<T>>);
    }

    template <typename T>
    __Any(T&& t, by_reference_tag) {
        buf_ = &t;
        castImpl_ = reinterpret_cast<void*>(
            &__Any<AllocT>::as_br<std::remove_reference_t<T>>);
    }

    template <typename T>
    __Any(T&& t, size_t size, by_deepcopy_tag) : dealloctor_(&AllocT::dealloc) {
        buf_ = AllocT::alloc(size);
        auto buf = reinterpret_cast<char*>(buf_);
        memcpy(buf, std::forward<T>(t), size);
        castImpl_ = reinterpret_cast<void*>(
            &__Any<AllocT>::as_dc<std::remove_reference_t<T>>);
    }

    ~__Any() { dealloctor_(buf_); }

    template <typename T>
    T& as() {
        return reinterpret_cast<T& (*)(__Any*)>(castImpl_)(this);
    }

    template <typename T>
    T& getReference() {
        return *reinterpret_cast<T*>(buf_);
    }

    template <typename T>
    operator T() {
        this->as<T>();
    }

    template <typename T>
    T release() {
        dealloctor_ = &AllocT::empty_dealloc;
        return std::move(this->as<T>());
    }

   private:
    template <typename T>
    static void destruct(void* buf) {
        auto ptr = reinterpret_cast<T*>(buf);
        ptr->T::~T();
        AllocT::dealloc(buf);
    }

    template <typename T>
    static T& as_dc(__Any* self) {
        return reinterpret_cast<T&>(self->buf_);
    }

    template <typename T>
    static T& as_bv(__Any* self) {
        using ValueT = typename std::remove_reference<T>::type;
        return *reinterpret_cast<ValueT*>(self->buf_);
    }

    template <typename T>
    static T& as_br(__Any* self) {
        using ValueT = typename std::remove_reference<T>::type;
        return *reinterpret_cast<ValueT*>(self->buf_);
    }

   private:
    void* buf_{nullptr};
    void* castImpl_{nullptr};
    using DellocPtrT = void (*)(void*);
    DellocPtrT dealloctor_{&AllocT::empty_dealloc};
};

using Any = __Any<>;

template <typename F, typename RF>
struct Invoker;

template <typename R, typename... Args>
struct Invoker<R(Args...), R> {
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

template <typename R, typename... Args, typename U>
struct Invoker<R(Args...), U> {
    template <size_t... idx>
    static R __InvokeImplement(void* funcPtr, Any* args,
                               std::index_sequence<idx...> = {}) {
        return reinterpret_cast<R (*)(Args...)>(funcPtr)(
            args[idx].template as<Args>()...);
    }

    static U InvokeImplement(void* funcPtr, Any* args) {
        static_assert(std::is_convertible<R, U>::value,
                      "R and U can't convertible!");
        return __InvokeImplement(funcPtr, args,
                                 std::make_index_sequence<sizeof...(Args)>());
    }
};

template <typename... Args>
struct Invoker<void, void(Args...)> {
    template <size_t... idx>
    static void __InvokeImplement(void* funcPtr, Any* args,
                                  std::index_sequence<idx...> = {}) {
        reinterpret_cast<void (*)(Args...)>(funcPtr)(
            args[idx].template as<Args>()...);
    }

    static void InvokeImplement(void* funcPtr, Any* args) {
        __InvokeImplement(funcPtr, args,
                          std::make_index_sequence<sizeof...(Args)>());
    }
};

template <typename R>
class FunctorBase {
   public:
    using InvokePtrT = R (*)(void*, Any*);
    InvokePtrT invoker_{nullptr};
    void* funcPtr_{nullptr};
    Any* args_{nullptr};
    FunctorBase(InvokePtrT i, void* f, Any* a)
        : invoker_(i), funcPtr_(f), args_(a) {}

    FunctorBase(FunctorBase&& other) { *this = std::move(other); }
    FunctorBase& operator=(FunctorBase&& other) {
        if (this == &other) {
            return *this;
        }
        std::swap(this->funcPtr_, other.funcPtr_);
        std::swap(this->invoker_, other.invoker_);
        this->args_ = std::exchange(other.args_, nullptr);
        return *this;
    }

    // TODO support
    FunctorBase(const FunctorBase& other) = delete;
    FunctorBase& operator=(const FunctorBase& other) = delete;
};

template <typename R>
class SpecialFunctorBase : public FunctorBase<R> {
   public:
    using Base = FunctorBase<R>;
    using Base::Base;
    using InvokePtrT = R (*)(void*, Any*);

    R operator()() { return this->invoker_(this->funcPtr_, this->args_); }
};

template <>
class SpecialFunctorBase<void> : public FunctorBase<void> {
   public:
    using Base = FunctorBase<void>;
    using Base::Base;
    using InvokePtrT = void (*)(void*, Any*);

    void operator()() { this->invoker_(this->funcPtr_, this->args_); }
};

template <typename U, typename AllocT = SimpleAllocater>
class Functor : public SpecialFunctorBase<U> {
   public:
    using Base = SpecialFunctorBase<U>;
    using Base::Base;

    template <typename R, typename... Args>
    explicit Functor(R (*ptr)(Args...))
        : Base(&Invoker<R(Args...), U>::InvokeImplement,
               reinterpret_cast<void*>(ptr),
               reinterpret_cast<Any*>(
                   AllocT::alloc(sizeof(Any) * sizeof...(Args)))),
          args_destructor_(args_destructor<sizeof...(Args)>) {
        for (size_t i = 0; i < sizeof...(Args); ++i) {
            (void)new (reinterpret_cast<void*>(this->args_ + i)) Any();
        }
    }

    Functor(Functor&& other) : Base(std::move(other)) {
        args_destructor_ = std::exchange(other.args_destructor_, nullptr);
    }
    Functor& operator=(Functor&& other) {
        if (this == &other) {
            return *this;
        }
        *static_cast<Base*>(this) = std::move(other);
        args_destructor_ = std::exchange(other.args_destructor_, nullptr);
        return *this;
    }

    ~Functor() {
        if (args_destructor_) args_destructor_(this->args_);
    }

    template <typename T>
    typename std::enable_if_t<
        !std::is_reference<std::remove_reference_t<T>>::value>
    capture(size_t argIndex, T&& arg) {
        this->args_[argIndex] = Any(std::forward<T>(arg), Any::by_value_tag());
    }

    template <typename T>
    void captureByReference(size_t argIndex, T&& arg) {
        this->args_[argIndex] =
            Any(std::forward<T>(arg), Any::by_reference_tag());
    }

    template <typename T>
    typename std::enable_if_t<
        std::is_rvalue_reference<std::remove_reference_t<T>>::value>
    capture(size_t argIndex, T&& arg) {
        this->args_[argIndex] = Any(std::forward<T>(arg), Any::by_value_tag());
    }

    template <typename T>
    void captureByDeepCopy(size_t argIndex, T&& arg, size_t size) {
        static_assert(
            std::is_pointer<typename std::remove_reference<T>::type>::value,
            "deep copy must pointer type.");
        this->args_[argIndex] =
            Any(std::forward<T>(arg), size, Any::by_deepcopy_tag());
    }

   private:
    template <size_t size>
    static void args_destructor(Any* any) {
        if (!any) return;
        for (size_t i = 0; i < size; ++i) {
            any[i].Any::~Any();
        }
        AllocT::dealloc(any);
    }

   private:
    void (*args_destructor_)(Any* any);
};

class OpFunctor : public Functor<Any> {
   public:
    template <typename R, typename... Args>
    OpFunctor(R (*ptr)(Args...)) : Functor<Any>::Functor(ptr) {
        result_ = R{};
        feed_placeholder_ = [this](Any& result) {
            result_.as<R>() = result.release<R>();
        };
    }

    void operator()() {
        Any result = Functor<Any>::operator()();
        feed_placeholder_(result);
    }

    // TODO: return placeHold type and call capture,
    // then remove capture by reference
    template <typename T>
    T& getResult() {
        return result_.as<T>();
    }

    Any* getArgs() { return this->args_; }

   private:
    Any result_;
    std::function<void(Any& result)> feed_placeholder_;
};

}  // namespace support
