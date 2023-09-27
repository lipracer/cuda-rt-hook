#pragma once
#include <stdlib.h>

#include <cassert>
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

// some project forbidden rtti
// #define SUPPORT_CHECK_TYPE
#ifdef SUPPORT_CHECK_TYPE
struct AnyTypeChecker {
    const std::type_info* type_info_{nullptr};
    std::string msg_;
    template <typename T>
    AnyTypeChecker(T&& t) : type_info_(&typeid(T)) {}
    AnyTypeChecker() = default;
    std::string name() const { return __support__demangle(type_info_->name()); }
    template <typename T>
    void check_islegal() {
        if (!type_info_) assert(false && "empty type info!");
        msg_ += std::string("cast ") + __support__demangle(type_info_->name()) +
                " to " + __support__demangle(typeid(T).name());
        assert(typeid(T) == *type_info_ && "type mismatch!");
    }
    const std::string& msg() const { return msg_; }
};
#else
struct AnyTypeChecker {
    std::string msg_;
    template <typename T>
    AnyTypeChecker(T&& t) {}
    AnyTypeChecker() = default;
    std::string name() const { return {}; }
    template <typename T>
    void check_islegal() {}
    std::string msg() const { return {}; }
};
#endif

template <typename AllocT = SimpleAllocater>
class __Any : public AnyTypeChecker {
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
        static_cast<AnyTypeChecker&>(*this) =
            static_cast<AnyTypeChecker&>(other);
        // buf_ = std::exchange(other.buf_, nullptr);
        buf_ = other.buf_;
        dealloctor_ = std::exchange(other.dealloctor_, &AllocT::empty_dealloc);
        castImpl_ = other.castImpl_;
        return *this;
    }

    template <typename T>
    __Any(T&& t, by_value_tag = {})
        : AnyTypeChecker(std::forward<T>(t)),
          dealloctor_(
              &__Any::destruct<typename std::remove_reference<T>::type>) {
        using ValueT = typename std::remove_reference<T>::type;
        buf_ = AllocT::alloc(sizeof(T));
        new (buf_) ValueT(std::forward<T>(t));
        castImpl_ = reinterpret_cast<void*>(
            &__Any<AllocT>::as_bv<std::remove_reference_t<T>>);
    }

    template <typename T>
    __Any(T&& t, by_reference_tag) : AnyTypeChecker(std::forward<T>(t)) {
        static_assert(!std::is_rvalue_reference<T>::value,
                      "can not ref a right value!");
        buf_ = &t;
        castImpl_ = reinterpret_cast<void*>(
            &__Any<AllocT>::as_br<std::remove_reference_t<T>>);
    }

    template <typename T>
    __Any(T* obj, by_reference_tag) : AnyTypeChecker(*obj) {
        buf_ = obj;
        castImpl_ = reinterpret_cast<void*>(&__Any<AllocT>::as_opaque);
    }

    template <typename T>
    __Any(T&& t, size_t size, by_deepcopy_tag)
        : AnyTypeChecker(std::forward<T>(t)), dealloctor_(&AllocT::dealloc) {
        buf_ = AllocT::alloc(size);
        auto buf = reinterpret_cast<char*>(buf_);
        memcpy(buf, std::forward<T>(t), size);
        castImpl_ = reinterpret_cast<void*>(
            &__Any<AllocT>::as_dc<std::remove_reference_t<T>>);
    }

    ~__Any() { dealloctor_(buf_); }

    template <typename T>
    T& as() {
        this->check_islegal<T>();
        assert(buf_ && "unexpect cast empty buffer to object!");
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

    operator bool() { return buf_; }

    bool operator!() { return !static_cast<bool>(*this); }

    template <typename T>
    T release() {
        dealloctor_ = &AllocT::empty_dealloc;
        return std::move(this->as<T>());
    }

    void* data() const { return buf_; }

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

    static void* as_opaque(__Any* self) { return self->buf_; }

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
struct Invoker<void(Args...), void> {
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

template <typename ClsT, typename R, typename... Args>
struct Invoker<R (ClsT::*)(Args...), R> {
    using MemPtr = R (ClsT::*)(Args...);

    template <size_t... idx>
    static R __InvokeImplement(Any* args, std::index_sequence<idx...> = {}) {
        MemPtr& m_ptr = args[0].as<MemPtr>();
        ClsT& self = args[1].as<ClsT&>();
        return (self.*m_ptr)(args[idx + 2].template as<Args>()...);
    }

    static R InvokeImplement(void*, Any* args) {
        return __InvokeImplement(args,
                                 std::make_index_sequence<sizeof...(Args)>());
    }
};

// write the output to last argument
template <typename ClsT, typename R, typename... Args>
struct Invoker<R (ClsT::*)(Args...), void> {
    using MemPtr = R (ClsT::*)(Args...);

    template <size_t... idx>
    static void __InvokeImplement(Any* args, std::index_sequence<idx...> = {}) {
        MemPtr& m_ptr = args[0].as<MemPtr>();
        ClsT& self = args[1].as<ClsT&>();
        R& result = (self.*m_ptr)(args[idx + 2].template as<Args>()...);
        args[2 + sizeof...(Args)] = Any(result, Any::by_reference_tag());
    }

    static void InvokeImplement(void*, Any* args) {
        __InvokeImplement(args, std::make_index_sequence<sizeof...(Args)>());
    }
};

// write the output to last argument
template <typename R, typename... Args>
struct Invoker<R(Args...), void> {
    template <size_t... idx>
    static void __InvokeImplement(void* ptr, Any* args,
                                  std::index_sequence<idx...> = {}) {
        auto func_ptr = reinterpret_cast<R (*)(Args...)>(ptr);
        R& result = func_ptr(args[idx].template as<Args>()...);
        args[sizeof...(Args)] = Any(result, Any::by_reference_tag());
    }

    static void InvokeImplement(void* ptr, Any* args) {
        __InvokeImplement(ptr, args,
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
    size_t currentArgIndex_{0};

    FunctorBase() = default;
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
        currentArgIndex_ = other.currentArgIndex_;
        return *this;
    }

    void increaseBindSize() { ++currentArgIndex_; }
    size_t argSize() const { return currentArgIndex_ + 1; };

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

    template <typename... Args, size_t... idx>
    void captureExternArgs(std::index_sequence<idx...>, Args&&... args) {
        (void)std::initializer_list<int>{
            (this->args_[this->currentArgIndex_ + idx] =
                 std::forward<Args>(args),
             0)...};
    }

    template <typename... Args>
    R operator()(Args&&... args) {
        captureExternArgs(std::make_index_sequence<sizeof...(Args)>(),
                          std::forward<Args>(args)...);
        return this->invoker_(this->funcPtr_, this->args_);
    }

    R operator()() { return this->invoker_(this->funcPtr_, this->args_); }
};

template <>
class SpecialFunctorBase<void> : public FunctorBase<void> {
   public:
    using Base = FunctorBase<void>;
    using Base::Base;
    using InvokePtrT = void (*)(void*, Any*);

    template <typename... Args, size_t... idx>
    void captureExternArgs(std::index_sequence<idx...>, Args&&... args) {
        (void)std::initializer_list<int>{
            (this->args_[this->currentArgIndex_ + idx] =
                 std::forward<Args>(args),
             0)...};
    }

    template <typename... Args>
    void operator()(Args&&... args) {
        captureExternArgs(std::make_index_sequence<sizeof...(Args)>(),
                          std::forward<Args>(args)...);
        this->invoker_(this->funcPtr_, this->args_);
    }

    void operator()() { this->invoker_(this->funcPtr_, this->args_); }
};

template <typename U, typename AllocT = SimpleAllocater>
class Functor : public SpecialFunctorBase<U> {
   public:
    using Base = SpecialFunctorBase<U>;
    using Base::Base;

    Functor() = default;

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
        this->increaseBindSize();
        this->args_[argIndex] = Any(std::forward<T>(arg), Any::by_value_tag());
    }

    template <typename T>
    void captureByReference(size_t argIndex, T&& arg) {
        this->increaseBindSize();
        this->args_[argIndex] =
            Any(std::forward<T>(arg), Any::by_reference_tag());
    }

    void captureByReference(size_t argIndex, void* arg) {
        this->increaseBindSize();
        this->args_[argIndex] = Any(arg, Any::by_reference_tag());
    }

    template <typename T>
    typename std::enable_if_t<
        std::is_rvalue_reference<std::remove_reference_t<T>>::value>
    capture(size_t argIndex, T&& arg) {
        this->increaseBindSize();
        this->args_[argIndex] = Any(std::forward<T>(arg), Any::by_value_tag());
    }

    template <typename T>
    void captureByDeepCopy(size_t argIndex, T&& arg, size_t size) {
        this->increaseBindSize();
        static_assert(
            std::is_pointer<typename std::remove_reference<T>::type>::value,
            "deep copy must pointer type.");
        this->args_[argIndex] =
            Any(std::forward<T>(arg), size, Any::by_deepcopy_tag());
    }

    template <typename... Args>
    void captureVariadic(Args&&... args) {
        (void)std::initializer_list<int>{
            (capture(this->currentArgIndex_, std::forward<Args>(args)), 0)...};
    }

    template <typename... Args>
    void captureVariadicByReference(Args&&... args) {
        (void)std::initializer_list<int>{
            (captureByReference(this->currentArgIndex_,
                                std::forward<Args>(args)),
             0)...};
    }

   protected:
    template <size_t size>
    static void args_destructor(Any* any) {
        if (!any) return;
        for (size_t i = 0; i < size; ++i) {
            any[i].Any::~Any();
        }
        AllocT::dealloc(any);
    }

   protected:
    void (*args_destructor_)(Any* any){nullptr};
};

class ViewFunctor : public Functor<void> {
   public:
    using Functor<void>::Functor;
    using Base = Functor<void>;
    static constexpr size_t extraArgumentSize = 3;
    // using Base::args_destructor;
    template <typename ClsT, typename R, typename... Args>
    ViewFunctor(R (ClsT::*ptr)(Args...))
        : Base(), resultIndex_(sizeof...(Args) + extraArgumentSize - 1) {
        // 1. member function point
        // 2. obj pointer
        // 3. output reference
        this->args_ = reinterpret_cast<Any*>(SimpleAllocater::alloc(
            sizeof(Any) * (extraArgumentSize + sizeof...(Args))));
        for (size_t i = 0; i < sizeof...(Args) + extraArgumentSize; ++i) {
            (void)new (reinterpret_cast<void*>(this->args_ + i)) Any();
        }
        this->capture(0, ptr);
        this->invoker_ = &Invoker<R (ClsT::*)(Args...), void>::InvokeImplement;
        this->args_destructor_ =
            &Base::template args_destructor<sizeof...(Args) +
                                            extraArgumentSize>;
    }

    template <typename ClsT, typename R, typename... Args>
    ViewFunctor(R (ClsT::*ptr)(Args...), ClsT& obj, Args... args)
        : ViewFunctor(ptr) {
        this->captureByReference(1, obj);
        this->captureVariadic(args...);
    }

    template <typename ClsT, typename R, typename... Args>
    ViewFunctor(R (ClsT::*ptr)(Args...), Any&& obj, Args... args)
        : ViewFunctor(ptr) {
        this->args_[1] = std::move(obj);
        this->increaseBindSize();
        this->captureVariadic(args...);
    }

    template <typename R, typename... Args, typename... RArgs>
    ViewFunctor(R (*ptr)(Args...), RArgs&&... args) : Base(ptr) {
        // relalloc append output argument
        this->args_ = reinterpret_cast<Any*>(
            SimpleAllocater::alloc(sizeof(Any) * (sizeof...(Args) + 1)));
        for (size_t i = 0; i < sizeof...(Args) + 1; ++i) {
            (void)new (args_ + i) Any();
        }
        captureVariadicByReference(std::forward<RArgs>(args)...);
        resultIndex_ = sizeof...(Args);
    }

    ViewFunctor(ViewFunctor&& other) : Base(std::move(other)) {
        resultIndex_ = other.resultIndex_;
        captureIndex_ = other.captureIndex_;
    }

    ViewFunctor& operator=(ViewFunctor&& other) {
        static_cast<Base&>(*this) = std::move(other);
        resultIndex_ = other.resultIndex_;
        captureIndex_ = other.captureIndex_;
        return *this;
    }

    template <typename ClsT, typename... Args>
    void operator()(ClsT& obj, Args&&... args) {
        this->captureByReference(1, obj);
        this->captureVariadic(std::forward<Args>(args)...);
        assert(this->argSize() == resultIndex_ + 1 && "error argument size!");
        this->invoker_(nullptr, this->args_);
    }

    void operator()() {
        assert(this->argSize() == resultIndex_ + 1 && "error argument size!");
        this->invoker_(this->funcPtr_, this->args_);
    }

    template <typename T>
    T& getResult() {
        assert(resultIndex_ && "index of result can't zero!");
        return this->args_[resultIndex_].as<T>();
    }

    void setCaptureIndex(int64_t n) { captureIndex_ = n; }

    void moveTo(Any* any) {
        assert(captureIndex_ >= 0 && "argument must >= 0!");
        any[captureIndex_] = std::move(this->args_[resultIndex_]);
    }
    size_t resultIndex_{0};
    int64_t captureIndex_{-1};
};

template <typename T>
auto __internal_operator_square_brackets(T& t, size_t index) {
    return t[index];
}

class OpFunctor;

template <typename T>
class PlaceHolder : public std::reference_wrapper<T> {
   public:
    using std::reference_wrapper<T>::reference_wrapper;

    auto operator[](size_t index);
};

template <>
class PlaceHolder<Any> : public std::reference_wrapper<Any> {
   public:
    using std::reference_wrapper<Any>::reference_wrapper;
};

class OpFunctor : public Functor<Any> {
   public:
    using Functor<Any>::capture;

    struct LazyFeedFuncs {
        struct Node {
            ViewFunctor func;
            Node* next{nullptr};
        };

        using Iterator = Node*;
        using Reference = Node&;

        Iterator begin() { return head; }
        Iterator end() { return last ? last->next : nullptr; }

        Node* head{nullptr};
        Node* last{nullptr};

        LazyFeedFuncs() = default;
        ~LazyFeedFuncs() { postorder_release(); }

        void push_back(ViewFunctor&& view) {
            auto func = new (SimpleAllocater::alloc(sizeof(Node))) Node();
            func->func = std::move(view);
            func->next = nullptr;
            if (!last) {
                last = head = func;
            } else {
                last->next = func;
                last = func;
            }
        }

        template <typename... Args>
        void emplace_back(Args&&... args) {
            auto func = new (SimpleAllocater::alloc(sizeof(Node))) Node();
            func->func = ViewFunctor(std::forward<Args>(args)...);
            func->next = nullptr;
            if (!last) {
                last = head = func;
            } else {
                last->next = func;
                last = func;
            }
        }

        Node& back() { return *last; }

        void postorder_release() {
            auto cur = head;
            postorder_release_impl(cur);
        }

        void postorder_release_impl(Node* node) {
            if (!node) {
                return;
            }
            if (node->next) {
                postorder_release_impl(node->next);
            }
            node->func.~ViewFunctor();
            SimpleAllocater::dealloc(&node->func);
        }
    };

    template <typename R>
    static void __feed_placeholder(Any* self, Any* result) {
        self->as<R>() = result->release<R>();
    }

    template <typename R, typename... Args>
    OpFunctor(R (*ptr)(Args...))
        : Functor<Any>::Functor(ptr), feed_placeholder_(__feed_placeholder<R>) {
        // TODO: maybe some type has not default ctor,
        // we need use view of any type refer the result, and modify function
        // __feed_placeholder
        result_ = R{};
    }

    template <typename... Args>
    void operator()(Args&&... args) {
        for (auto iter = lazyFeedFuncs_.begin(); iter != lazyFeedFuncs_.end();
             iter = iter->next) {
            iter->func();
            iter->func.moveTo(this->args_);
        }
        Any result = Functor<Any>::operator()(std::forward<Args>(args)...);
        feed_placeholder_(&result_, &result);
    }

    template <typename T>
    void capture(size_t index, PlaceHolder<T> ph) {
        Functor<Any>::captureByReference(index, ph.get());
    }

    template <typename ClsT, typename R, typename... Args>
    void capture(size_t index, PlaceHolder<Any> placeHolder,
                 R (ClsT::*ptr)(Args...), Args... args) {
        lazyFeedFuncs_.emplace_back(ptr, placeHolder.get(), args...);
        lazyFeedFuncs_.back().func.setCaptureIndex(int64_t(index));
    }

    template <typename T, typename ClsT, typename R, typename... Args>
    void capture(size_t index, PlaceHolder<T> placeHolder,
                 R (ClsT::*ptr)(Args...), Args... args) {
        lazyFeedFuncs_.emplace_back(ptr, placeHolder.get(), args...);
        lazyFeedFuncs_.back().func.setCaptureIndex(int64_t(index));
    }

    template <typename R, typename... Args>
    void capture(size_t index, PlaceHolder<Any> placeHolder,
                 R (*ptr)(Args...)) {
        lazyFeedFuncs_.emplace_back(ptr, placeHolder.get());
        lazyFeedFuncs_.back().func.setCaptureIndex(int64_t(index));
    }

    template <typename T, typename R, typename... Args>
    void capture(size_t index, PlaceHolder<T> placeHolder, R (*ptr)(Args...)) {
        lazyFeedFuncs_.emplace_back(ptr, placeHolder.get());
        lazyFeedFuncs_.back().func.setCaptureIndex(int64_t(index));
    }

    template <typename T>
    PlaceHolder<T> getResult() {
        return PlaceHolder<T>(result_.as<T>());
    }

    Any* getResult() { return &result_; }

    Any* getArgs() { return this->args_; }

   private:
    Any result_;
    Functor<void> feed_placeholder_;
    LazyFeedFuncs lazyFeedFuncs_;
};

template <typename T>
auto PlaceHolder<T>::operator[](size_t index) {
    OpFunctor accessor(__internal_operator_square_brackets<T>);
    accessor.captureByReference(0, this->get());
    accessor.capture(1, index);
    return accessor;
}

}  // namespace support
