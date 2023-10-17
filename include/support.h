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

template <typename T>
using VoidType = void;

class SimpleAllocater {
   public:
    static void* alloc(size_t size) {
        auto ptr = malloc(size);
        return ptr;
    }

    static void dealloc(void* ptr) { free(ptr); }
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
        destructor_ =
            std::exchange(other.destructor_, &__Any::destructor_empty);
        return *this;
    }

    template <typename T>
    constexpr __Any(T&& t, by_value_tag = {})
        : AnyTypeChecker(std::forward<T>(t)),
          destructor_(
              &__Any::destructor<typename std::remove_reference<T>::type>) {
        using ValueT = typename std::remove_reference<T>::type;
        buf_ = AllocT::alloc(sizeof(T));
        if (std::is_rvalue_reference<decltype(std::forward<T>(t))>::value) {
            new (buf_) ValueT(std::move(std::forward<T>(t)));
        } else {
            new (buf_) ValueT(std::forward<T>(t));
        }
    }

    template <typename T>
    constexpr __Any(T&& t, void* buf)
        : AnyTypeChecker(std::forward<T>(t)),
          buf_(buf),
          destructor_(
              &__Any::destructor<typename std::remove_reference<T>::type>) {
        (void)new (buf) typename std::remove_reference<T>::type(std::move(t));
    }

    template <typename T>
    constexpr __Any(T&& t, by_reference_tag)
        : AnyTypeChecker(std::forward<T>(t)) {
        static_assert(!std::is_rvalue_reference<T>::value,
                      "can not ref a right value!");
        buf_ = &t;
    }

    template <typename T>
    constexpr __Any(const T& t, by_reference_tag) : AnyTypeChecker(t) {
        buf_ = const_cast<void*>((void*)&t);
    }

    template <typename T>
    constexpr __Any(T t, size_t size, by_deepcopy_tag)
        : __Any(t, by_value_tag()) {
        auto buf = AllocT::alloc(size);
        memcpy(reinterpret_cast<char*>(buf), t, size);
        *reinterpret_cast<void**>(buf_) = buf;
    }

    ~__Any() { destructor_(buf_); }

    template <typename T>
    T& as() {
        this->check_islegal<T>();
        assert(buf_ && "unexpect cast empty buffer to object!");
        using ValueT = typename std::remove_reference<T>::type;
        return *reinterpret_cast<ValueT*>(buf_);
    }

    template <typename T>
    T& getReference() {
        return *reinterpret_cast<T*>(buf_);
    }

    template <typename T>
    operator T() {
        return this->as<T>();
    }

    operator bool() { return buf_; }

    bool operator!() { return !static_cast<bool>(*this); }

    template <typename T>
    T release() {
        destructor_ = &__Any::destructor_empty;
        return std::move(this->as<T>());
    }

    // The lifecycle of this object must be greater than the lifecycle of the
    // shared object, which may require a reference count
    void share_self(__Any& any) {
        static_cast<AnyTypeChecker&>(any) = static_cast<AnyTypeChecker&>(*this);
        any.buf_ = buf_;
        any.destructor_ = &__Any::destructor_empty;
    }

    void* data() const { return buf_; }

    template <typename T>
    void reverse() {
        buf_ = AllocT::alloc(sizeof(T));
    }

   private:
    template <typename T>
    static void destructor(void* buf) {
        auto ptr = reinterpret_cast<T*>(buf);
        ptr->T::~T();
        AllocT::dealloc(buf);
    }

    static void destructor_empty(void* buf) {}

   private:
    void* buf_{nullptr};
    using DellocPtrT = void (*)(void*);
    DellocPtrT destructor_{&__Any::destructor_empty};
};

using Any = __Any<>;

template <typename F, typename RF>
struct Invoker;

template <typename R, typename... Args>
struct Invoker<R(Args...), R> {
    template <size_t... idx>
    static inline R __InvokeImplement(void* funcPtr, Any* args,
                                      std::index_sequence<idx...> = {}) {
        return reinterpret_cast<R (*)(Args...)>(funcPtr)(
            args[idx].template as<Args>()...);
    }

    static inline R InvokeImplement(void* funcPtr, Any* args) {
        return __InvokeImplement(funcPtr, args,
                                 std::make_index_sequence<sizeof...(Args)>());
    }
};

template <typename R, typename... Args, typename U>
struct Invoker<R(Args...), U> {
    template <size_t... idx>
    static inline R __InvokeImplement(void* funcPtr, Any* args,
                                      std::index_sequence<idx...> = {}) {
        return reinterpret_cast<R (*)(Args...)>(funcPtr)(
            args[idx].template as<Args>()...);
    }

    static inline U InvokeImplement(void* funcPtr, Any* args) {
        static_assert(std::is_convertible<R, U>::value,
                      "R and U can't convertible!");
        return __InvokeImplement(funcPtr, args,
                                 std::make_index_sequence<sizeof...(Args)>());
    }
};

template <typename... Args>
struct Invoker<void(Args...), void> {
    template <size_t... idx>
    static inline void __InvokeImplement(void* funcPtr, Any* args,
                                         std::index_sequence<idx...> = {}) {
        reinterpret_cast<void (*)(Args...)>(funcPtr)(
            args[idx].template as<Args>()...);
    }

    static inline void InvokeImplement(void* funcPtr, Any* args) {
        __InvokeImplement(funcPtr, args,
                          std::make_index_sequence<sizeof...(Args)>());
    }
};

template <typename ClsT, typename R, typename... Args>
struct Invoker<R (ClsT::*)(Args...), R> {
    using MemPtr = R (ClsT::*)(Args...);

    template <size_t... idx>
    static inline R __InvokeImplement(Any* args,
                                      std::index_sequence<idx...> = {}) {
        MemPtr& m_ptr = args[0].as<MemPtr>();
        ClsT& self = args[1].as<ClsT&>();
        return (self.*m_ptr)(args[idx + 2].template as<Args>()...);
    }

    static inline R InvokeImplement(void*, Any* args) {
        return __InvokeImplement(args,
                                 std::make_index_sequence<sizeof...(Args)>());
    }
};

// write the output to last argument
template <typename ClsT, typename R, typename... Args>
struct Invoker<R (ClsT::*)(Args...), void> {
    using MemPtr = R (ClsT::*)(Args...);

    template <size_t... idx>
    static inline void __InvokeImplement(Any* args,
                                         std::index_sequence<idx...> = {}) {
        MemPtr& m_ptr = args[0].as<MemPtr>();
        ClsT& self = args[1].as<ClsT&>();
        R& result = (self.*m_ptr)(args[idx + 2].template as<Args>()...);
        args[2 + sizeof...(Args)] = Any(result, Any::by_reference_tag());
    }

    static inline void InvokeImplement(void*, Any* args) {
        __InvokeImplement(args, std::make_index_sequence<sizeof...(Args)>());
    }
};

// write the output to last argument
template <typename R, typename... Args>
struct Invoker<R(Args...), void> {
    template <size_t... idx>
    static inline void __InvokeImplement(void* ptr, Any* args,
                                         std::index_sequence<idx...> = {}) {
        auto func_ptr = reinterpret_cast<R (*)(Args...)>(ptr);
        R result = func_ptr(args[idx].template as<Args>()...);
        args[sizeof...(Args)] = Any(result, Any::by_reference_tag());
    }

    static inline void InvokeImplement(void* ptr, Any* args) {
        __InvokeImplement(ptr, args,
                          std::make_index_sequence<sizeof...(Args)>());
    }

    template <size_t... idx>
    static inline void __InvokeImplementW(void* ptr, Any* args,
                                          std::index_sequence<idx...> = {}) {
        auto func_ptr = reinterpret_cast<R (*)(Args...)>(ptr);
        void* buf = args[sizeof...(Args)].data();
        (void)new (args + sizeof...(Args))
            Any(func_ptr(args[idx].template as<Args>()...), buf);
    }

    static inline void InvokeImplementW(void* ptr, Any* args) {
        __InvokeImplementW(ptr, args,
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
    size_t argsSize_{0};

    FunctorBase() = default;
    constexpr FunctorBase(InvokePtrT i, void* f, size_t argsSize)
        : invoker_(i), funcPtr_(f) {
        resizeArgs(argsSize);
    }

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

    void resizeArgs(size_t argsSize) {
        if (args_) {
            for (size_t i = 0; i < argsSize_; ++i) {
                (args_ + i)->~__Any();
            }
            SimpleAllocater::dealloc(args_);
        }
        argsSize_ = argsSize;
        args_ = reinterpret_cast<Any*>(
            SimpleAllocater::alloc(sizeof(Any) * argsSize));
        for (size_t i = 0; i < argsSize; ++i) {
            (void)new (reinterpret_cast<void*>(args_ + i)) Any();
        }
    }

    size_t validArgSize() {
        size_t validSize = 0;
        for (; validSize < argsSize_; ++validSize) {
            if (!args_[validSize]) {
                break;
            }
        }
        return validSize;
    }
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
            (this->args_[this->validArgSize() + idx] = std::forward<Args>(args),
             0)...};
    }

    template <typename... Args>
    R operator()(Args&&... args) {
        captureExternArgs(std::make_index_sequence<sizeof...(args)>(),
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
            (this->args_[this->validArgSize() + idx] = std::forward<Args>(args),
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
    constexpr explicit Functor(R (*ptr)(Args...))
        : Base(&Invoker<R(Args...), U>::InvokeImplement,
               reinterpret_cast<void*>(ptr), sizeof...(Args)),
          args_destructor_(args_destructor<sizeof...(Args)>) {}

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

    void captureByReference(size_t argIndex, void* arg) {
        this->args_[argIndex] = Any(arg, Any::by_reference_tag());
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

    template <typename... Args>
    void captureVariadic(Args&&... args) {
        (void)std::initializer_list<int>{
            (capture(this->validArgSize(), std::forward<Args>(args)), 0)...};
    }

    template <typename... Args>
    void captureVariadicByReference(Args&&... args) {
        (void)std::initializer_list<int>{
            (captureByReference(this->validArgSize(), std::forward<Args>(args)),
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

class PlaceHolder {
   public:
    constexpr PlaceHolder(Any* any) : any_(any) {}

    auto operator[](size_t index);

    template <typename T>
    T& get() {
        return any_->as<T>();
    }
    Any* data() { return any_; }

   private:
    Any* any_{nullptr};
};

class ViewFunctor : public Functor<void> {
   public:
    using Functor<void>::Functor;
    using Base = Functor<void>;
    static constexpr size_t extraArgumentSize = 3;
    // using Base::args_destructor;
    template <typename ClsT, typename R, typename... Args>
    constexpr ViewFunctor(R (ClsT::*ptr)(Args...), PlaceHolder obj = nullptr)
        : Base(),
          resultIndex_(sizeof...(Args) + extraArgumentSize - 1),
          objIndex_(1),
          anyObj_(obj.data()) {
        // 1. member function point
        // 2. obj pointer
        // 3. output reference
        resizeArgs(extraArgumentSize + sizeof...(Args));
        this->capture(0, ptr);
        this->invoker_ = &Invoker<R (ClsT::*)(Args...), void>::InvokeImplement;
        this->args_destructor_ =
            &Base::template args_destructor<sizeof...(Args) +
                                            extraArgumentSize>;
    }

    template <typename... Args, size_t... idx>
    void unpack(size_t startIndex, std::index_sequence<idx...>,
                Args&&... args) {
        (void)std::initializer_list<int>{
            (capture(startIndex + idx, std::forward<Args>(args)), 0)...};
    }

    template <typename ClsT, typename R, typename... Args, typename... RArgs>
    ViewFunctor(R (ClsT::*ptr)(Args...), PlaceHolder obj, RArgs&&... args)
        : ViewFunctor(ptr, obj) {
        unpack(2, std::make_index_sequence<sizeof...(args)>(),
               static_cast<Args>(std::forward<RArgs>(args))...);
    }

    template <typename R, typename... Args>
    ViewFunctor(R (*ptr)(Args...), PlaceHolder obj)
        : Base(ptr), anyObj_(obj.data()) {
        // relalloc append output argument
        resizeArgs(sizeof...(Args) + 1);
        resultIndex_ = sizeof...(Args);
    }

    ViewFunctor(PlaceHolder obj) : Base(), anyObj_(obj.data()) {
        resizeArgs(1);
        objIndex_ = 0;
        resultIndex_ = 0;
        this->invoker_ = nullptr;
    }

    ViewFunctor(ViewFunctor&& other) : Base(std::move(other)) {
        resultIndex_ = other.resultIndex_;
        captureIndex_ = other.captureIndex_;
    }

    ViewFunctor& operator=(ViewFunctor&& other) {
        static_cast<Base&>(*this) = std::move(other);
        resultIndex_ = other.resultIndex_;
        captureIndex_ = other.captureIndex_;
        anyObj_ = other.anyObj_;
        objIndex_ = other.objIndex_;
        return *this;
    }

    template <typename ClsT, typename... Args>
    void operator()(ClsT& obj, Args&&... args) {
        this->captureByReference(1, obj);
        this->captureVariadic(std::forward<Args>(args)...);
        // assert(this->argSize() == resultIndex_ + 1 && "error argument
        // size!");
        this->invoker_(nullptr, this->args_);
    }

    void operator()() {
        assert(*anyObj_ && "can't share empty object!");
        anyObj_->share_self(this->args_[objIndex_]);
        if (this->invoker_) this->invoker_(this->funcPtr_, this->args_);
    }

    template <typename T>
    T& getResult() {
        assert(resultIndex_ && "index of result can't zero!");
        return this->args_[resultIndex_].as<T>();
    }

    void setCaptureIndex(int64_t n) { captureIndex_ = n; }

    void moveTo(Any* any) {
        assert(captureIndex_ >= 0 && "argument must >= 0!");
        assert(this->args_[resultIndex_] && "empty placehold value!");
        any[captureIndex_] = std::move(this->args_[resultIndex_]);
    }
    size_t resultIndex_{0};
    int64_t captureIndex_{-1};
    size_t objIndex_{0};
    Any* anyObj_{nullptr};
};

class OpFunctor;

class OpFunctor : public Functor<void> {
   public:
    using Functor<void>::capture;

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

    template <typename R, typename... Args>
    constexpr OpFunctor(R (*ptr)(Args...))
        : Functor<void>::Functor(ptr), argsSize_(sizeof...(Args)) {
        this->invoker_ = Invoker<R(Args...), void>::InvokeImplementW;
        this->resizeArgs(argsSize_ + 1);
        this->args_[argsSize_].reverse<R>();
    }

    template <typename... Args>
    constexpr OpFunctor(void (*ptr)(Args...))
        : Functor<void>::Functor(ptr), argsSize_(sizeof...(Args)) {
        this->invoker_ = Invoker<void(Args...), void>::InvokeImplement;
        this->resizeArgs(argsSize_);
    }

    template <typename... Args>
    void operator()(Args&&... args) {
        for (auto iter = lazyFeedFuncs_.begin(); iter != lazyFeedFuncs_.end();
             iter = iter->next) {
            iter->func();
            iter->func.moveTo(this->args_);
        }
        Functor<void>::operator()(std::forward<Args>(args)...);
    }

    void capture(size_t index, PlaceHolder ph) {
        lazyFeedFuncs_.emplace_back(ph);
        auto& back = lazyFeedFuncs_.back();
        back.func.setCaptureIndex(static_cast<int64_t>(index));
    }

    template <typename ClsT, typename R, typename... Args>
    void capture(size_t index, PlaceHolder placeHolder, R (ClsT::*ptr)(Args...),
                 Args... args) {
        lazyFeedFuncs_.emplace_back(ptr, placeHolder, args...);
        lazyFeedFuncs_.back().func.setCaptureIndex(static_cast<int64_t>(index));
    }

    template <typename R, typename... Args>
    void capture(size_t index, PlaceHolder placeHolder, R (*ptr)(Args...)) {
        lazyFeedFuncs_.emplace_back(ptr, placeHolder.data());
        lazyFeedFuncs_.back().func.setCaptureIndex(static_cast<int64_t>(index));
    }

    template <typename T>
    T& getResultValue() {
        return getAnyResult()->as<T>();
    }

    PlaceHolder getResult() { return PlaceHolder(this->getAnyResult()); }

    Any* getAnyResult() { return this->args_ + argsSize_; }

    Any* getArgs() { return this->args_; }

   private:
    // TODO: remove this and write the output to args and reserve the any
    // storage avoid call malloc at runtime
    size_t argsSize_{0};
    LazyFeedFuncs lazyFeedFuncs_;
};

}  // namespace support
