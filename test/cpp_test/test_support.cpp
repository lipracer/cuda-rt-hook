
#include <stdlib.h>

#include <algorithm>
#include <numeric>

#include "gtest/gtest.h"
#include "logger/logger_stl.h"
#include "support.h"

using namespace support;

int wrap_rt_malloc(void** ptr, size_t size) {
    LOG(WARN) << __func__ << " ptr:" << ptr << " size:" << size;
    return 0;
}

int rt_malloc(void** ptr, size_t size) {
    LOG(WARN) << __func__ << " ptr:" << ptr << " size:" << size;
    Functor<int> functor(&wrap_rt_malloc);
    functor.capture(0, ptr);
    functor.capture(1, size);
    functor();
    return 0;
}

int wrap_set_parameter(size_t index, void* ptr, size_t size) {
    LOG(WARN) << __func__ << " index:" << index << " ptr:" << ptr
              << " size:" << size;
    return 0;
}

int set_parameter(size_t index, void* ptr, size_t size) {
    LOG(WARN) << __func__ << " index:" << index << " ptr:" << ptr
              << " size:" << size;
    Functor<int> functor(&wrap_set_parameter);
    functor.capture(0, index);
    functor.captureByDeepCopy(1, ptr, size);
    functor.capture(2, size);
    functor();
    return 0;
}

int wrap_config_hw(void* s, int ng, int nb) {
    LOG(WARN) << __func__ << " s:" << s << " ng:" << ng << " nb:" << nb;
    return 0;
}

int config_hw(void* s, int ng, int nb) {
    LOG(WARN) << __func__ << " s:" << s << " ng:" << ng << " nb:" << nb;
    Functor<int> functor(&wrap_config_hw);
    functor.captureByDeepCopy(0, s, 4);
    functor.capture(1, ng);
    functor.capture(2, nb);
    functor();
    return 0;
}

static const int gLogDummp =
    (logger::initLogger(logger::LogConfig{.mode = logger::LogConfig::kSync}),
     0);

TEST(SupportTest, functor_malloc) {
    void* ptr = nullptr;
    rt_malloc(&ptr, 100);
}

TEST(SupportTest, functor_deepcopy) {
    const size_t buffer_size = 100;
    void* ptr = malloc(buffer_size);
    set_parameter(0, ptr, buffer_size);
}

TEST(SupportTest, functor_any_cast) {
    // test by value to any ahnd move constructor
    {
        int n = 100;
        support::Any any(n);
        EXPECT_EQ(any.as<int>(), n);

        auto other_any = std::move(any);
        EXPECT_EQ(other_any.as<int>(), n);
    }

    // test by deepcopy to any ahnd move constructor
    {
        const size_t size = 100;
        std::vector<int> vec(size, 0);
        std::iota(vec.begin(), vec.end(), 0);
        support::Any any(vec.data(), sizeof(int) * size,
                         support::Any::by_deepcopy_tag());
        auto other_any = std::move(any);
        auto buf = other_any.as<int*>();
        EXPECT_TRUE(std::equal(vec.begin(), vec.end(), buf));
    }

    // test by reference to any ahnd move constructor
    {
        size_t size = 100;
        support::Any any(size, support::Any::by_reference_tag());
        auto other_any = std::move(any);
        auto& value = other_any.as<size_t&>();
        EXPECT_EQ(size, value);
    }

    void* ptr = malloc(4);
    config_hw(ptr, 10, 10);
}

TEST(SupportTest, functor_any_destructor) {
    const size_t size = 10;
    // test any destructor
    {
        auto ptr = new int[size];
        std::iota(ptr, ptr + size, 0);
        std::shared_ptr<int> sp(ptr);
        {
            support::Any any(sp);
            EXPECT_EQ(sp.use_count(), 2);
            auto any_sp = any.as<std::shared_ptr<int>>();
            for (size_t i = 0; i < size; ++i) {
                EXPECT_EQ(ptr[i], any_sp.get()[i]);
            }
        }
        EXPECT_EQ(sp.use_count(), 1);
    }
    // test functor destructor
    {
        auto ptr = new int[size];
        std::shared_ptr<int> sp(ptr);
        void (*func_ptr)(std::shared_ptr<int>) = nullptr;
        {
            auto functor = std::make_unique<support::Functor<void>>(func_ptr);
            EXPECT_EQ(sp.use_count(), 1);
            functor->capture(0, sp);
            EXPECT_EQ(sp.use_count(), 2);
        }
        EXPECT_EQ(sp.use_count(), 1);
    }
}

TEST(SupportTest, any_improve) {
    std::shared_ptr<int> sp(new int);
    support::Any any = sp;
    EXPECT_EQ(sp.use_count(), 2);
    auto sp0 = any.release<std::shared_ptr<int>>();
    EXPECT_EQ(sp.use_count(), 2);
    Any empty_any;
    EXPECT_FALSE(empty_any);
    EXPECT_TRUE(!empty_any);
}

TEST(SupportTest, any_ref_log) {
    using Type = std::vector<int>;
    Type vec_int = {0, 1, 2, 3};
    Any any(vec_int, Any::by_reference_tag());
    Type& vec = any.as<Type>();
    LOG(WARN) << vec;
}

TEST(SupportTest, functor_ctor) {
    std::vector<support::Functor<int>> functors;

    Functor<int> functor(&rt_malloc);
    functor.capture(0, nullptr);
    functor.capture(1, 1);
    functors.emplace_back(std::move(functor));
}

template <typename T = int>
static T scalar_add(T a, T b) {
    return a + b;
}

template <typename T = int>
static T scalar_sub(T a, T b) {
    return a - b;
}

static auto vector_add(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> result(a.size());
    std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                   [](auto lhs, auto rhs) { return lhs + rhs; });
    return result;
}

static auto vector_sub(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> result(a.size());
    std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                   [](auto lhs, auto rhs) { return lhs + rhs; });
    return result;
}

TEST(SupportTest, functor_member_func) {
    std::vector<int> vec_int = {1, 2, 3, 4, 5, 6};
    using VecInt = std::vector<int>;
    OpFunctor functor(vector_add);
    functor.capture(0, vec_int);
    functor.capture(1, vec_int);

    OpFunctor functor_s(&scalar_add<int>);
    functor_s.capture(0, functor.getResult(), &VecInt::operator[], size_t(2));
    functor_s.capture(1, 2);

    functor();
    LOG(INFO) << functor.getResult().get<VecInt>();
    functor_s();
    EXPECT_EQ(functor_s.getResultValue<int>(), 8);
}

TEST(SupportTest, functor_member_func_direct_capture) {
    std::vector<int> vec_int = {1, 2, 3, 4, 5, 6};
    using VecInt = std::vector<int>;
    OpFunctor functor(vector_add);
    functor.capture(0, vec_int);
    functor.capture(1, vec_int);

    OpFunctor functor_s(&scalar_add<int>);
    functor_s.capture(0, functor.getResult(), &VecInt::operator[], size_t(2));
    functor_s.capture(1, 2);

    functor();
    functor_s();
    EXPECT_EQ(functor_s.getResultValue<int>(), 8);
}

TEST(SupportTest, functor_member_func_partial_arg) {
    OpFunctor functor(&scalar_add<int>);
    functor.capture(0, 1);
    functor(2);
    EXPECT_EQ(functor.getResultValue<int>(), 3);
}

TEST(SupportTest, opfunctor_scalar) {
    int a = 1, b = 2;

    OpFunctor functor0(&scalar_add<int>);
    functor0.capture(0, a);
    functor0.capture(1, b);

    OpFunctor functor1(&scalar_sub<int>);
    functor1.capture(0, functor0.getResult());
    functor1.capture(1, a);

    functor0();
    functor1();

    EXPECT_EQ(functor1.getResultValue<int>(), b);
}

using Dim = int64_t;

struct Tensor {
    std::vector<Dim> shape;
    Dim totalSize;
    size_t ByteSize;
    std::shared_ptr<void> storage;

    Tensor() = default;

    Tensor(const std::vector<Dim>& shape, size_t elementSize)
        : shape(shape),
          totalSize(std::accumulate(shape.begin(), shape.end(), 1,
                                    std::multiplies<Dim>())),
          ByteSize(totalSize * elementSize),
          storage(new char[ByteSize]) {}

    Tensor(const Tensor& other) { *this = other; }

    Tensor(Tensor&& other) { *this = std::move(other); }

    Tensor& operator=(const Tensor& other) {
        shape = other.shape;
        totalSize = other.totalSize;
        ByteSize = other.ByteSize;
        storage = other.storage;
        return *this;
    }
    Tensor& operator=(Tensor&& other) {
        shape = std::move(other.shape);
        totalSize = other.totalSize;
        ByteSize = other.ByteSize;
        storage = std::move(other.storage);
        return *this;
    }

    template <typename T>
    T& getElement(Dim index) const {
        return reinterpret_cast<T*>(storage.get())[index];
    }

    template <typename T>
    using iterator = T*;

    template <typename T>
    iterator<T> element_begin() const {
        return &getElement<T>(0);
    }

    template <typename T>
    iterator<T> element_end() const {
        return &getElement<T>(totalSize);
    }

    Tensor clone() {
        Tensor result(shape, ByteSize / totalSize);
        memcpy(result.storage.get(), storage.get(), ByteSize);
        return result;
    }
};

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << "storage:" << tensor.storage.get() << " data:[";
    for (Dim i = 0; i < tensor.totalSize; ++i) {
        os << *(tensor.element_begin<float>() + i) << " ";
    }
    os << "]";
    return os;
}

template <typename T>
Tensor apply_empty(const std::vector<Dim>& shape) {
    return Tensor(shape, sizeof(T));
}

template <typename T, template <typename> class OpT>
static Tensor arith_op(const Tensor& lhs, const Tensor& rhs) {
    size_t n = std::accumulate(lhs.shape.begin(), lhs.shape.end(), 1,
                               std::multiplies<>());
    Tensor result = apply_empty<float>(lhs.shape);
    for (size_t i = 0; i < n; ++i) {
        result.getElement<float>(i) =
            OpT<T>()(lhs.getElement<float>(i), rhs.getElement<float>(i));
    }
    return result;
}

static Tensor add(const Tensor& lhs, const Tensor& rhs) {
    return arith_op<float, std::plus>(lhs, rhs);
}

static Tensor sub(const Tensor& lhs, const Tensor& rhs) {
    return arith_op<float, std::minus>(lhs, rhs);
}

auto float_eq(float lhs, float rhs) {
    return std::abs(lhs - rhs) <= std::numeric_limits<float>::epsilon();
}

TEST(SupportTest, opfunctor_tensor) {
    Tensor param0 = apply_empty<float>({3, 4});
    Tensor param1 = apply_empty<float>({3, 4});
    std::iota(param0.element_begin<float>(), param0.element_end<float>(), 0);
    std::iota(param1.element_begin<float>(), param1.element_end<float>(), 0);

    OpFunctor functor0(&add);
    functor0.capture(0, param0);
    functor0.capture(1, param1);

    OpFunctor functor1(&sub);
    functor1.capture(0, functor0.getResult());
    functor1.capture(1, param1);

    functor0();
    functor1();

    Tensor& tensor = functor1.getResultValue<Tensor>();

    EXPECT_TRUE(std::equal(param0.element_begin<float>(),
                           param0.element_end<float>(),
                           tensor.element_begin<float>(), float_eq));
}

static std::vector<Tensor> getVectorResult(size_t size) {
    std::vector<Tensor> result;
    std::vector<Dim> shape{2, 3};
    for (size_t i = 0; i < size; ++i) {
        result.push_back(apply_empty<float>(shape));
        for (Dim j = 0; j < result.back().totalSize; ++j) {
            result.back().element_begin<float>()[j] = i * 2 * 3 + j + 1;
        }
    }
    return result;
}

TEST(SupportTest, opfunctor_accessor_vector) {
    OpFunctor params(&getVectorResult);
    OpFunctor addFunc(&add);

    addFunc.capture(0, params.getResult(), &std::vector<Tensor>::operator[],
                    size_t(1));
    addFunc.capture(1, params.getResult(), &std::vector<Tensor>::operator[],
                    size_t(2));

    params(size_t{4});
    addFunc();

    for (auto& t : params.getResultValue<std::vector<Tensor>>()) {
        LOG(WARN) << t;
    }
    LOG(WARN) << addFunc.getResultValue<Tensor>();

    auto tensors = getVectorResult(size_t{4});
    auto direct_result = add(tensors[1], tensors[2]);
    EXPECT_TRUE(std::equal(direct_result.element_begin<float>(),
                           direct_result.element_end<float>(),
                           direct_result.element_begin<float>(), float_eq));
}

std::tuple<Tensor, size_t> getTupleResult() {
    std::vector<Dim> shape{2, 3};
    auto tensor = apply_empty<float>(shape);
    for (Dim j = 0; j < tensor.totalSize; ++j) {
        tensor.element_begin<float>()[j] = j + 1;
    }

    return std::make_tuple(tensor, 16);
}

TEST(SupportTest, opfunctor_accessor_tuple) {
    OpFunctor params(&getTupleResult);
    OpFunctor tensor_addFunc(&add);
    OpFunctor scalar_addFunc(&scalar_add<size_t>);

    auto rhs_tensor = apply_empty<float>({2, 3});
    for (Dim j = 0; j < rhs_tensor.totalSize; ++j) {
        rhs_tensor.element_begin<float>()[j] = 6 + j + 1;
    }

    Tensor& (*get0)(std::tuple<Tensor, size_t>&) = &std::get<0, Tensor, size_t>;
    size_t& (*get1)(std::tuple<Tensor, size_t>&) = &std::get<1, Tensor, size_t>;

    tensor_addFunc.capture(0, params.getResult(), get0);
    tensor_addFunc.capture(1, rhs_tensor);

    scalar_addFunc.capture(0, params.getResult(), get1);
    scalar_addFunc.capture(1, size_t(1));

    params();
    tensor_addFunc();
    auto& tt = params.getResultValue<std::tuple<Tensor, size_t>>();
    std::get<1>(tt) += 1;
    scalar_addFunc();

    LOG(WARN) << tensor_addFunc.getResultValue<Tensor>();
    LOG(WARN) << scalar_addFunc.getResultValue<size_t>();

    EXPECT_EQ(scalar_addFunc.getResultValue<size_t>(), 18);
}

TEST(SupportTest, functor_pure_view) {
    size_t index0 = 1;
    using Type = std::vector<int>;
    Type vec_int = {0, 1, 2, 3};
    ViewFunctor view0(&Type::operator[]);
    view0(vec_int, index0);
    view0.getResult<int>() += 1;
    LOG(WARN) << vec_int;
    EXPECT_EQ(vec_int[index0], 2);

    Any any(vec_int, Any::by_reference_tag());
    PlaceHolder ph(&any);

    size_t index1 = 2;
    ViewFunctor view1(&Type::operator[], ph, index1);
    view1();
    view1.getResult<int>() += 1;
    EXPECT_EQ(vec_int[index1], 3);

    size_t index2 = 3;
    ViewFunctor view2(&Type::operator[], ph, index2);
    view2();
    view2.getResult<int>() += 1;
    EXPECT_EQ(vec_int[index2], 4);
}

struct DurationDuard {
    DurationDuard() : st(std::chrono::steady_clock::now()) {}
    ~DurationDuard() {
        LOG(ERROR) << "cost:"
                   << std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - st)
                          .count()
                   << "ms";
    }
    decltype(std::chrono::steady_clock::now()) st;
};

// the performence of our functor and lambda function is very closely
// we have get the statistic of the performence so that reduce the loop count to
// reduce test case cost
constexpr size_t kLoopCount = 10000;
std::initializer_list<Dim> kPerfShape{64};

TEST(SupportTest, lambda_tensor_perf) {
    Tensor param0 = apply_empty<float>(kPerfShape);
    Tensor param1 = apply_empty<float>(kPerfShape);
    std::iota(param0.element_begin<float>(), param0.element_end<float>(), 0);
    std::iota(param1.element_begin<float>(), param1.element_end<float>(), 0);

    auto param0_clone_lambda = param0.clone();
    auto param1_clone_lambda = param1.clone();

    using FunctorType = std::function<void(void)>;

    void* add_ptr = reinterpret_cast<void*>(&add);
    void* sub_ptr = reinterpret_cast<void*>(&sub);

    Tensor lambda_result;
    FunctorType lambda_add = [&param0_clone_lambda, &param1_clone_lambda,
                              &lambda_result, add_ptr]() {
        auto result = (*reinterpret_cast<decltype(&add)>(add_ptr))(
            param0_clone_lambda, param1_clone_lambda);
        lambda_result = std::move(result);
    };

    Tensor lambda_tensor;
    FunctorType lambda_sub = [&lambda_result, &param1_clone_lambda,
                              &lambda_tensor, sub_ptr]() {
        lambda_tensor = (*reinterpret_cast<decltype(&sub)>(sub_ptr))(
            lambda_result, param1_clone_lambda);
    };

    Tensor plain_result;
    {
        DurationDuard _;
        for (size_t i = 0; i < kLoopCount; ++i) {
            auto add_ret = add(param0, param1);
            plain_result = sub(add_ret, param1);
        }
    }
    {
        DurationDuard _;
        for (size_t i = 0; i < kLoopCount; ++i) {
            lambda_add();
            lambda_sub();
        }
    }

    EXPECT_TRUE(std::equal(lambda_tensor.element_begin<float>(),
                           lambda_tensor.element_end<float>(),
                           plain_result.element_begin<float>(), float_eq));
}

TEST(SupportTest, opfunctor_tensor_perf) {
    Tensor param0 = apply_empty<float>(kPerfShape);
    Tensor param1 = apply_empty<float>(kPerfShape);
    std::iota(param0.element_begin<float>(), param0.element_end<float>(), 0);
    std::iota(param1.element_begin<float>(), param1.element_end<float>(), 0);

    auto param0_clone = param0.clone();
    auto param1_clone = param1.clone();

    OpFunctor functor0(&add);
    functor0.captureByReference(0, param0_clone);
    functor0.captureByReference(1, param1_clone);

    OpFunctor functor1(&sub);
    functor1.capture(0, functor0.getResult());
    functor1.captureByReference(1, param1_clone);

    Tensor plain_result;
    {
        DurationDuard _;
        for (size_t i = 0; i < kLoopCount; ++i) {
            auto add_ret = add(param0, param1);
            plain_result = sub(add_ret, param1);
        }
    }
    LOG(INFO) << "start funcor perf";
    {
        DurationDuard _;
        for (size_t i = 0; i < kLoopCount; ++i) {
            functor0();
            functor1();
        }
    }
    Tensor& tensor = functor1.getResultValue<Tensor>();

    EXPECT_TRUE(std::equal(tensor.element_begin<float>(),
                           tensor.element_end<float>(),
                           plain_result.element_begin<float>(), float_eq));
}

TEST(SupportTest, opfunctor_tensor_perf_single) {
    Tensor param0 = apply_empty<float>({128});
    Tensor param1 = apply_empty<float>({128});
    std::iota(param0.element_begin<float>(), param0.element_end<float>(), 0);
    std::iota(param1.element_begin<float>(), param1.element_end<float>(), 0);

    Tensor lambda_result;
    auto lambda_add = [&param0, &param1, &lambda_result]() {
        auto result = add(param0, param1);
        lambda_result = std::move(result);
    };

    auto param0_clone = param0.clone();
    auto param1_clone = param1.clone();

    OpFunctor functor0(&add);
    functor0.captureByReference(0, param0_clone);
    functor0.captureByReference(1, param1_clone);
    constexpr size_t loop_count = 100000;
    Tensor plain_result;

    {
        DurationDuard _;
        for (size_t i = 0; i < loop_count; ++i) {
            lambda_add();
        }
    }

    {
        DurationDuard _;
        for (size_t i = 0; i < loop_count; ++i) {
            functor0();
        }
    }

    Tensor& tensor = functor0.getResultValue<Tensor>();

    EXPECT_TRUE(std::equal(tensor.element_begin<float>(),
                           tensor.element_end<float>(),
                           lambda_result.element_begin<float>(), float_eq));
}
