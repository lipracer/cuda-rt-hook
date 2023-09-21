
#include <stdlib.h>

#include <algorithm>
#include <numeric>

#include "gtest/gtest.h"
#include "logger/logger.h"
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
}

TEST(SupportTest, functor_ctor) {
    std::vector<support::Functor<int>> functors;

    Functor<int> functor(&rt_malloc);
    functor.capture(0, nullptr);
    functor.capture(1, 1);
    functors.emplace_back(std::move(functor));
}

template <typename T = int>
static auto scalar_add(T a, T b) {
    return a + b;
}

template <typename T = int>
static auto scalar_sub(T a, T b) {
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
    auto accessor = functor.getResult<VecInt>()[2];

    OpFunctor functor_s(scalar_add<>);
    functor_s.capture(0, accessor.getResult<int>());
    functor_s.capture(1, 2);

    functor();
    accessor();
    functor_s();
    EXPECT_EQ(functor_s.getResult<int>(), 8);
}

TEST(SupportTest, functor_member_func_direct_capture) {
    std::vector<int> vec_int = {1, 2, 3, 4, 5, 6};
    using VecInt = std::vector<int>;
    OpFunctor functor(vector_add);
    functor.capture(0, vec_int);
    functor.capture(1, vec_int);

    OpFunctor functor_s(scalar_add<>);
    functor_s.captureVector(0, functor.getResult<VecInt>(), 2);
    functor_s.capture(1, 2);

    functor();
    functor_s();
    EXPECT_EQ(functor_s.getResult<int>(), 8);
}

TEST(SupportTest, functor_member_func_partial_arg) {
    OpFunctor functor(scalar_add<>);
    functor.capture(0, 1);
    functor(2);
    EXPECT_EQ(functor.getResult<int>(), 3);
}

TEST(SupportTest, opfunctor_scalar) {
    int a = 1, b = 2;

    OpFunctor functor0(&scalar_add<>);
    functor0.capture(0, a);
    functor0.capture(1, b);

    OpFunctor functor1(&scalar_sub<>);
    functor1.capture(0, functor0.getResult<int>());
    functor1.capture(1, a);

    functor0();
    functor1();

    EXPECT_EQ(functor1.getResult<int>().get(), b);
}

using Dim = int64_t;

struct Tensor {
    std::shared_ptr<void> storage;
    std::vector<Dim> shape;
    Dim totalSize;

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
    Dim n =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<Dim>());
    return Tensor{.storage = std::shared_ptr<void>(new T[n]),
                  .shape = shape,
                  .totalSize = n};
}

template <typename T, template <typename> typename OpT>
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

static Tensor sub(Tensor lhs, Tensor rhs) {
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
    functor1.capture(0, functor0.getResult<Tensor>());
    functor1.capture(1, param1);

    functor0();
    functor1();

    Tensor& tensor = functor1.getResult<Tensor>();

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

    addFunc.capture(0, params.getResult<std::vector<Tensor>>(), 1);
    addFunc.capture(1, params.getResult<std::vector<Tensor>>(), 2);

    params(size_t{4});
    addFunc();

    for (auto& t : params.getResult<std::vector<Tensor>>().get()) {
        LOG(WARN) << t;
    }
    LOG(WARN) << addFunc.getResult<Tensor>();

    auto tensors = getVectorResult(size_t{4});
    auto direct_result = add(tensors[1], tensors[2]);
    EXPECT_TRUE(std::equal(direct_result.element_begin<float>(),
                           direct_result.element_end<float>(),
                           direct_result.element_begin<float>(), float_eq));
}

std::tuple<Tensor, size_t> getTupleResult() {
    std::tuple<Tensor, size_t> result;
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

    tensor_addFunc.capture<0>(0,
                              params.getResult<std::tuple<Tensor, size_t>>());
    tensor_addFunc.capture(1, rhs_tensor);

    scalar_addFunc.capture<1>(0,
                              params.getResult<std::tuple<Tensor, size_t>>());
    scalar_addFunc.capture(1, size_t(1));

    params();
    tensor_addFunc();
    auto& tt = params.getResult<std::tuple<Tensor, size_t>>().get();
    std::get<1>(tt) += 1;
    scalar_addFunc();

    LOG(WARN) << tensor_addFunc.getResult<Tensor>();
    LOG(WARN) << scalar_addFunc.getResult<size_t>();

    EXPECT_EQ(scalar_addFunc.getResult<size_t>(), 18);
}
