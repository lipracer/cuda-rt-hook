
#include <stdlib.h>

#include <algorithm>
#include <numeric>

#include "gtest/gtest.h"
#include "logger/logger.h"
#include "support.h"

using namespace support;

int wrap_rt_malloc(void** ptr, size_t size) {
    LOG(DEBUG) << __func__ << " ptr:" << ptr << " size:" << size;
    return 0;
}

int rt_malloc(void** ptr, size_t size) {
    LOG(DEBUG) << __func__ << " ptr:" << ptr << " size:" << size;
    Functor<int> functor(&wrap_rt_malloc);
    functor.capture(0, ptr);
    functor.capture(1, size);
    functor();
    return 0;
}

int wrap_set_parameter(size_t index, void* ptr, size_t size) {
    LOG(DEBUG) << __func__ << " index:" << index << " ptr:" << ptr
               << " size:" << size;
    return 0;
}

int set_parameter(size_t index, void* ptr, size_t size) {
    LOG(DEBUG) << __func__ << " index:" << index << " ptr:" << ptr
               << " size:" << size;
    Functor<int> functor(&wrap_set_parameter);
    functor.capture(0, index);
    functor.captureByDeepCopy(1, ptr, size);
    functor.capture(2, size);
    functor();
    return 0;
}

int wrap_config_hw(void* s, int ng, int nb) {
    LOG(DEBUG) << __func__ << " s:" << s << " ng:" << ng << " nb:" << nb;
    return 0;
}

int config_hw(void* s, int ng, int nb) {
    LOG(DEBUG) << __func__ << " s:" << s << " ng:" << ng << " nb:" << nb;
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

static auto scalar_add(int a, int b) { return a + b; }

static auto scalar_sub(int a, int b) { return a - b; }

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

    OpFunctor functor_s(scalar_add);
    functor_s.capture(0, accessor.getResult<int>());
    functor_s.capture(1, 2);

    functor();
    accessor();
    functor_s();
    EXPECT_EQ(functor_s.getResult<int>(), 8);
}

TEST(SupportTest, opfunctor_scalar) {
    int a = 1, b = 2;

    OpFunctor functor0(&scalar_add);
    functor0.capture(0, a);
    functor0.capture(1, b);

    OpFunctor functor1(&scalar_sub);
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
    os << "storage:" << tensor.storage.get();
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
    Tensor result = apply_empty<float>({3, 4});
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
    auto float_eq = [](float lhs, float rhs) {
        return std::abs(lhs - rhs) <= std::numeric_limits<float>::epsilon();
    };
    EXPECT_TRUE(std::equal(param0.element_begin<float>(),
                           param0.element_end<float>(),
                           tensor.element_begin<float>(), float_eq));
}
