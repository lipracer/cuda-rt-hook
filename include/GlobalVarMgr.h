#pragma once

#include <functional>

namespace hook {

using registry_func = std::function<void*(void)>;
using destroy_func = std::function<void(void*)>;

void init_all_global_variables();
void register_global_variable(int priority, const registry_func& rfunc,
                              const destroy_func& dfunc);

template <typename T, int Priority = 0>
class GlobalVarMgr {
   public:
    // why don't use std::forward, instead capture args by value always safe
    template <typename... Args>
    GlobalVarMgr(Args... args) {
        register_global_variable(
            Priority,
            [=]() {
                auto ptr = new T(args...);
                this->ptr_ = ptr;
                return ptr;
            },
            [](void* obj) { delete reinterpret_cast<T*>(obj); });
    }
    ~GlobalVarMgr() { ptr_ = nullptr; }

    T* operator->() { return ptr_; }
    const T* operator->() const { return ptr_; }

    T& operator*() { return *ptr_; }
    const T& operator*() const { return *ptr_; }

   private:
    T* ptr_;
};

}  // namespace hook