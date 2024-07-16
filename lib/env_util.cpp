#include "env_util.h"

namespace hook {

void str2value_impl<int>::operator()(int& value, adt::StringRef str) {
    auto result = str.toIntegral<int>();
    value = result.has_value() ? *result : 0;
}
}  // namespace hook
