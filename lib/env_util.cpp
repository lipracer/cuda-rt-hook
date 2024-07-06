#include "env_util.h"

namespace hook {

void str2value_impl<int>::operator()(int& value, const char* cstr, size_t len) {
    adt::StringRef str;
    if (len != std::string::npos) {
        str = adt::StringRef(cstr, cstr + len);
    } else {
        str = cstr;
    }
    auto result = str.toIntegral<int>();
    value = result.has_value() ? *result : 0;
}
}  // namespace hook
