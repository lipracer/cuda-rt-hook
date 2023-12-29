#pragma once

#include <stdlib.h>

#include <sstream>
#include <string>
#include <utility>

namespace hook {

template <typename T>
T str2value(const char* str, int len = -1) {
    std::stringstream ss;
    if (len > 0) {
        ss << std::string(str, str + len);

    } else {
        ss << str;
    }
    T ret{};
    ss >> ret;
    return ret;
}

template <typename T>
typename std::enable_if<std::is_trivial<T>::value, T>::type get_env_value(
    const char* str) {
    auto env_str = std::getenv(str);
    if (!env_str) {
        return {};
    }
    return str2value<T>(env_str);
}

template <typename T, typename K = typename T::first_type,
          typename V = typename T::second_type>
typename std::enable_if<std::is_same<T, std::pair<K, V>>::value, T>::type
get_env_value(const char* str) {
    auto env_str = std::getenv(str);
    if (!env_str) {
        return {};
    }
    auto iter = env_str;
    while (*iter++ != '=')
        ;
    --iter;
    std::pair<K, V> ret;
    ret.first = str2value<K>(env_str, iter - env_str);
    ret.second = str2value<V>(iter + 1);
    return ret;
}

}  // namespace hook