#pragma once

#include <stdlib.h>

#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace hook {

template <typename T>
T str2value(
    const char* str, size_t len = std::string::npos,
    std::void_t<decltype(operator<<(std::stringstream(), std::declval<T>()))>* =
        nullptr) {
    std::stringstream ss;
    if (len != std::string::npos) {
        ss << std::string(str, str + len);

    } else {
        ss << str;
    }
    T ret{};
    ss >> ret;
    return ret;
}

template <typename T, typename K = typename T::first_type,
          typename V = typename T::second_type>
typename std::enable_if<std::is_same<T, std::pair<K, V>>::value, T>::type
str2value(const char* str, size_t len = std::string::npos) {
    std::pair<K, V> ret;
    for (size_t i = 0; i < len && str[i] != '\0'; ++i) {
        if (str[i] == '=') {
            ret.first = str2value<K>(str, i);
            ret.second = str2value<V>(str + i + 1);
            break;
        }
    }
    return ret;
}

template <typename T, typename V = typename T::value_type>
typename std::enable_if<std::is_same<T, std::vector<V>>::value, T>::type
str2value(const char* str, size_t len = std::string::npos) {
    std::vector<V> ret;
    size_t i = 0, j = 0;
    for (; j < len && str[j] != '\0'; ++j) {
        if (str[j] == ',') {
            ret.push_back(str2value<V>(str + i, str[j]));
            i = j + 1;
        }
    }
    ret.push_back(str2value<V>(str + i, j - i));
    return ret;
}

template <typename T>
typename std::enable_if<std::is_trivial<T>::value, T>::type get_env_value(
    const char* str) {
    auto env_value_str = std::getenv(str);
    if (!env_value_str) {
        return {};
    }
    return str2value<T>(env_value_str);
}

template <typename T, typename K = typename T::first_type,
          typename V = typename T::second_type>
typename std::enable_if<std::is_same<T, std::pair<K, V>>::value, T>::type
get_env_value(const char* str) {
    auto env_value_str = std::getenv(str);
    if (!env_value_str) {
        return {};
    }
    return str2value<T>(env_value_str);
}

template <typename T, typename V = typename T::value_type>
typename std::enable_if<std::is_same<T, std::vector<V>>::value, T>::type
get_env_value(const char* str) {
    auto env_value_str = std::getenv(str);
    if (!env_value_str) {
        return {};
    }
    return str2value<T>(env_value_str);
}

}  // namespace hook