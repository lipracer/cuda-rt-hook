#pragma once

#include <stdlib.h>

#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace hook {

template <typename T>
void str2value_impl(T& value, const char* str, size_t len = std::string::npos,
                    std::enable_if_t<std::is_same<T, std::string>::value ||
                                     std::is_integral<T>::value>* = nullptr) {
    std::stringstream ss;
    if (len != std::string::npos) {
        ss << std::string(str, str + len);
    } else {
        ss << str;
    }
    ss >> value;
}

template <typename T>
T str2value(const char* str, size_t len = std::string::npos);

template <typename T, typename K = typename T::first_type,
          typename V = typename T::second_type>
std::enable_if_t<std::is_same<std::pair<K, V>, T>::value> str2value_impl(
    T& pair, const char* str, size_t len = std::string::npos) {
    size_t i = 0;
    for (; i < len && str[i] != '\0'; ++i) {
        if (str[i] == '=') {
            pair.first = str2value<K>(str, i);
            pair.second = str2value<V>(str + i + 1);
            break;
        }
    }
    if (i == '\0' || i == len) {
        pair.first = str2value<K>(str, i);
    }
}

template <typename T, typename V = typename T::value_type>
std::enable_if_t<std::is_same<std::vector<V>, T>::value> str2value_impl(
    T& vec, const char* str, size_t len = std::string::npos) {
    size_t i = 0, j = 0;
    for (; j < len && str[j] != '\0'; ++j) {
        if (str[j] == ',') {
            vec.push_back(str2value<V>(str + i, j));
            i = j + 1;
        }
    }
    vec.push_back(str2value<V>(str + i, j - i));
}

template <typename T>
T str2value(const char* str, size_t len) {
    T ret;
    str2value_impl(ret, str, len);
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