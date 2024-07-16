#pragma once

#include <stdlib.h>

#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "logger/StringRef.h"

namespace hook {

// c++ parse has three phase
// first: parse the legal template class
// second: instantiated class template parameters
// third: instantiated function template parameters
// if we use the function template we need implement all of the str2value_impl
// functions to support more type, but sometimes we need implement the
// str2value_impl near the type define

template <typename T>
struct str2value_impl {
    void operator()(T& value, adt::StringRef str,
                    std::enable_if_t<std::is_same<T, std::string>::value ||
                                     std::is_integral<T>::value>* = nullptr) {
        std::stringstream ss;
        ss << str;
        ss >> value;
    }
};

// TODO: return bool value to check parse result
template <>
struct str2value_impl<int> {
    void operator()(int& value, adt::StringRef str);
};

template <typename T>
struct str2value {
    T operator()(adt::StringRef str) {
        T ret;
        str2value_impl<T>()(ret, str);
        return ret;
    }
};

template <typename K, typename V>
struct str2value_impl<std::pair<K, V>> {
    void operator()(std::pair<K, V>& pair, adt::StringRef str) {
        size_t i = 0;
        for (; i < str.size() && str[i] != '\0'; ++i) {
            if (str[i] == '=') {
                pair.first = str2value<K>()(str.slice(0, i));
                pair.second = str2value<V>()(str.slice(i + 1));
                break;
            }
        }
    }
};

template <typename V>
struct str2value_impl<std::vector<V>> {
    void operator()(std::vector<V>& vec, adt::StringRef str) {
        size_t i = 0, j = 0;
        for (; j < str.size() && str[j] != '\0'; ++j) {
            if (str[j] == ',') {
                vec.push_back(str2value<V>()(str.slice(i, j)));
                i = j + 1;
            }
        }
        vec.push_back(str2value<V>()(str.slice(i, j)));
    }
};

template <typename T>
T get_env_value(adt::StringRef str,
                std::__void_t<decltype(std::declval<std::istringstream>() >>
                                       std::declval<T&>())>* = nullptr) {
    auto env_value_str = std::getenv(str.data());
    if (!env_value_str) {
        return {};
    }
    return str2value<T>()(env_value_str);
}

template <typename T, typename K = typename T::first_type,
          typename V = typename T::second_type>
typename std::enable_if<std::is_same<T, std::pair<K, V>>::value, T>::type
get_env_value(adt::StringRef str) {
    auto env_value_str = std::getenv(str.data());
    if (!env_value_str) {
        return {};
    }
    return str2value<T>()(env_value_str);
}

template <typename T, typename V = typename T::value_type>
typename std::enable_if<std::is_same<T, std::vector<V>>::value, T>::type
get_env_value(adt::StringRef str) {
    auto env_value_str = std::getenv(str.data());
    if (!env_value_str) {
        return {};
    }
    return str2value<T>()(env_value_str);
}

template <typename T>
inline typename std::enable_if<std::is_same<T, const char*>::value, T>::type
get_env_value(adt::StringRef str) {
    return std::getenv(str.data());
}

template <typename T>
inline typename std::enable_if<std::is_same<T, adt::StringRef>::value, T>::type
get_env_value(adt::StringRef str) {
    auto env_value_str = std::getenv(str.data());
    if (!env_value_str) {
        return {};
    }
    return adt::StringRef(env_value_str);
}

}  // namespace hook