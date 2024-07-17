#pragma once

#include <stddef.h>
#include <string.h>

#include <algorithm>
#include <cassert>
#include <iosfwd>
#include <optional>
#include <stdexcept>
#include <string>

namespace adt {
class StringRef {
   public:
    using CharT = char;
    using iterator = const CharT*;
    using const_iterator = const CharT*;

    friend inline bool operator==(StringRef lhs, StringRef rhs);
    friend inline bool operator!=(StringRef lhs, StringRef rhs);
    friend inline std::ostream& operator<<(std::ostream& os, StringRef str);

    StringRef() : size_(0), str_(nullptr) {}

    StringRef(const char* str) : size_(strlen(str)), str_(str) {}

    StringRef(const char* str, size_t size)
        : size_(size != std::string::npos ? size : strlen(str_)), str_(str) {}

    StringRef(const_iterator begin, const_iterator end)
        : size_(std::distance(begin, end)), str_(begin) {}

    template <size_t N>
    StringRef(const char (&str)[N]) : size_(N), str_(str) {}

    StringRef(const std::string& str) : size_(str.size()), str_(str.c_str()) {}

    StringRef(const StringRef& other) = default;
    StringRef(StringRef&& other) = default;

    StringRef& operator=(const StringRef& other) = default;
    StringRef& operator=(StringRef&& other) = default;

    char operator[](size_t i) const {
        assert(i < size_);
        return str_[i];
    }

    const CharT* c_str() const { return str_; }

    bool empty() const { return !size_ || !str_; }

    size_t size() const { return size_; }

    std::string str() const {
        return size_ != 0 ? std::string(str_, str_ + size_) : std::string();
    }

    const char* data() const { return str_; }

    const_iterator begin() const { return str_; }
    const_iterator end() const { return str_ + size_; }

    std::string::size_type find(StringRef other) {
        auto p = strstr(str_, other.c_str());
        if (!p) {
            return std::string::npos;
        }
        return p - str_;
    }

    std::string::size_type contain(StringRef other) {
        return find(other) != std::string::npos;
    }

    std::string lower() const {
        std::string result(size(), 0);
        std::transform(begin(), end(), result.begin(),
                       [](int chr) { return std::tolower(chr); });
        return result;
    }
    std::string upper() const {
        std::string result(size(), 0);
        std::transform(begin(), end(), result.begin(),
                       [](int chr) { return std::toupper(chr); });
        return result;
    }

    StringRef drop_front(size_t size = 1) const {
        return StringRef(this->begin() + size, this->end());
    }
    StringRef drop_back(size_t size = 1) const {
        return StringRef(this->begin(), this->end() - size);
    }

    StringRef slice(size_t s, size_t e) { return StringRef(str_ + s, e - s); }
    StringRef slice(size_t s) { return this->slice(s, this->size()); }

    bool startsWith(StringRef prefix) {
        if (prefix.size() > this->size()) return false;
        return drop_back(this->size() - prefix.size()) == prefix;
    }
    bool endsWith(StringRef suffix) {
        if (suffix.size() > size()) return false;
        return drop_front(this->size() - suffix.size()) == suffix;
    }

    template <typename T, typename RetT = typename std::enable_if<
                              std::is_integral_v<T>, T>::type>
    std::optional<RetT> toIntegral() const {
        static_assert(sizeof(T) <= 4, "");
        auto std_lower_str = this->lower();
        auto lower_str = StringRef(std_lower_str);
        // special handiling for binary number
        int base = 0;
        if (lower_str.startsWith(StringRef("0b", 2))) {
            lower_str = lower_str.drop_front(2);
            base = 2;
        }
        try {
            return std::stoi(lower_str.str(), nullptr, base);
        } catch (const std::invalid_argument&) {
            return {};
        } catch (const std::out_of_range&) {
            return {};
        }
    }

   private:
    size_t size_;
    const CharT* str_;
};

inline bool operator==(StringRef lhs, StringRef rhs) {
    return lhs.size_ == rhs.size_ &&
           (lhs.str_ == rhs.str_ ||
            std::equal(lhs.begin(), lhs.end(), rhs.begin()));
}

inline bool operator!=(StringRef lhs, StringRef rhs) { return !(lhs == rhs); }

inline std::ostream& operator<<(std::ostream& os, StringRef str) {
    os << str.str();
    return os;
}

}  // namespace adt
