#pragma once

#include <vector>
#include <ostream>
#include "logger.h"

namespace logger {

using LoggerType = logger::LogStream;

template <typename T, typename AllocT>
inline LogStream& WriteToLoggerStream(LogStream& os,
                                      const std::vector<T, AllocT>& vec) {
    os.getStream() << "[";
    if (vec.empty()) {
        os.getStream() << "]";
        return os;
    } else {
        for (auto iter = std::begin(vec); iter != std::prev(std::end(vec));
             ++iter) {
            os.getStream() << *iter << ", ";
        }
        os.getStream() << vec.back() << "]";
    }

    return os;
}

}  // namespace logger
