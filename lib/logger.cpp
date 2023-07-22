#include "logger.h"
namespace logger {

size_t StringLiteralBase::MN = 0;

LogStream& LogStream::instance() {
    static LogStream __instance;
    return __instance;
}

}  // namespace logger