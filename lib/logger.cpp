#include "logger.h"
namespace logger {

LogStream& LogStream::instance() {
    static LogStream __instance;
    return __instance;
}

}  // namespace logger