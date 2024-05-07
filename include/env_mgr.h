#pragma once

#define DEF_ENVIRONMENT_STR(name) const char* name = #name;

namespace env_mgr {

extern const char* LOG_LEVEL;

extern const char* HOOK_ENABLE_TRACE;

extern const char* LOG_OUTPUT_PATH;

extern const char* TARGET_LIB_FILTER;

}  // namespace env_mgr
