#pragma once

#define DEF_ENVIRONMENT_STR(name) const char* name = #name;

namespace env_mgr {

extern const char* LOG_LEVEL;

extern const char* HOOK_ENABLE_TRACE;

extern const char* LOG_OUTPUT_PATH;

// a regex pattern, when the target lib name match this pattern, the lib's
// symbol will be replace
extern const char* TARGET_LIB_FILTER;

// don't use buffer cache log message, output message direct to stdcout
extern const char* LOG_SYNC_MODE;

}  // namespace env_mgr
