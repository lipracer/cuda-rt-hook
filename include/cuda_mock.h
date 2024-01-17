#pragma once

#include <functional>

#define HOOK_API __attribute__((__visibility__("default")))

#ifdef __cplusplus

extern "C" {

typedef void* HookContext_t;
typedef const char* HookString_t;

void dh_initialize();

void dh_uninitialize();

void dh_internal_install_hook(HookString_t srcLib, HookString_t targetLib,
                              HookString_t symbolName,
                              HookString_t hookerLibPath = nullptr,
                              HookString_t hookerSymbolName = nullptr);

void dh_internal_install_hook_regex(HookString_t srcLib, HookString_t targetLib,
                                    HookString_t symbolName,
                                    HookString_t hookerLibPath,
                                    HookString_t hookerSymbolName);

HookContext_t dh_open_hook_context();

HookContext_t dh_close_hook_context();

void dh_set_symbol_define_library(HookContext_t ctx, HookString_t str,
                                  bool regex);

void dh_set_symbol_replace_library(HookContext_t ctx, HookString_t str,
                                   bool regex);

void dh_set_new_symbol_define_library(HookContext_t ctx, HookString_t str,
                                      bool regex);

void dh_set_new_symbol_name(HookContext_t ctx, HookString_t str, bool regex);

void dh_create_py_hook_installer(
    const std::function<bool(const char* name)>& isTarget,
    const std::function<bool(const char* name)>& isSymbol, HookString_t lib);
}

#endif