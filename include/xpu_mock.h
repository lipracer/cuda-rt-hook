#pragma once
#include <string>
#include <vector>

#include "logger/StringRef.h"

/// @brief replace xpu_set_device or cuda_set_device
void dh_patch_runtime();

/// @brief replace more runtime api to profile api performance and arguments
void __runtimeapi_hook_initialize();

/// @brief config which library and which printf like symbols will be replace
/// @param target_libs
/// @param target_symbols
void __print_hook_initialize(const std::vector<adt::StringRef> &target_libs,
                             const std::vector<adt::StringRef> &target_symbols);

/// @brief start capture printf output
void __print_hook_start_capture();

/// @brief end capture printf output and return all of printf's output
/// @return
std::string __print_hook_end_capture();
