#pragma once
#include <string>
#include <vector>
#include "logger/StringRef.h"


void dh_patch_runtime();


void __runtimeapi_hook_initialize();

void __print_hook_initialize(std::vector<adt::StringRef> &target_libs, std::vector<adt::StringRef> &target_symbols);

void __print_hook_start_capture();

std::string __print_hook_end_capture();
