#pragma once
#include <string>

extern "C" {

void xpu_dh_initialize(bool use_improve);

void dh_patch_runtime();

void dh_start_capture_rt_print();
std::string dh_end_capture_rt_print();
}
