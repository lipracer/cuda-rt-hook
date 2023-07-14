#pragma once

namespace cuda_mock {

void initialize();

void internal_install_hook(const char* srcLib, const char* targetLib,
                           const char* symbolName,
                           const char* hookerLibPath = nullptr,
                           const char* hookerSymbolName = nullptr);

} // cuda_mock