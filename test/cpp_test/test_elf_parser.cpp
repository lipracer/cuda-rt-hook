#include <stdlib.h>

#include <algorithm>
#include <numeric>

#include "GlobalVarMgr.h"
#include "elf_parser.h"
#include "gtest/gtest.h"
#include "hook.h"
#include "logger/logger_stl.h"

using namespace hook;

namespace {
int foo(void*) { return 0; }

void* org_foo = nullptr;

class SymbolParserHook : public hook::HookInstallerWrap<SymbolParserHook> {
   public:
    bool targetLib(const char* name) {
        return adt::StringRef(name).contain("libmock_api.so");
    }
    hook::HookFeature symbols[1] = {
        HookFeature("_ZN4mock3fooEPv", &foo, &org_foo)
            .setGetNewCallback([](const hook::OriginalInfo& info) {
                hook::createSymbolTable(info.libName, info.baseHeadPtr);
            })};

    void onSuccess() {}
};

}  // namespace

TEST(TestHookWrap, symbol_parser) {
    auto sh = std::make_shared<SymbolParserHook>();
    sh->install();
}

// TEST(MockAnyHook, symbol) {
//     hook::CachedSymbolTable ctb("./test/cpp_test/mock_api/libmock_api.so",
//     nullptr); auto strtab = ctb.load_section_data(".strtab"); LOG(WARN) <<
//     "strtab size:" << strtab.size();

//     for (auto it : ctb.getSymbolTable()) {
//         LOG(WARN) << it.second;
//     }
// }
