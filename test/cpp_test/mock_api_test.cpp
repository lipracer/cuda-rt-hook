#include "mock_api/mock_api.h"

#include <stdlib.h>

#include <algorithm>
#include <numeric>

#include "GlobalVarMgr.h"
#include "cuda_mock.h"
#include "elf_parser.h"
#include "gtest/gtest.h"
#include "hook.h"
#include "logger/logger_stl.h"

using namespace hook;

TEST(MockAnyHook, base) {
    dh_any_hook_install();
    int ret = mock::foo(nullptr);
    EXPECT_EQ(ret, 0);
}
