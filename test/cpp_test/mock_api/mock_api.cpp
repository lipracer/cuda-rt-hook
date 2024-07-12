#include "mock_api.h"

#include <stdio.h>

namespace mock {
int foo(void* p) {
    printf("call mock api:%s args:%p\n", __func__, p);
    return 0;
}
int bar(void* p) {
    printf("call mock api:%s args:%p\n", __func__, p);
    return 0;
}
}  // namespace mock
