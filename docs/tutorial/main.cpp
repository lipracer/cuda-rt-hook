#include <stdio.h>
#include "hook.h"
#include "hello.h"

void bye(){
    printf("byebye\n");
}

int main() {
    hello();
    plthook("hello", reinterpret_cast<void *>(bye));
    hello();
    return 0;
}
