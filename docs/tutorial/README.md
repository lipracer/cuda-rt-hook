# 背景

### 动态链接
C++中，动态链接是比静态链接更常用的一种链接方式。


### 动态链接的实现



# 一个prototype plthook实现


在这个tutorial中，我们将通过一个简单的例子来演示如何使用plthook来改变被调用的的函数。具体来讲，main中会调用libhello.so中的`hello()`。

```c++
#include "hello.h" //引入了定义在libhello.so中的hello函数

int main() {
    hello(); //打印hello
}
```

我们的目标是在运行时将`hello()`替换成另一个函数`bye()`:

```c++
#include "hello.h" //引入了定义在libhello.so中的hello函数

void bye() {
    printf("bye\n");
}

int main() {
    plthook("hello", bye); //替换hello为bye
    hello(); //打印bye
}
```

如果你对python足够了解，你会发现这非常类似python的monkey patch。确实如此，plthook所达到的目的和moneky patch非常类似，但由于涉及动态链接库的底层实现，要复杂的多。

## 准备工作

### 编写libhello.so

首先进行一些准备工作，比如编写libhello.so，这个so特别简单，只定义了一个hello函数，全部代码如下：

- hello.h
```c++
#ifndef LIBHELLO_H
#define LIBHELLO_H


extern "C" {
    void hello();
}

#endif // LIBHELLO_H
```

- hello.cpp
```c++
#include <stdio.h>
#include "hello.h"

void hello() {
    printf("hello\n");
}
```

最后使用 `g++ -shared -fPIC -o lib/libhello.so lib/src/hello.cpp -Ilib/include` 编译生成动态链接库`libhello.so`；可以通过 `nm -C` 来验证`hello`符号的存在。
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241204014247.png)

### 编写main

```c++
#include "hello.h"

int main() {
    hello();
    return 0;
}
```

编译并运行：
```
> g++ -std=c++20 main.cpp -o main -L lib -lhello -I lib/include
> LD_LIBRARY_PATH=./lib ./main
hello #输出hello
```
这里使用了c++20标准，因为后面大量用到了`std::format`。

## 编写plthook

在实际编写代码之前，先实际想想我们要做什么：

1. 目前的情况是，我们有一个main可执行文件，其中的main函数调用了libhello.so中的`hello()`函数。所以main可执行文件运行时，这个进程的虚拟内存中至少存在 main 和 libhello.so。当然，还有其他很多东西，比如libc.so等，这里就省略了。
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241204015004.png)

2. 回顾动态链接的实现方式，实际上main调用hello()的过程如下图所示，先在GOT中查找hello的地址，然后跳转到该地址执行。注意到main、libhello.so都有自己的GOT，当libhello.so调用printf时，实际上也是通过查找GOT表中的printf的地址，然后跳转执行。

![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241204020031.png)

3. 现在，在main中新增bye()，我们的目标是main在调用hello()时，实际上执行的是bye()

![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241204020143.png)

4. 为了达到这个目的，我们需要修改GOT表中的hello的地址，使其指向bye()。这里bye调用了printf的时候，也会在GOT中printf这个entry处查找printf的地址，在图中也画出来了

![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241204020448.png)

所以，我们要做的事情就很清晰了：
1. 对于Memory Space中的加载的每个elf文件（比如main libhello.so libc.so）都有自己的GOT，我们需要找到main的GOT
2. main的GOT中有多个entry，至少有hello printf等，我们需要找到hello的entry
3. 修改hello的entry，使其指向bye()
其中第3步是最简单的，就像python monkey patch一样，直接赋值即可，但1、2是比较复杂的，接下来会重点说
### 