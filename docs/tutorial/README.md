# [Tutorial] 一个prototype plthook实现


在这个tutorial中，我们将通过一个简单的例子来演示如何使用plthook来改变被调用的的函数。具体来讲，main中会调用libhello.so中的`hello()`。

```c++
#include "hello.h" //引入了定义在libhello.so中的hello函数

int main() {
    hello(); //打印hello
}
```

我们的**目标**是在运行时将`hello()`替换成另一个函数`bye()`:

```c++
#include "hello.h" //引入了定义在libhello.so中的hello函数

void bye() {
    printf("byebye\n");
}

int main() {
    hello(); //打印hello
    plthook("hello", bye); //替换hello为bye
    hello(); //打印byebye
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
    printf("Hello, World\n");
}
```
这里使用了extern "C" 防止编译器对函数名进行mangling，简化实现。

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
- 对于Memory Space中的加载的每个elf文件（比如main libhello.so libc.so）都有自己的GOT，我们需要找到main的GOT
- main的GOT中有多个entry，至少有hello printf等，我们需要找到hello的entry
- 修改hello的entry，使其指向bye()
其中最后一步是最简单的，就像python monkey patch一样，直接赋值即可，但前面2步比较复杂，接下来会重点说明。

### 1. 遍历所有的elf(elf=so+main): dl_iterate_phdr

这里我们借助一个GNU C Library (glibc)提供的Linux库函数 [dl_iterate_phdr](https://man7.org/linux/man-pages/man3/dl_iterate_phdr.3.html)。

该函数的作用是遍历当前进程中的每一个shared object以及main，并调用一个回调函数。这个回调函数就是我们需要实现的。
```c++
int dl_iterate_phdr(
    int (*callback)(struct dl_phdr_info *info,size_t size, void *data), //回调函数
    void *data //回调函数参数
);
```
在cuda-rt-hook的tutorial中，附带了一个callback的实现，具体请参考以下函数：
```c++
int retrieve_dyn_lib(struct dl_phdr_info* info, size_t info_size, void* vecPltTables);
```
其中`dl_phdr_info`提供了shared object 或者 main（以下统称elf）的信息，该结构体的定义如下：
```c++
//定义在/usr/include/link.h
struct dl_phdr_info {
               ElfW(Addr)        dlpi_addr;  /* Base address of object */
               const char       *dlpi_name;  /* (Null-terminated) name of
                                                object */
               const ElfW(Phdr) *dlpi_phdr;  /* Pointer to array of
                                                ELF program headers
                                                for this object */
               ElfW(Half)        dlpi_phnum; /* # of items in dlpi_phdr */
    //其他成员略
}
```
以上这个4个成员都非常重要：
1. dlpi_addr：该elf在进程地址空间的基地址
2. dlpi_name：该elf的名字，具体来讲，是绝对路径，比如 `/lib/x86_64-linux-gnu/libpthread.so.0` 或者 `/root/miniconda/envs/python310_torch25_cuda/lib/python3.10/site-packages/torch/lib/libtorch_python.so`。特殊的，对于main，它的dlpi_name为空
3. dlpi_phdr & dlpi_phnum：该elf加载到进程中之后，所拥有的Segment Header数组，数组长度为dlpi_phnum。Elfw(Phdr)是一个宏，可以是Elf32_Phdr或者Elf64_Phdr，取决于进程是32位还是64位的，以Elf32_Phdr为例，定义如下：
```c++
typedef struct
{
    Elf64_Word	p_type;		// Segment type
    Elf64_Word	p_flags;	// Segment flags
    Elf64_Off	p_offset;	// Segment file offset
    Elf64_Addr	p_vaddr;	// Segment virtual address
    Elf64_Addr	p_paddr;	// Segment physical address
    Elf64_Xword	p_filesz;	// Segment size in file
    Elf64_Xword	p_memsz;	// Segment size in memory
    Elf64_Xword	p_align;	// Segment alignment
} Elf64_Phdr;
```
这里我们只关注其中2项：
- p_type：该Segment的类型，比如PT_LOAD、PT_DYNAMIC等，这里我们只关心PT_DYNAMIC段
- p_vaddr：该Segment在进程地址空间中的基址，结合dlpi_addr，可以计算出该elf在进程地址空间中的基址:
```c++
segment_base_addr = info->dlpi_addr + info->dlpi_phdr[x].p_vaddr;
```

此外，dl_iterate_phdr遍历的顺序为main第一，其他shared object按照加载顺序遍历；不过这个特性在这并没什么用，只是顺便提一下。



### 2. 找到PT_DYNAMIC段
在这一步，我们需要通过`p_type`找到了每个elf的PT_DYNAMIC段，并通过`p_vaddr + dlpi_addr`得到了PT_DYNAMIC段在进程地址空间中的地址。

而我们为什么要特别关注PT_DYNAMIC段呢？因为通过PT_DYNAMIC段，能找到动态链接PLT表的地址，进而找到GOT表的地址；此外，PT_DYNAMIC段中还有符号表的地址、字符串表的地址，这些信息都是必要的。

PT_DYNAMIC段，即.dynamic section是一个Elf64_Dyn结构数组，定义如下：
```c++
typedef struct {
        Elf64_Xword d_tag;
        union {
                Elf64_Xword     d_val;
                Elf64_Addr      d_ptr;
        } d_un;
} Elf64_Dyn;
```

d_tag是可能为以下值中的任意一种：
```c++
//完整列表见elf.h
/* Legal values for d_tag (dynamic entry type).  */

#define DT_NULL		0		/* Marks end of dynamic section */
#define DT_NEEDED	1		/* Name of needed library */
#define DT_PLTRELSZ	2		/* Size in bytes of PLT relocs */
#define DT_PLTGOT	3		/* Processor defined value */
#define DT_HASH		4		/* Address of symbol hash table */
#define DT_STRTAB	5		/* Address of string table */
#define DT_SYMTAB	6		/* Address of symbol table */
#define DT_RELA		7		/* Address of Rela relocs */
#define DT_RELASZ	8		/* Total size of Rela relocs */
#define DT_RELAENT	9		/* Size of one Rela reloc */
...
```
随着d_tag发生变化，其d_un或者取d_val，或者取d_ptr；比如
- 当d_tag是`DT_STRTAB`是，d_un就代表了字符串表在进程地址空间中的基址；
- 当d_tag是`DT_STRSZ`时，d_un就代表了字符串表的大小；当
- 当d_tag是`DT_SYMTAB`时，d_un就代表了符号表在进程地址空间中的基址；

我们需要遍历`.dynamic` section的每一个Elf64_Dyn结构体，找到其中d_tag为`DT_SYMTAB`、`DT_STRTAB`、`DT_JMPREL`和的Elf64_Dyn，它们分别对应 符号表、字符串表、PLT表的起始地址；以及PLT表的长度：`DT_PLTRELSZ`。

### 3. 符号表、字符串表的区别

这里顺便说一下符号表和字符串表的区别是什么。符号表（SYMTAB）里没有具体的char []类型的符号名，而又是一个结构数组：
```c++
typedef struct {
        Elf64_Word      st_name;
        ... //其他暂时不重要
} Elf64_Sym;
```

st_name是一个index，这个index代表符号名在字符串表（STRTAB）中的索引。而符号表是许多个 null-string 拼接而成的，从index开始，到下一个 `\0`，就是符号的名字；如下图所示
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241217184832.png)

### 4. PLT表 GOT表

PLT表是一个如下结构体组成的数组：
```c++
typedef struct
{
  Elf64_Addr	r_offset;		/* Address */
  Elf64_Xword	r_info;			/* Relocation type and symbol index */
  Elf64_Sxword	r_addend;		/* Addend */
} Elf64_Rela;
```

其中，`r_addend`不重要，忽略。

至此，我们有了最关键的两个拼图：
- `r_offset`是该符号的GOT表项的地址；准确来说，是在虚拟地址空间中，相对于动态库起始地址（前面的`dlpi_addr`）的偏移；我们可以通过修改该地址指向的GOT表项，来修改该符号实际对应的外部符号
- `r_info`的低32位是符号的重定位类型（不重要），而高32位是符号在符号表中的index，`/usr/include/elf.h`中定义了2个宏来辅助计算：
```c++
#define ELF64_R_SYM(i)			((i) >> 32) //符号在符号表中的index
#define ELF64_R_TYPE(i)			((i) & 0xffffffff) //符号的重定位类型
```
通过在符号表 & 字符串表中查询 该符合的名字，我们可以精准匹配需要被修改的符号，否则如果所有符合的GOT表都被修改，那意义不大，程序也会立刻crash。

我们可以用一张图来总结上面的过程，从`dl_phdr_info`开始，最终找到GOT表。
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241218000807.png)

### 5. 改写GOT表
找到GOT表后，下一步就是使用bye来替换hello。这一步是所有步骤中最简单的，`*got_addr = bye`就可以了：
``` c++
void bye(){
    printf("byebye\n");
}

void hook_plt(...){
...
    void * got_addr = reinterpret_cast<void*>(plt_entry.r_offset + pltTable.dl_base_addr);
    *reinterpret_cast<size_t*>(got_addr) = reinterpret_cast<size_t>(bye);
...
}
```

这里需要注意，用于替换的函数bye和原来的函数hello，需要是完全一样的函数签名，否则调用方和被调用方的对参数类型、个数没有达成一致，可能也会crash或者出现莫名其妙的错误


## 效果如下
编译：
```bash
g++ -std=c++20 main.cpp hook.cpp -o main -L lib -lhello -I lib/include
LD_LIBRARY_PATH=./lib ./main

#打开logging
LOGGING=1 LD_LIBRARY_PATH=./lib ./main
```
测试代码如下：
```c++
int main() {
    hello();
    plthook("hello", reinterpret_cast<void *>(bye)); //替换hello为bye
    hello();
    return 0;
}
```

输出如下
```
Hello, world!
byebye
```


