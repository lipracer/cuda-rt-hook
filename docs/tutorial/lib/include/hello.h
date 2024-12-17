#ifndef LIBHELLO_H
#define LIBHELLO_H

// 声明全局变量，使用extern关键字
extern int time;

extern "C"{
    void hello();
}
// 声明hello函数

#endif // LIBHELLO_H