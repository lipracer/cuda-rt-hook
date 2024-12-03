#include <dlfcn.h>
#include <elf.h>
#include <errno.h>
#include <limits.h>
#include <link.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>


/*
这部分宏定义主要由
https://github.com/kubo/plthook/blob/master/plthook_elf.c
修改而来，主要是为了适配不同的架构和平台。
*/

#if defined __x86_64__ || defined __x86_64
#define R_JUMP_SLOT R_X86_64_JUMP_SLOT
#define R_GLOBAL_DATA R_X86_64_GLOB_DAT
#elif defined __i386__ || defined __i386
#define R_JUMP_SLOT R_386_JMP_SLOT
#define R_GLOBAL_DATA R_386_GLOB_DAT
#define USE_REL
#elif defined __arm__ || defined __arm
#define R_JUMP_SLOT R_ARM_JUMP_SLOT
#define R_GLOBAL_DATA R_ARM_GLOB_DAT
#define USE_REL
#elif defined __aarch64__ || defined __aarch64 /* ARM64 */
#define R_JUMP_SLOT R_AARCH64_JUMP_SLOT
#define R_GLOBAL_DATA R_AARCH64_GLOB_DAT
#elif defined __powerpc64__
#define R_JUMP_SLOT R_PPC64_JMP_SLOT
#define R_GLOBAL_DATA R_PPC64_GLOB_DAT
#elif defined __powerpc__
#define R_JUMP_SLOT R_PPC_JMP_SLOT
#define R_GLOBAL_DATA R_PPC_GLOB_DAT
#elif defined __riscv
#define R_JUMP_SLOT R_RISCV_JUMP_SLOT
#if __riscv_xlen == 32
#define R_GLOBAL_DATA R_RISCV_32
#elif __riscv_xlen == 64
#define R_GLOBAL_DATA R_RISCV_64
#else
#error unsupported RISCV implementation
#endif
#elif 0 /* disabled because not tested */ && \
    (defined __sparcv9 || defined __sparc_v9__)
#define R_JUMP_SLOT R_SPARC_JMP_SLOT
#elif 0 /* disabled because not tested */ && \
    (defined __sparc || defined __sparc__)
#define R_JUMP_SLOT R_SPARC_JMP_SLOT
#elif 0 /* disabled because not tested */ && \
    (defined __ia64 || defined __ia64__)
#define R_JUMP_SLOT R_IA64_IPLTMSB
#else
#error unsupported OS
#endif

#ifdef USE_REL
#define Elf_Plt_Rel Elf_Rel
#define PLT_DT_REL DT_REL
#define PLT_DT_RELSZ DT_RELSZ
#define PLT_DT_RELENT DT_RELENT
#else
#define Elf_Plt_Rel Elf_Rela
#define PLT_DT_REL DT_RELA
#define PLT_DT_RELSZ DT_RELASZ
#define PLT_DT_RELENT DT_RELAENT
#endif

#if defined __LP64__
#ifndef ELF_CLASS
#define ELF_CLASS ELFCLASS64
#endif
#define SIZE_T_FMT "lu"
#define ELF_WORD_FMT "u"
#ifdef __ANDROID__
#define ELF_XWORD_FMT "llu"
#else
#define ELF_XWORD_FMT "lu"
#endif
#define ELF_SXWORD_FMT "ld"
#define Elf_Half Elf64_Half
#define Elf_Xword Elf64_Xword
#define Elf_Sxword Elf64_Sxword
#define Elf_Ehdr Elf64_Ehdr
#define Elf_Phdr Elf64_Phdr
#define Elf_Sym Elf64_Sym
#define Elf_Dyn Elf64_Dyn
#define Elf_Rel Elf64_Rel
#define Elf_Rela Elf64_Rela
#ifndef ELF_R_SYM
#define ELF_R_SYM ELF64_R_SYM
#endif
#ifndef ELF_R_TYPE
#define ELF_R_TYPE ELF64_R_TYPE
#endif
#else /* __LP64__ */
#ifndef ELF_CLASS
#define ELF_CLASS ELFCLASS32
#endif
#define SIZE_T_FMT "u"
#ifdef __sun
#define ELF_WORD_FMT "lu"
#define ELF_XWORD_FMT "lu"
#define ELF_SXWORD_FMT "ld"
#else
#define ELF_WORD_FMT "u"
#define ELF_XWORD_FMT "u"
#define ELF_SXWORD_FMT "d"
#endif
#define Elf_Half Elf32_Half
#define Elf_Xword Elf32_Word
#define Elf_Sxword Elf32_Sword
#define Elf_Ehdr Elf32_Ehdr
#define Elf_Phdr Elf32_Phdr
#define Elf_Sym Elf32_Sym
#define Elf_Dyn Elf32_Dyn
#define Elf_Rel Elf32_Rel
#define Elf_Rela Elf32_Rela
#ifndef ELF_R_SYM
#define ELF_R_SYM ELF32_R_SYM
#endif
#ifndef ELF_R_TYPE
#define ELF_R_TYPE ELF32_R_TYPE
#endif
#endif /* __LP64__ */

//在elf.h中找不到，似乎是一个新加入的segment type, 详见：
//https://bugzilla.redhat.com/show_bug.cgi?id=1748802
#define PT_GNU_PROPERTY 0x6474E553