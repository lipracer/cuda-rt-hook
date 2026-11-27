#include "hook.h"

#include <dlfcn.h>
#include <elf.h>
#include <errno.h>
#include <limits.h>
#include <link.h>
#include <regex.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <fstream>
#include <functional>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// #include "backward.hpp"

#include "cuda_types.h"
#include "env_mgr.h"
#include "env_util.h"
#include "logger/logger.h"

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

struct PltTable {
    std::string lib_name;
    const char* base_addr = nullptr;
    char* base_header_addr = nullptr;
    ElfW(Sym) * dynsym = nullptr;
    ElfW(Rel) * rela_dyn = nullptr;
    ElfW(Rela) * rela_plt = nullptr;
    size_t rela_plt_cnt = 0;
    char* symbol_table = nullptr;
    // Elf_Plt_Rel
    operator bool() const { return !!base_header_addr && !!rela_plt_cnt; }
    bool operator!() const { return (bool)(*this); }
};

struct MemoryMapping {
    uintptr_t start;
    uintptr_t end;
    int protection;
};

static std::vector<MemoryMapping> read_memory_mappings() {
    std::ifstream maps("/proc/self/maps");
    std::vector<MemoryMapping> mappings;
    std::string line;
    while (std::getline(maps, line)) {
        unsigned long start, end;
        char perms[5];
        if (sscanf(line.c_str(), "%lx-%lx %4s", &start, &end, perms) != 3 ||
            strlen(perms) != 4 || perms[3] != 'p') {
            continue;
        }
        int protection = (perms[0] == 'r' ? PROT_READ : 0) |
                         (perms[1] == 'w' ? PROT_WRITE : 0) |
                         (perms[2] == 'x' ? PROT_EXEC : 0);
        mappings.push_back({start, end, protection});
    }
    return mappings;
}

struct GotPermissions {
    bool make_writable(uintptr_t address) {
        if (!mappings_loaded) {
            mappings = read_memory_mappings();
            mappings_loaded = true;
        }
        const uintptr_t page = address - address % page_size;
        if (writable_pages.count(page)) {
            return true;
        }
        auto it = std::upper_bound(mappings.begin(), mappings.end(), address,
                                   [](uintptr_t addr, const MemoryMapping& m) {
                                       return addr < m.start;
                                   });
        if (it == mappings.begin()) {
            return false;
        }
        const auto& mapping = *std::prev(it);
        if (address >= mapping.end || mapping.protection == PROT_NONE) {
            return false;
        }
        if (!(mapping.protection & PROT_WRITE) &&
            mprotect(reinterpret_cast<void*>(page), page_size,
                     mapping.protection | PROT_WRITE) != 0) {
            return false;
        }
        // Keep the page writable for the existing uninstall path.
        writable_pages.insert(page);
        return true;
    }

    const size_t page_size = sysconf(_SC_PAGESIZE);
    bool mappings_loaded = false;
    std::vector<MemoryMapping> mappings;
    std::unordered_set<uintptr_t> writable_pages;
};

static bool isTargetLibFromEnv(const char* libName) {
    static const char* pattern =
        hook::get_env_value<const char*>(env_mgr::TARGET_LIB_FILTER);
    if (!pattern) {
        return true;
    }
    static regex_t regex;
    static int comp_result = regcomp(&regex, pattern, 0);
    if (comp_result) {
        LOG(WARN) << "ilegal regex pattern:" << pattern;
        return false;
    }
    int reti = regexec(&regex, libName, 0, nullptr, 0);
    if (reti == REG_NOMATCH) {
        return false;
    }
    return reti == 0;
}

static void install_hooker(const PltTable& table,
                           const std::vector<hook::HookInstaller>& installers,
                           GotPermissions& permissions) {
    std::vector<const hook::HookInstaller*> targets;
    for (const auto& installer : installers) {
        if (installer.isTargetLib(table.lib_name.c_str())) {
            targets.push_back(&installer);
        }
    }
    if (targets.empty()) {
        return;
    }

    for (size_t index = 0; index < table.rela_plt_cnt; ++index) {
        const auto& relocation = table.rela_plt[index];
        if (ELF64_R_TYPE(relocation.r_info) != R_JUMP_SLOT) {
            continue;
        }

        const auto& symbol = table.dynsym[ELF64_R_SYM(relocation.r_info)];
        const char* name = table.symbol_table + symbol.st_name;
        for (const auto* installer : targets) {
            if (!installer->isTargetSymbol(name)) {
                continue;
            }
            auto** slot = reinterpret_cast<void**>(table.base_header_addr +
                                                   relocation.r_offset);
            if (!permissions.make_writable(reinterpret_cast<uintptr_t>(slot))) {
                MLOG(HOOK, ERROR) << "Unable to make GOT slot writable for "
                                  << name << " in " << table.lib_name;
                return;
            }

            hook::OriginalInfo original;
            original.libName = table.lib_name.c_str();
            original.basePtr = table.base_addr;
            original.relaPtr = table.rela_plt;
            original.baseHeadPtr = table.base_header_addr;
            original.pltTablePtr = slot;
            original.oldFuncPtr = *slot;
            *slot = installer->newFuncPtr(original);
            MLOG(HOOK, INFO) << "replaced GOT[" << index << "] " << name
                             << " in " << table.lib_name;
            if (installer->onSuccess) {
                installer->onSuccess();
            }
        }
    }
}

int retrieve_dyn_lib(struct dl_phdr_info* info, size_t info_size, void* table) {
    using VecTable = std::vector<PltTable>;
    auto* vecPltTable = reinterpret_cast<VecTable*>(table);
    const char* name = info->dlpi_name ? info->dlpi_name : "";
    if (!isTargetLibFromEnv(name)) {
        return 0;
    }

    PltTable pltTable;
    pltTable.lib_name = name;
    pltTable.base_header_addr = (char*)info->dlpi_phdr - info_size;
    pltTable.base_addr = reinterpret_cast<const char*>(info->dlpi_addr);
    // pltTable.base_addr = pltTable.base_header_addr;
    ElfW(Dyn*) dyn;
    MLOG(HOOK, INFO) << "install lib name:" << pltTable.lib_name
                     << " dlpi_addr:" << std::hex
                     << reinterpret_cast<void*>(info->dlpi_addr)
                     << " dlpi_phdr:" << std::hex
                     << reinterpret_cast<const void*>(info->dlpi_phdr)
                     << " info_size:" << info_size;
    /*
        遍历info->dlpi_phdr数组中的program header,
       dlpi_phdr是一个ElfW(Phdr)类型的数组。
        info->dlpi_phnum是一个ElfW(Half)类型的变量，表示dlpi_phdr数组的长度，也就是header的个数

        ElfW(Phdr)是segment
       header，表征了segment的各个属性，它是一个宏，在64位系统下是Elf64_Phdr，定义在/usr/include/elf.h中，其定义如下：
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
    */
    for (size_t header_index = 0; header_index < info->dlpi_phnum;
         header_index++) {
        /*
        如果一个elf文件参与动态链接，那么program
        header中会出现类型为PT_DYNAMIC的header
        只有这个段是我们所关心的，其他不关心。
        */
        if (info->dlpi_phdr[header_index].p_type == PT_DYNAMIC) {
            /*
                info->dlpi_addr:
               共享对象的虚拟内存起始地址，相对于进程的地址空间。对于可执行文件，这通常是0。
                info->dlpi_phdr[header_index].p_vaddr
               第header_index个segment的虚拟起始地址，相对于info->dlpi_addr
                那么二者相加，就是segment的虚拟地址，也就是segment在进程中的实际地址

                ElfW(Dyn)宏扩展为，定义在/usr/include/elf.h中
                typedef struct
                {
                Elf64_Sxword	d_tag;			// Dynamic entry type
                union
                    {
                    Elf64_Xword d_val;		// Integer value
                    Elf64_Addr d_ptr;			// Address value
                    } d_un;
                } Elf64_Dyn;
                根据d_tag不同，d_un可以表示整数或者地址
            */
            dyn = (ElfW(Dyn)*)(info->dlpi_addr +
                               info->dlpi_phdr[header_index].p_vaddr);
            /*
                dynamic段是一个ElfW(Dyn)类型的数组，以DT_NULL结束，类似用null代表字符串结尾。
                DT_*的详细含义见：https://docs.oracle.com/cd/E19683-01/816-7529/chapter6-42444/index.html
            */
            while (dyn->d_tag != DT_NULL) {
                switch (dyn->d_tag) {
                    /*
                        字符串表，elf文件使用了一个单独的区域存储所有的字符串。引用字符串时，需要给出字符串在字符串表中的偏移。
                        从该偏移开始到null，就是所需要的字符串。
                        这是做target_symbol匹配时需要的
                        Q：这是dynamic段独占的还是elf共享的？
                    */
                    case DT_STRTAB: {
                        pltTable.symbol_table =
                            reinterpret_cast<char*>(dyn->d_un.d_ptr);
                    } break;
                    case DT_STRSZ: {
                    } break;
                    /*
                        符号表，实际存的不是字符串，而是DT_STRTAB的offset，参考install_hooker。
                        这也是做target_symbol匹配时需要的
                    */
                    case DT_SYMTAB: {
                        pltTable.dynsym =
                            reinterpret_cast<ElfW(Sym)*>(dyn->d_un.d_ptr);
                    } break;
                    /*
                        plt表的起始地址，什么是plt表：https://docs.oracle.com/cd/E19683-01/816-7529/6mdhf5r3k/index.html#chapter6-1235
                        这是我们需要patch的。
                    */
                    case DT_JMPREL: {
                        pltTable.rela_plt =
                            reinterpret_cast<ElfW(Rela)*>(dyn->d_un.d_ptr);
                    } break;
                    /*
                        plt表的长度
                    */
                    case DT_PLTRELSZ: {
                        pltTable.rela_plt_cnt =
                            dyn->d_un.d_val / sizeof(Elf_Plt_Rel);
                    } break;
                    case PLT_DT_REL: {
                        pltTable.rela_dyn =
                            reinterpret_cast<ElfW(Rel)*>(dyn->d_un.d_ptr);
                    } break;
                    case PLT_DT_RELSZ: {
                        // pltTable->rela_plt_cnt = dyn->d_un.d_val /
                        // sizeof(Elf_Plt_Rel);
                    } break;
                }
                dyn++;
            }
        }
    }
    if (pltTable) {
        vecPltTable->emplace_back(pltTable);
    }
    return 0;
}

namespace hook {

std::unordered_map<void*, void*> function_map;

// std::unordered_map<std::string, std::vector<void*>>

std::ostream& operator<<(std::ostream& os, const OriginalInfo& info) {
    os << "OriginalInfo:";
    os << "\nlibName:" << info.libName << "\nbasePtr:" << info.basePtr
       << "\nrelaPtr:" << info.relaPtr << "\noldFuncPtr:" << info.oldFuncPtr;
    return os;
}

void* dlopen_wrapper(const char* pathname, int mode) {
    return dlopen(pathname, mode);
}

// void* dlsym_wrapper(void* handle, const char* symbol) {
//     auto ret = dlsym(handle, symbol);
//     if (std::string(symbol).find("cudaLaunchKernel") != std::string::npos) {
//         LOG(DEBUG) << "replace cudaLaunchKernel!";
//         auto new_func = reinterpret_cast<void*>(&cudaLaunchKernel_wrapper);
//         function_map[new_func] = ret;
//         return new_func;
//     }
//     return ret;
// }

void install_hook(const HookInstaller& installer) {
    install_hooks({installer});
}

void install_hooks(const std::vector<HookInstaller>& installers) {
    if (installers.empty()) {
        return;
    }
    for (const auto& installer : installers) {
        CHECK(installer.isTargetLib, "isTargetLib can't be empty!");
        CHECK(installer.isTargetSymbol, "isTargetSymbol can't be empty!");
        CHECK(installer.newFuncPtr, "new_func_ptr can't be empty!");
    }

    std::vector<PltTable> tables;
    dl_iterate_phdr(retrieve_dyn_lib, &tables);
    MLOG(HOOK, INFO) << "collect plt table size:" << tables.size();

    // Installer callbacks may run Python code. Keep them outside loader
    // iteration.
    GotPermissions permissions;
    for (const auto& table : tables) {
        install_hooker(table, installers, permissions);
    }
}

void uninstall_hook(const OriginalInfo& info) {
    *info.pltTablePtr = info.oldFuncPtr;
}

}  // namespace hook