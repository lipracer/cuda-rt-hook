#include "hook.h"

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

#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <list>
#include <atomic>
#include <fstream>

#include "cuda_types.h"
#include "internal.h"

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
    char* base_addr = nullptr;
    ElfW(Sym) * dynsym = nullptr;
    ElfW(Rel) * rela_dyn = nullptr;
    ElfW(Rela) * rela_plt = nullptr;
    size_t rela_plt_cnt = 0;
    char* symbol_table = nullptr;
    // Elf_Plt_Rel
    operator bool() const { return !!base_addr && !!rela_plt_cnt; }
    bool operator!() const { return (bool)(*this); }
};

static size_t page_size;
#define ALIGN_ADDR(addr) ((void*)((size_t)(addr) & ~(page_size - 1)))

static int get_memory_permission(void* address) {
    unsigned long addr = (unsigned long)address;
    FILE* fp;
    char buf[PATH_MAX];
    char perms[5];
    int bol = 1;

    fp = fopen("/proc/self/maps", "r");
    if (fp == NULL) {
        return 0;
    }
    while (fgets(buf, PATH_MAX, fp) != NULL) {
        unsigned long start, end;
        int eol = (strchr(buf, '\n') != NULL);
        if (bol) {
            /* The fgets reads from the beginning of a line. */
            if (!eol) {
                /* The next fgets reads from the middle of the same line. */
                bol = 0;
            }
        } else {
            /* The fgets reads from the middle of a line. */
            if (eol) {
                /* The next fgets reads from the beginning of a line. */
                bol = 1;
            }
            continue;
        }

        if (sscanf(buf, "%lx-%lx %4s", &start, &end, perms) != 3) {
            continue;
        }
        if (start <= addr && addr < end) {
            int prot = 0;
            if (perms[0] == 'r') {
                prot |= PROT_READ;
            } else if (perms[0] != '-') {
                goto unknown_perms;
            }
            if (perms[1] == 'w') {
                prot |= PROT_WRITE;
            } else if (perms[1] != '-') {
                goto unknown_perms;
            }
            if (perms[2] == 'x') {
                prot |= PROT_EXEC;
            } else if (perms[2] != '-') {
                goto unknown_perms;
            }
            if (perms[3] != 'p') {
                goto unknown_perms;
            }
            if (perms[4] != '\0') {
                perms[4] = '\0';
                goto unknown_perms;
            }
            fclose(fp);
            return prot;
        }
    }
    fclose(fp);
    return 0;
unknown_perms:
    fclose(fp);
    return 0;
}

struct HookTarget {
    std::string lib_name;
    std::string func_name;
    void* new_func = nullptr;
    void* old_func = nullptr;
    std::function<bool(const std::string& name)> fileter;
};

class CudaInfoCollection {
   public:
    static CudaInfoCollection& instance() {
        static CudaInfoCollection self;
        return self;
    }
    void collectRtLib(std::string lib) {
        if (libcudart_.empty() &&
            lib.find(cudaRTLibPath) != std::string::npos) {
            libcudart_ = lib;
        }
    }

    void* getSymbolAddr(const std::string& name) {
        CHECK(!libcudart_.empty(), "libcudart empty!");
        auto handle = dlopen(libcudart_.c_str(), RTLD_LAZY);
        CHECK(handle, std::string("can't open ") + libcudart_);
        return dlsym(handle, name.c_str());
    }

   private:
    const std::string cudaRTLibPath = "libcudart.so";
    std::string libcudart_;
    std::atomic<void*> handle_{nullptr};
};

int install_hooker(PltTable* pltTable, HookTarget& hookTarget) {
    CudaInfoCollection::instance().collectRtLib(pltTable->lib_name);
    if (hookTarget.fileter && hookTarget.fileter(pltTable->lib_name)) {
        return 0;
    }
    LOG(0) << "install lib name:" << pltTable->lib_name;
    size_t index = 0;
    while (index < pltTable->rela_plt_cnt) {
        auto plt = pltTable->rela_plt + index++;
        if (ELF64_R_TYPE(plt->r_info) != R_JUMP_SLOT) {
            continue;
        }

        size_t idx = ELF64_R_SYM(plt->r_info);
        idx = pltTable->dynsym[idx].st_name;
        // printf("enum func name:%s\n", pltTable->symbol_table + idx);
        if (hookTarget.func_name.find(pltTable->symbol_table + idx) ==
            std::string::npos) {
            continue;
        }
        void* addr =
            reinterpret_cast<void*>(pltTable->base_addr + plt->r_offset);
        int prot = get_memory_permission(addr);
        if (prot == 0) {
            return -1;
        }
        if (!(prot & PROT_WRITE)) {
            if (mprotect(ALIGN_ADDR(addr), page_size, PROT_READ | PROT_WRITE) !=
                0) {
                return -1;
            }
        }
        hookTarget.old_func = addr;
        *reinterpret_cast<size_t*>(addr) =
            reinterpret_cast<size_t>(hookTarget.new_func);
        if (!(prot & PROT_WRITE)) {
            mprotect(ALIGN_ADDR(addr), page_size, prot);
        }
        LOG(0) << "replace:" << pltTable->symbol_table + idx << " with "
               << hookTarget.func_name << " success";
        return 0;
    }
    LOG(0) << "can't find symbol:" << hookTarget.func_name;
    return 0;
}

int retrieve_dyn_lib(struct dl_phdr_info* info, size_t info_size, void* table) {
    using VecTable = std::vector<PltTable>;
    auto* vecPltTable = reinterpret_cast<VecTable*>(table);
    PltTable pltTable;
    pltTable.lib_name = info->dlpi_name ? info->dlpi_name : "";
    pltTable.base_addr = (char*)info->dlpi_phdr - info_size;
    ElfW(Dyn*) dyn;
    for (size_t header_index = 0; header_index < info->dlpi_phnum;
         header_index++) {
        if (info->dlpi_phdr[header_index].p_type == PT_DYNAMIC) {
            dyn = (ElfW(Dyn)*)(info->dlpi_addr +
                               info->dlpi_phdr[header_index].p_vaddr);
            while (dyn->d_tag != DT_NULL) {
                switch (dyn->d_tag) {
                    case DT_STRTAB: {
                        pltTable.symbol_table =
                            reinterpret_cast<char*>(dyn->d_un.d_ptr);
                    } break;
                    case DT_STRSZ: {
                    } break;
                    case DT_SYMTAB: {
                        pltTable.dynsym =
                            reinterpret_cast<ElfW(Sym)*>(dyn->d_un.d_ptr);
                    } break;
                    case DT_JMPREL: {
                        pltTable.rela_plt =
                            reinterpret_cast<ElfW(Rela)*>(dyn->d_un.d_ptr);
                    } break;
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

namespace cuda_mock {

std::unordered_map<void*, void*> function_map;

const char* kPytorchCudaLibName = "libtorch_cuda.so";

#include <execinfo.h>

class BackTraceCollection {
   public:
    using BackTrace = std::vector<std::string>;
    static BackTraceCollection& instance() {
        static BackTraceCollection self;
        return self;
    }

    void collect_backtrace(const void* func_ptr) {
        auto iter = cached_map_.find(func_ptr);

        if (iter != cached_map_.end()) {
            ++std::get<1>(backtraces_[iter->second]);
            return;
        }
        cached_map_.insert(std::make_pair(func_ptr, backtraces_.size()));
        void* buffer[32];
        char** symbols;
        int num = backtrace(buffer, 32);
        symbols = backtrace_symbols(buffer, num);
        if (symbols == NULL) {
            LOG(2) << "can't get backtrace symbol!";
        }
        backtraces_.emplace_back();
        for (int j = 0; j < num; j++) {
            std::get<0>(backtraces_.back()).push_back(symbols[j]);
        }
        free(symbols);
    }

    void dump() {
        std::ofstream ofs("./backtrace.log");
        for(const auto& backtrace : backtraces_) {
            ofs << "call this cuda func " << std::get<1>(backtrace) << "\n";
            for(const auto& line : std::get<0>(backtrace)) {
                ofs << line << "\n";
            }
        }
        ofs.flush();
    }

    ~BackTraceCollection() { dump(); }

   private:
    std::vector<std::tuple<BackTrace, size_t>> backtraces_;
    std::unordered_map<const void*, size_t> cached_map_;
};

extern "C" CUresult cudaLaunchKernel_wrapper(const void* func, dim3 gridDim,
                                             dim3 blockDim, void** args,
                                             size_t sharedMem,
                                             cudaStream_t stream) {
    BackTraceCollection::instance().collect_backtrace(func);
    auto func_name = reinterpret_cast<const char*>(func);
    // LOG(0) << __func__ << ":" << std::string(func_name, 16);
    static void* org_addr =
        CudaInfoCollection::instance().getSymbolAddr("cudaLaunchKernel");
    // org_addr =
    // function_map[reinterpret_cast<void*>(&cudaLaunchKernel_wrapper)];
    CHECK(org_addr, "empty cudaLaunchKernel addr!");
    return reinterpret_cast<decltype(&cudaLaunchKernel_wrapper)>(org_addr)(
        func, gridDim, blockDim, args, sharedMem, stream);
}

void* my_molloc(size_t size) {
    LOG(0) << "my_malloc";
    return nullptr;
}

// std::unordered_map<std::string, std::vector<void*>>

void* dlopen_wrapper(const char* pathname, int mode) {
    return dlopen(pathname, mode);
}

void* dlsym_wrapper(void* handle, const char* symbol) {
    auto ret = dlsym(handle, symbol);
    if (std::string(symbol).find("cudaLaunchKernel") != std::string::npos) {
        LOG(0) << "replace cudaLaunchKernel!";
        auto new_func = reinterpret_cast<void*>(&cudaLaunchKernel_wrapper);
        function_map[new_func] = ret;
        return new_func;
    }
    return ret;
}

void install_hook() {
    page_size = sysconf(_SC_PAGESIZE);

    std::vector<PltTable> vecPltTable;
    dl_iterate_phdr(retrieve_dyn_lib, &vecPltTable);
    LOG(0) << "collect plt table size:" << vecPltTable.size();
    {
        HookTarget hookTarget = HookTarget{
            .lib_name = kPytorchCudaLibName,
            .func_name = "cudaLaunchKernel",
            .new_func = reinterpret_cast<void*>(&cudaLaunchKernel_wrapper),
            .old_func = nullptr};

        for (auto& pltTable : vecPltTable) {
            install_hooker(&pltTable, hookTarget);
        }
        function_map[hookTarget.new_func] = hookTarget.old_func;
    }

    // auto libFilter = [](const std::string& name) -> bool {
    //     if (name.empty()) {
    //         return true;
    //     }
    //     auto pos = name.find_last_of("/");
    //     if (pos != std::string::npos) {
    //         auto libName = name.substr(pos);
    //         if (libName.find("torch") == std::string::npos) {
    //             return true;
    //         }
    //     }
    //     return false;
    // };

    // {
    //     HookTarget hookTarget =
    //         HookTarget{.lib_name = "torch",
    //                    .func_name = "dlopen",
    //                    .new_func = reinterpret_cast<void*>(&dlopen_wrapper),
    //                    .old_func = nullptr,
    //                    .fileter = libFilter};

    //     for (auto& pltTable : vecPltTable) {
    //         install_hooker(&pltTable, hookTarget);
    //     }
    // }

    // {
    //     HookTarget hookTarget =
    //         HookTarget{.lib_name = "torch",
    //                    .func_name = "dlsym",
    //                    .new_func = reinterpret_cast<void*>(&dlsym_wrapper),
    //                    .old_func = nullptr,
    //                    .fileter = libFilter};

    //     for (auto& pltTable : vecPltTable) {
    //         install_hooker(&pltTable, hookTarget);
    //     }
    // }

    // CHECK(hookTarget.old_func, "install_hooker func error!");
}
}  // namespace cuda_mock