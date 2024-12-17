
#include <atomic>
#include <fstream>
#include <functional>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <format>  // std::format 所在的头文件
#include <iostream> // 用于 std::cout

#include "hook.h"



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


std::string segmentType2String(ElfW(Word) p_type){
    std::string type = (p_type == PT_LOAD) ? "PT_LOAD" :
                      (p_type == PT_DYNAMIC) ? "PT_DYNAMIC" :
                      (p_type == PT_INTERP) ? "PT_INTERP" :
                      (p_type == PT_NOTE) ? "PT_NOTE" :
                      (p_type == PT_INTERP) ? "PT_INTERP" :
                      (p_type == PT_PHDR) ? "PT_PHDR" :
                      (p_type == PT_TLS) ? "PT_TLS" :
                      (p_type == PT_GNU_EH_FRAME) ? "PT_GNU_EH_FRAME" :
                      (p_type == PT_GNU_STACK) ? "PT_GNU_STACK" :
                      (p_type == PT_GNU_PROPERTY) ? "PT_GNU_PROPERTY" :
                      (p_type == PT_GNU_RELRO) ? "PT_GNU_RELRO" : std::format("unknown:{:#x}", p_type);
    return type;
}


/**
 * vecPltTables : &std::vector<PltTable> 由于dl_iterate_phdr接受的参数只能是void *
*/
int retrieve_dyn_lib(struct dl_phdr_info* info, size_t info_size, void* vecPltTables) {
    std::vector<PltTable> &plts = *reinterpret_cast<std::vector<PltTable>*>(vecPltTables);
    std::cout << "\n=============   retrieve_dyn_lib   ===========" << std::endl;
    PltTable pltTable;
    pltTable.lib_name = info->dlpi_name ? info->dlpi_name : "unknown";
    pltTable.dl_base_addr = const_cast<char*>(reinterpret_cast<const char*>(info->dlpi_addr));
    std::cout << std::format("lib name: {}, dlpi_addr: {:#x}, dlpi_phdr(header addr): {:#x}, info_size: {}, dlpi_phnum(header num): {}", info->dlpi_name, reinterpret_cast<uintptr_t>(info->dlpi_addr), reinterpret_cast<uintptr_t>(info->dlpi_phdr), info_size, info->dlpi_phnum) << std::endl;
    /*
        遍历info->dlpi_phdr数组中的program header, dlpi_phdr是一个ElfW(Phdr)类型的数组。
        info->dlpi_phnum是一个ElfW(Half)类型的变量，表示dlpi_phdr数组的长度，也就是header的个数
    */
    for (size_t header_index = 0; header_index < info->dlpi_phnum;
         header_index++) {

        auto segment_header = info->dlpi_phdr[header_index];
        // 根据文档，这里只打印关键的几个属性，其他属性并不重要：https://man7.org/linux/man-pages/man3/dl_iterate_phdr.3.html#COLOPHON
        std::cout << std::format("[Seg{:>2}/{}, TYPE={:>15}][addr:{:>#15x}, size:{:>10}] p_flags:{}",
        // p_offset:{:#x}, p_paddr:{:#x}, p_filesz:{:#x}, p_align:{:#x}, ", 
        header_index,
        info->dlpi_phnum,
        segmentType2String(segment_header.p_type),
        reinterpret_cast<uintptr_t>(info->dlpi_addr + segment_header.p_vaddr),
        segment_header.p_memsz,
        segment_header.p_flags
        // segment_header.p_offset, 
        // reinterpret_cast<uintptr_t>(segment_header.p_paddr),
        // segment_header.p_filesz,
        // segment_header.p_align
        ) << std::endl;
        
        /*
        如果一个elf文件参与动态链接，那么program header中会出现类型为PT_DYNAMIC的header
        只有这个段是我们所关心的，其他不关心。
        */
        if (info->dlpi_phdr[header_index].p_type == PT_DYNAMIC) {
            /*
                info->dlpi_addr: 共享对象的虚拟内存起始地址，相对于进程的地址空间。对于可执行文件，这通常是0。
                info->dlpi_phdr[header_index].p_vaddr 第header_index个segment的虚拟起始地址，相对于info->dlpi_addr
                那么二者相加，就是segment的虚拟地址，也就是segment在进程中的实际地址

                ElfW(dyn)宏扩展为，定义在/usr/include/elf.h中
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
            ElfW(Dyn)* dynamic_segment_item = (ElfW(Dyn)*)(info->dlpi_addr + segment_header.p_vaddr); // 在64位系统下是Elf64_Dyn ，定义在/usr/include/elf.h

            /*
                dynamic段是一个ElfW(dynamic_segment_item)类型的数组，以DT_NULL结束，类似用null代表字符串结尾。
                DT_*的详细含义见：https://docs.oracle.com/cd/E19683-01/816-7529/chapter6-42444/index.html，这里摘抄《程序员自我修养》中的表格，记录了常用的类型
                d_tag 类型            d_un 的含义
                DT_SYMTAB          动态链接符号表的地址，d_ptr 表示 “.symtable” 的地址
                DT_STRTAB          动态链接字符串表地址，d_ptr 表示 “.dynstr” 的地址
                DT_STRSZ           动态链接字符串表大小，d_val 表示大小
                DT_HASH            动态链接哈希表地址，d_ptr 表示 “.hash” 的地址
                DT_SONAME          本共享对象的 “SO-NAME”，我们在后面会介绍 “SO-NAME”
                DT_RPATH           动态链接共享对象搜索路径
                DT_INIT            初始化代码地址
                DT_FINIT           结束代码地址
                DT_NEED            依赖的共享对象文件，d_ptr 表示所依赖的共享对象文件名
                DT_REL             动态链接重定位表地址
                DT_RELA            动态链接重定位表地址（同上）
                DT_RELENT          动态重定位表入口数量（同上）
                DT_RELAENT         动态重定位表入口数量
            */
            while (dynamic_segment_item->d_tag != DT_NULL) {
                switch (dynamic_segment_item->d_tag) {
                    case DT_STRTAB: {
                        pltTable.str_table =
                            reinterpret_cast<char*>(dynamic_segment_item->d_un.d_ptr);
                            
                    } break;
                    case DT_STRSZ: {
                        std::cout << std::format("DT_STRSZ: {:#x}", reinterpret_cast<uintptr_t>(dynamic_segment_item->d_un.d_val)) << std::endl;
                    } break;
                    /*
                        符号表，实际存的不是字符串，而是DT_STRTAB的offset，参考install_hooker。
                        这也是做target_symbol匹配时需要的
                    */
                    case DT_SYMTAB: {
                        pltTable.sym_table =
                            reinterpret_cast<ElfW(Sym)*>(dynamic_segment_item->d_un.d_ptr);
                        std::cout << std::format("DT_JMPREL: {:#x}", reinterpret_cast<uintptr_t>(pltTable.rela_plt)) << std::endl;
                    } break;
                    /*
                        代码段的plt表的起始地址
                    */
                    case DT_JMPREL: {
                        pltTable.rela_plt =
                            reinterpret_cast<ElfW(Rela)*>(dynamic_segment_item->d_un.d_ptr);
                        std::cout << std::format("DT_JMPREL: {:#x}", reinterpret_cast<uintptr_t>(pltTable.rela_plt)) << std::endl;
                    } break;
                    /*
                        代码段的plt表的长度，以bytes为单位
                    */
                    case DT_PLTRELSZ: {
                        pltTable.rela_plt_cnt =
                            dynamic_segment_item->d_un.d_val / sizeof(Elf_Plt_Rel);
                        std::cout << std::format("DT_PLTRELSZ: {:#x}", dynamic_segment_item->d_un.d_val / sizeof(Elf_Plt_Rel) ) << std::endl;

                    } break;

                    /*
                        以下分别是数据段的plt表及长度，这里不关心
                    */
                    case PLT_DT_REL: {
                        std::cout << std::format("PLT_DT_REL: {:#x}", reinterpret_cast<uintptr_t>(dynamic_segment_item->d_un.d_ptr)) << std::endl;
                    } break;
                    case PLT_DT_RELSZ: {
                        std::cout << std::format("PLT_DT_RELSZ: {:#x}", dynamic_segment_item->d_un.d_val / sizeof(Elf_Plt_Rel) ) << std::endl;
                    } break;
                }
                dynamic_segment_item++;
            }
        }
    }
    plts.emplace_back(pltTable);
    std::cout << std::format("retrieve_dyn_lib finish [lib={}]", plts.size()) << std::endl;
    return 0;
}

std::vector<PltTable> retrieve_plts_of_dynlibs(){
    page_size = sysconf(_SC_PAGESIZE);
    std::vector<PltTable> pltTables;
    dl_iterate_phdr(retrieve_dyn_lib, (void *)&pltTables);
    std::cout << std::format("retrieve_plts_of_dynlibs: get plts of {} libs", pltTables.size()) << std::endl;
    return pltTables;
}

void hook_plt(const std::vector<PltTable> &pltTables, const std::string & target_symbol, void * new_func){
    for (auto pltTable : pltTables) {
        for(size_t plt_idx = 0; plt_idx < pltTable.rela_plt_cnt; plt_idx++){
            ElfW(Rela) plt_entry = pltTable.rela_plt[plt_idx];

            // 解析符号名
            size_t sym_idx = ELF_R_SYM(plt_entry.r_info);
            size_t offset_in_strtable = pltTable.sym_table[sym_idx].st_name;
            const char * symbol_name = pltTable.str_table + offset_in_strtable;
            std::cout << std::format("[lib={}] symbol_name: {}, target_symbol: {}", pltTable.lib_name, symbol_name, target_symbol) << std::endl;

            if(target_symbol != std::string(symbol_name)){
                
                continue;

            }
            std::cout << std::format("[lib={}] symbol_name: {} match target_symbol: {}", pltTable.lib_name, symbol_name, target_symbol) << std::endl;

            void * got_addr = reinterpret_cast<void*>(plt_entry.r_offset + pltTable.dl_base_addr);
            size_t origin_addr = *reinterpret_cast<size_t*>(got_addr);
            //TODO memory permission
            //如果got_addr内存页无写权限，直接写got_addr会segfault，所以需要更改为写权限。
            int prot = get_memory_permission(got_addr);
            if (prot == 0) {
                std::cout << std::format("memory is not writable") << std::endl;

                return ;
            }
            std::cout << std::format("[lib={}] prot = {:#x}, PROT_WRITE = {:#x}", pltTable.lib_name, prot, PROT_WRITE) << std::endl;

            if (!(prot & PROT_WRITE)) { //not writable and cannot convert to writable page
                if (mprotect(ALIGN_ADDR(got_addr), page_size, PROT_READ | PROT_WRITE) !=
                    0) {
                    return ;
                }
            }

            //TODO用size_t表示地址是否合理
            std::cout << std::format("[lib={}] start overwrite {:#x}", pltTable.lib_name, reinterpret_cast<size_t>(got_addr)) << std::endl;
            *reinterpret_cast<size_t*>(got_addr) = reinterpret_cast<size_t>(new_func);
            std::cout << std::format("[lib={}] overwrite {:#x} with {:#x}", pltTable.lib_name, origin_addr, reinterpret_cast<size_t>(new_func)) << std::endl;
            return;
        }

    }
}

void plthook(const std::string & target_symbol, void* new_func){
    //临时禁用std::cout
    const char* loggingEnv = std::getenv("LOGGING");
    bool enableLogging = loggingEnv != nullptr && std::string(loggingEnv) == "1";
    std::streambuf* originalBuffer = std::cout.rdbuf();
    std::ofstream nullStream("/dev/null");
    if(!enableLogging){
        std::cout.rdbuf(nullStream.rdbuf());
    }

    auto pltTables = retrieve_plts_of_dynlibs();
    hook_plt(pltTables, target_symbol, new_func);

    //恢复std::cout
    if(!enableLogging){
        std::cout.rdbuf(originalBuffer);
    }
}