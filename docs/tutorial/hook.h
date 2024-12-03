#include <vector>
#include <string>
#include "hook_marcos.h"

struct PltTable {
    std::string lib_name;         // elf的名字，如果是main，则为空字符串
    char* dl_base_addr = nullptr; // elf在虚拟地址空间的起始地址
    ElfW(Rela) * rela_plt = nullptr; // elf的PLT表
    size_t rela_plt_cnt = 0;         // elf的PLT表长度（表项个数）
    ElfW(Sym) * sym_table = nullptr; // elf的符号表，需要搭配字符串表一起使用
    char* str_table = nullptr;       // elf字符串表
};

int retrieve_dyn_lib(struct dl_phdr_info* info, size_t info_size, void* vecPltTables);


// typedef struct
// {
//   Elf64_Word	p_type;			/* Segment type */
//   Elf64_Word	p_flags;		/* Segment flags */
//   Elf64_Off	p_offset;		/* Segment file offset */
//   Elf64_Addr	p_vaddr;		/* Segment virtual address */
//   Elf64_Addr	p_paddr;		/* Segment physical address */
//   Elf64_Xword	p_filesz;		/* Segment size in file */
//   Elf64_Xword	p_memsz;		/* Segment size in memory */
//   Elf64_Xword	p_align;		/* Segment alignment */
// } Elf64_Phdr;
std::string segmentType2String(ElfW(Word) p_type);


void plthook(const std::string & target_symbol, void *new_func);