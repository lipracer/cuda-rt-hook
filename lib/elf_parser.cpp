#include "elf_parser.h"

#include "util.h"

namespace hook {

CachedSymbolTable::CStrIterator::CStrIterator(const char *str) : str_(str) {}

CachedSymbolTable::CStrIterator &CachedSymbolTable::CStrIterator::operator++()
    & {
    size_t len = strlen(str_);
    ++len;
    str_ += len;
    return *this;
}

CachedSymbolTable::CStrIterator CachedSymbolTable::CStrIterator::operator++(
    int) & {
    auto ret = CStrIterator(str_);
    ++*this;
    return ret;
}

adt::StringRef CachedSymbolTable::CStrIterator::operator*() {
    return adt::StringRef(str_);
}

bool CachedSymbolTable::CStrIterator::operator==(
    const CStrIterator &other) const {
    return str_ == other.str_;
}
bool CachedSymbolTable::CStrIterator::operator!=(
    const CStrIterator &other) const {
    return !(*this == other);
}

const void *CachedSymbolTable::CStrIterator::data() const { return str_; }

CachedSymbolTable::CachedSymbolTable(
    const std::string &name, const void *base_address,
    const std::vector<std::string> &section_names)
    : libName(name),
      ifs(name),
      base_address(base_address),
      section_names(section_names) {
    CHECK(ifs.is_open(), "can't open file:{}", name);
    MLOG(TRACE, INFO) << name << " base address:" << base_address;
    ifs.read(reinterpret_cast<char *>(&elf_header), sizeof(elf_header));
    parse_named_section();
    parse_section_header();
    load_symbol_table();
    for (size_t i = 0; i < sections_.size(); ++i) {
        MLOG(TRACE, INFO) << "found section:" << i
                          << " name:" << getSectionName(i)
                          << " size:" << prettyFormatSize(sections_[i].sh_size);
    }
}

CachedSymbolTable::CStrIterator CachedSymbolTable::strtab_begin(
    const char *str) const {
    return CachedSymbolTable::CStrIterator(str);
}

CachedSymbolTable::CStrIterator CachedSymbolTable::strtab_end(
    const char *str) const {
    return CachedSymbolTable::CStrIterator(str);
}

void CachedSymbolTable::move_to_section_header() {
    ifs.seekg(elf_header.e_shoff, std::ios::beg);
}

void CachedSymbolTable::move_to_section_header(size_t index) {
    move_to_section_header();
    size_t shstr_h_oft = sizeof(ElfW(Shdr)) * index;
    ifs.seekg(shstr_h_oft, std::ios::cur);
}

adt::StringRef CachedSymbolTable::getSectionName(size_t index) const {
    return adt::StringRef(&section_header_str.at(sections_.at(index).sh_name));
}

size_t CachedSymbolTable::find_section(adt::StringRef name) const {
    size_t index = 0;
    for (; index < sections_.size(); index++) {
        if (getSectionName(index) == name) {
            break;
        }
    }
    return index;
}

void CachedSymbolTable::load_symbol_table() {
    auto symbol_tb = load_section_data<ElfW(Sym)>(".symtab");
    auto xpu_symbol_tb = load_section_data<ElfW(Sym)>("XPU_KERNEL");
    auto strtab_buf = load_section_data(".strtab");
#if 0
    auto [begin, end] = strtab_range(buf.data(), buf.size());
#endif
    for (size_t i = 0; i < symbol_tb.size(); ++i) {
        if (strtab_buf.size() <= symbol_tb[i].st_name) {
            MLOG(TRACE, INFO)
                << "symbol_tb[" << i << "].st_name(" << symbol_tb[i].st_name
                << ") over buf size:" << strtab_buf.size();
            continue;
        }
        symbol_table.emplace(
            symbol_tb[i].st_value,
            adt::StringRef(&strtab_buf[symbol_tb[i].st_name]).str());
        if (symbol_tb[i].st_value > max_address_) {
            max_address_ = symbol_tb[i].st_value;
        }
        if (symbol_tb[i].st_value < min_address_) {
            min_address_ = symbol_tb[i].st_value;
        }
    }

    for (size_t i = 0; i < xpu_symbol_tb.size(); ++i) {
        if (strtab_buf.size() <= xpu_symbol_tb[i].st_name) {
            MLOG(TRACE, INFO) << "xpu_symbol_tb[" << i << "].st_name("
                              << xpu_symbol_tb[i].st_name
                              << ") over buf size:" << strtab_buf.size();
            continue;
        }
        symbol_table.emplace(
            xpu_symbol_tb[i].st_value,
            adt::StringRef(&strtab_buf[xpu_symbol_tb[i].st_name]).str());
    }

    MLOG(TRACE, INFO) << libName << "\naddress range:" << min_address_ << "~"
                      << max_address_;
}

void CachedSymbolTable::parse_section_header() {
    CHECK_EQ(sizeof(ElfW(Shdr)), elf_header.e_shentsize);
    move_to_section_header();
    sections_.resize(elf_header.e_shnum);
    MLOG(TRACE, INFO) << "elf_header.e_shnum:" << elf_header.e_shnum;
    ifs.read(reinterpret_cast<char *>(sections_.data()),
             sections_.size() * sizeof(sections_[0]));
}

void CachedSymbolTable::parse_named_section() {
    move_to_section_header(elf_header.e_shstrndx);

    ElfW(Shdr) shstr_h;
    ifs.read(reinterpret_cast<char *>(&shstr_h), sizeof(shstr_h));
    ifs.seekg(shstr_h.sh_offset, std::ios::beg);

    section_header_str.resize(shstr_h.sh_size);
    ifs.read(section_header_str.data(), section_header_str.size());
}

CachedSymbolTable::OwnerBuf<char> CachedSymbolTable::load_section_data(
    adt::StringRef name) {
    size_t section_index = find_section(name);
    if (section_index >= sections_.size()) {
        LOG(INFO) << "can't found secton: " << name;
        return {};
    }
    ifs.seekg(sections_[section_index].sh_offset, std::ios::beg);
    auto result = CachedSymbolTable::OwnerBuf<char>::alloc(
        sections_[section_index].sh_size);
    ifs.read(reinterpret_cast<char *>(result.data()), result.size());
    return result;
}

const std::string &CachedSymbolTable::lookUpSymbol(const void *func) const {
    static std::string empty("");
    auto offset = reinterpret_cast<const char *>(func) -
                  reinterpret_cast<const char *>(base_address);
    MLOG(TRACE, INFO) << "lookup address:" << offset;
    auto iter = symbol_table.find(offset);
    if (iter == symbol_table.end()) {
        MLOG(TRACE, INFO) << libName
                          << "\nnot find launch_async symbol offset:" << offset
                          << " base address:" << base_address
                          << " func address:" << func << " range("
                          << min_address_ << "~" << max_address_;
        return empty;
    }
    return iter->second;
}

static std::unordered_map<std::string, std::unique_ptr<CachedSymbolTable>>
    table;

CachedSymbolTable *createSymbolTable(const std::string &lib,
                                     const void *address) {
    auto iter = table.emplace(lib, new CachedSymbolTable(lib, address));
    return iter.first->second.get();
}

CachedSymbolTable *getSymbolTable(const std::string &lib) {
    return table[lib].get();
}

}  // namespace hook
