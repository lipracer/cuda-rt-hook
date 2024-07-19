#include <elf.h>
#include <link.h>

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "logger/logger.h"
#include "macro.h"

namespace hook {

/// @brief elf file parser for find symbol
class HOOK_API CachedSymbolTable {
   public:
    // CStrIterator: A class to iterate over C-style strings
    struct CStrIterator {
        CStrIterator(const char *str);

        // Pre-increment operator
        CStrIterator &operator++() &;

        // Post-increment operator
        CStrIterator operator++(int) &;

        // Dereference operator to get a StringRef
        adt::StringRef operator*();

        // Equality comparison operator
        bool operator==(const CStrIterator &other) const;

        // Inequality comparison operator
        bool operator!=(const CStrIterator &other) const;

        // Return the raw pointer to the string
        const void *data() const;

       private:
        const char *str_;
    };
    /// @brief simple buffer manager
    /// @tparam T
    template <typename T>
    struct OwnerBuf {
        OwnerBuf() = default;
        OwnerBuf(const OwnerBuf &other) = delete;
        OwnerBuf &operator=(const OwnerBuf &other) = delete;

        OwnerBuf(OwnerBuf &&other) { *this = std::move(other); }
        OwnerBuf &operator=(OwnerBuf &&other) {
            std::swap(buf_, other.buf_);
            std::swap(size_, other.size_);
            return *this;
        }

        static OwnerBuf alloc(size_t size) {
            OwnerBuf buf;
            buf.buf_ = reinterpret_cast<T *>(malloc(size * sizeof(T)));
            buf.size_ = size;
            return buf;
        }

        T &operator[](size_t index) {
            assert(index < size_);
            assert(buf_);
            return buf_[index];
        }

        T *data() const { return buf_; }

        size_t size() const { return size_; }

        std::tuple<T *, size_t> release() {
            auto result = std::make_tuple(buf_, size_);
            buf_ = 0;
            size_ = 0;
            return result;
        }

        template <typename N>
        friend class OwnerBuf;

        template <typename N>
        OwnerBuf<N> cast() {
            OwnerBuf<N> result;
            auto [buf, size] = release();
            result.buf_ = reinterpret_cast<N *>(buf);
            result.size_ = size * sizeof(T) / sizeof(N);
            return result;
        }

        ~OwnerBuf() {
            if (buf_) {
                free(buf_);
#ifndef NDEBUG
                buf_ = nullptr;
                size_ = 0;
#endif
            }
        }

       private:
        T *buf_ = nullptr;
        size_t size_ = 0;
    };

    CStrIterator strtab_begin(const char *str) const;

    CStrIterator strtab_end(const char *str) const;

    std::tuple<CStrIterator, CStrIterator> strtab_range(const char *str,
                                                        size_t size) const {
        return std::make_tuple(strtab_begin(str), strtab_begin(str + size));
    }

    CachedSymbolTable(const std::string &name, const void *base_address,
                      const std::vector<std::string> &section_names = {});

    /// @brief move to sections header location
    void move_to_section_header();

    /// @brief move target section location
    /// @param index
    void move_to_section_header(size_t index);

    adt::StringRef getSectionName(size_t index) const;
    size_t find_section(adt::StringRef name) const;

    void load_symbol_table();
    void parse_section_header();
    void parse_named_section();

    template <typename T>
    OwnerBuf<T> load_section_data(adt::StringRef name) {
        auto buf = load_section_data(name);
        return buf.cast<T>();
    }

    OwnerBuf<char> load_section_data(adt::StringRef name);

    const std::string &lookUpSymbol(const void *func) const;

    const std::unordered_map<size_t, std::string> &getSymbolTable() const {
        return symbol_table;
    }

    size_t min_addrtess() const { return min_address_; }
    size_t max_addrtess() const { return max_address_; }

    const std::vector<ElfW(Shdr)> &sections() const { return sections_; }

   private:
    std::string libName;
    std::ifstream ifs;
    ElfW(Ehdr) elf_header;
    std::vector<char> section_header_str;
    std::vector<ElfW(Shdr)> sections_;
    std::unordered_map<size_t, std::string> symbol_table;
    const void *base_address;
    size_t min_address_ = -1;
    size_t max_address_ = 0;
    std::vector<std::string> section_names;
};

/// @brief create symbol table
/// @param lib a elf file
/// @param address The address where the elf file is loaded at runtime
/// @return
CachedSymbolTable *createSymbolTable(const std::string &lib,
                                     const void *address);

/// @brief get symbol table
/// @param lib a elf file
/// @return
CachedSymbolTable *getSymbolTable(const std::string &lib);

}  // namespace hook
