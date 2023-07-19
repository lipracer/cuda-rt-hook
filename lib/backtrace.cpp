#include "backtrace.h"

#include <dlfcn.h>
#include <elf.h>
#include <errno.h>
#include <execinfo.h>

#include <fstream>
#include <regex>

#include "logger.h"

namespace trace {

bool BackTraceCollection::CallStackInfo::snapshot() {
    void* buffer[kMaxStackDeep] = {0};
    char** symbols = nullptr;
    int num = backtrace(buffer, kMaxStackDeep);
    CHECK(num > 0, "Expect frams num {} > 0!", num);
    symbols = backtrace_symbols(buffer, num);
    if (symbols == nullptr) {
        return false;
    }
    LOG(INFO) << "get stack deep num:" << num;
    for (int j = 0; j < num; j++) {
        LOG(INFO) << "current frame " << j << " addr:" << buffer[j]
                  << " symbol:" << symbols[j];
        backtrace_addrs_.push_back(buffer[j]);
        backtrace_.emplace_back(symbols[j]);
    }
    free(symbols);
    return true;
}

BackTraceCollection::CallStackInfo gTestInstance(
    +[](const std::string& name) -> void* {
        return reinterpret_cast<void*>(0xbULL);
    });

void BackTraceCollection::CallStackInfo::test_feed_and_parse() {
    std::ifstream ifs("sample.log");
    std::string line;
    while (std::getline(ifs, line)) {
        backtrace_.emplace_back(std::move(line));
    }
    parse();
}

static std::vector<std::string> exec_shell(const std::string& cmd) {
    LOG(INFO) << "exec_shell:" << cmd;
    auto pf = popen(cmd.c_str(), "r");
    if (!pf) {
        return {};
    }
    std::vector<std::string> results;
    while (true) {
        constexpr size_t kMaxBufSize = 256;
        std::string buf(kMaxBufSize, 0);
        if (!fgets(const_cast<char*>(buf.data()), kMaxBufSize, pf)) {
            results.emplace_back(std::move(buf));
            break;
        }
    }
    if (pclose(pf) == -1) {
        LOG(INFO) << "exec_shell fail!";
        return {};
    }
    return results;
}

static std::vector<std::string> addr2line(const std::string& libName,
                                          const std::string& addr) {
    std::stringstream ss;
    ss << "addr2line -e " << libName << " -f " << addr;
    return exec_shell(ss.str());
}

struct MatchedInfo {
    std::string libName;
    std::string symbol;
    std::string textAddr;
    std::string rtAddr;
};

static MatchedInfo parse_backtrace_line(const std::string& line) {
    MatchedInfo result;
    std::pair<const char*, const char*> libRange;
    std::pair<const char*, const char*> symbolRange;
    std::pair<const char*, const char*> textAddrRange;
    std::pair<const char*, const char*> rtAddrRange;
    const char* ptr = line.c_str();
    libRange.first = line.c_str();
    while (*ptr) {
        switch (*ptr) {
            case '(': {
                libRange.second = ptr;
                symbolRange.first = ptr + 1;
            } break;
            case ')': {
                if (symbolRange.second) {  // found +
                    textAddrRange.second = ptr;
                } else {
                    symbolRange.second = ptr;
                }
            } break;
            case '+': {
                if (symbolRange.first) {
                    symbolRange.second = ptr;
                    textAddrRange.first = ptr + 1;
                }
            } break;
            case '[': {
                rtAddrRange.first = ptr + 1;
            } break;
            case ']': {
                rtAddrRange.second = ptr;
            } break;
        }
        ++ptr;
    }
    auto check_and_legal = [&](std::pair<const char*, const char*>& range) {
        CHECK((range.first && range.second) || (!range.first && !range.second),
              "parse fail:{}", line);
        if (!range.first && !range.second) {
            range.first = line.c_str();
            range.second = line.c_str();
        }
    };
    check_and_legal(libRange);
    check_and_legal(symbolRange);
    check_and_legal(textAddrRange);
    check_and_legal(rtAddrRange);
    result.libName = std::string(libRange.first, libRange.second);
    result.symbol = std::string(symbolRange.first, symbolRange.second);
    result.textAddr = std::string(textAddrRange.first, textAddrRange.second);
    result.rtAddr = std::string(rtAddrRange.first, rtAddrRange.second);
    return result;
}

bool BackTraceCollection::CallStackInfo::parse() {

    // backtrace format lib_name(symbol_name(+add)?) [address]
    for (auto& line : backtrace_) {
        MatchedInfo matchedInfo;
        // try {
        //     std::smatch m;
        //     std::regex e(R"((.*?)\((.*?)\+?(0[x|X].*?)\) \[(.*?)\])");
        //     if (!std::regex_search(line, m, e)) {
        //         LOG(WARN) << "backtrace has ilegal format line:" << line;
        //         continue;
        //     }
        //     if (m.empty()) {
        //         continue;
        //     }
        //     if (m.size() - 1 < sizeof(matchedInfo) / sizeof(std::string)) {
        //         continue;
        //     }
        //     matchedInfo.libName = m[1].str();
        //     matchedInfo.symbol = m[2].str();
        //     matchedInfo.textAddr = m[3].str();
        //     matchedInfo.rtAddr = m[4].str();
        // } catch (const std::exception& e) {
        //     std::cerr << e.what() << '\n';
        //     continue;
        // }
        matchedInfo = parse_backtrace_line(line);
        if (matchedInfo.symbol.empty()) {
            continue;
        }
        auto baseAddr = getBaseAddr_(matchedInfo.libName);
        if (matchedInfo.textAddr.empty() && !baseAddr) {
            continue;
        }
        if (matchedInfo.textAddr.empty()) {
            size_t rtAddr = 0;
            std::stringstream ss;
            ss << matchedInfo.rtAddr;
            ss >> rtAddr;
            CHECK_LT(reinterpret_cast<size_t>(baseAddr), rtAddr,
                     "runtime address:{} must less the lib base address{} !",
                     reinterpret_cast<size_t>(baseAddr), rtAddr);
            rtAddr -= reinterpret_cast<size_t>(baseAddr);
            ss.clear();
            ss.str("");
            ss << std::hex << rtAddr;
            ss >> matchedInfo.textAddr;
        }
        LOG(INFO) << "lib name:" << matchedInfo.libName;
        LOG(INFO) << "parse addr:" << matchedInfo.textAddr;
        auto lineInfo = addr2line(matchedInfo.libName, matchedInfo.textAddr);
        if (lineInfo.size() < 2) {
            continue;
        }
        line = lineInfo[0] + "(" + lineInfo[1] + ") " + "[" +
               matchedInfo.rtAddr + "]";
    }
    return true;
}

BackTraceCollection& BackTraceCollection::instance() {
    static BackTraceCollection self;
    return self;
}

void BackTraceCollection::collect_backtrace(const void* func_ptr) {
    auto iter = cached_map_.find(func_ptr);

    if (iter != cached_map_.end()) {
        ++std::get<1>(backtraces_[iter->second]);
        return;
    }
    cached_map_.insert(std::make_pair(func_ptr, backtraces_.size()));

    backtraces_.emplace_back(
        std::make_tuple(CallStackInfo([this](const std::string& name) -> void* {
                            auto iter = base_addrs_.find(name);
                            if (iter == base_addrs_.end()) {
                                return nullptr;
                            }
                            return iter->second;
                        }),
                        size_t(0)));
    if (!std::get<0>(backtraces_.back()).snapshot()) {
        LOG(2) << "can't get backtrace symbol!";
    }
}

void BackTraceCollection::dump() {
    std::ofstream ofs("./backtrace.log");

    for (auto& stack_info : backtraces_) {
        ofs << "ignore:[call " << std::get<1>(stack_info) << " times"
            << "]\n";
        std::get<0>(stack_info).parse();
        ofs << std::get<0>(stack_info);
    }
    ofs.flush();
    ofs.close();
}

std::ostream& operator<<(std::ostream& os,
                         const BackTraceCollection::CallStackInfo& info) {
    for (size_t i = 0; i < info.backtrace_.size(); ++i) {
        os << info.backtrace_[i] << "\n";
    }
    return os;
}

}  // namespace trace
