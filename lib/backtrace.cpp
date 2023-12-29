#include "backtrace.h"

#include <Python.h>
#include <dlfcn.h>
#include <elf.h>
#include <errno.h>
#include <execinfo.h>
#include <frameobject.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <regex>

#include "logger/logger.h"
#include "support.h"

namespace trace {

bool CallFrames::CollectNative() {
    buffers_.clear();
    buffers_.resize(kMaxStackDeep, nullptr);
    native_frames_.reserve(kMaxStackDeep);
    char** symbols = nullptr;
    int num = backtrace(buffers_.data(), kMaxStackDeep);
    CHECK(num > 0, "Expect frams num {} > 0!", num);
    symbols = backtrace_symbols(buffers_.data(), num);
    if (symbols == nullptr) {
        return false;
    }
    Dl_info info;
    for (int j = 0; j < num; j++) {
        if (dladdr(buffers_[j], &info) && info.dli_sname) {
            auto demangled = __support__demangle(info.dli_sname);
            std::string path(info.dli_fname);
            std::stringstream ss;
            ss << "    frame " << j << path << ":" << demangled;
            native_frames_.push_back(ss.str());
        } else {
            // filtering useless print
            // LOG(WARN) << "    frame " << j << buffers_[j];
        }
    }
    free(symbols);
    return true;
}

bool CallFrames::CollectPython() {
    // https://stackoverflow.com/questions/33637423/pygilstate-ensure-after-py-finalize
    if (!Py_IsInitialized()) {
        LOG(WARN) << "python process finished!";
        return false;
    }
    python_frames_.reserve(kMaxStackDeep);
    // Acquire the Global Interpreter Lock (GIL) before calling Python C API
    // functions from non-Python threads.
    PyGILState_STATE gstate = PyGILState_Ensure();
    // https://stackoverflow.com/questions/1796510/accessing-a-python-traceback-from-the-c-api
    PyThreadState* tstate = PyThreadState_GET();
    if (NULL != tstate && NULL != tstate->frame) {
        PyFrameObject* frame = tstate->frame;

        while (NULL != frame) {
            // int line = frame->f_lineno;
            /*
            frame->f_lineno will not always return the correct line number
            you need to call PyCode_Addr2Line().
            */
            int line = PyCode_Addr2Line(frame->f_code, frame->f_lasti);
            const char* filename = PyUnicode_AsUTF8(frame->f_code->co_filename);
            const char* funcname = PyUnicode_AsUTF8(frame->f_code->co_name);
            std::stringstream ss;
            ss << "    " << filename << "(" << line << "): " << funcname;
            python_frames_.push_back(ss.str());
            frame = frame->f_back;
        }
    }
    PyGILState_Release(gstate);
    return !python_frames_.empty();
}

std::ostream& operator<<(std::ostream& os, const CallFrames& frames) {
    for (const auto& f : frames.python_frames_) {
        os << f << "\n";
    }
    for (const auto& f : frames.native_frames_) {
        os << f << "\n";
    }
    return os;
}

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
        LOG(WARN) << "popen cmd:" << cmd << "fail!";
        return {};
    }
    std::vector<std::string> results;
    while (true) {
        constexpr size_t kMaxBufSize = 1024;
        std::string buf(kMaxBufSize, 0);
        if (!fgets(const_cast<char*>(buf.data()), kMaxBufSize, pf)) {
            break;
        }
        // strip last '\n'
        auto str_length = strlen(buf.c_str());
        if (str_length > 0 && buf[str_length - 1] == '\n') {
            buf[str_length - 1] = '\0';
            str_length -= 1;
        }
        buf.resize(str_length);
        LOG(INFO) << buf;
        results.emplace_back(std::move(buf));
    }
    if (pclose(pf) == -1) {
        LOG(WARN) << "exec_shell fail!";
        return {};
    }
    return results;
}

static std::vector<std::string> addr2line(const std::string& libName,
                                          const std::string& addr) {
    std::stringstream ss;
    ss << "addr2line -e " << libName << " -f -C " << addr;
    return exec_shell(ss.str());
}

static std::string get_addr2line(const std::string& libName, size_t addr,
                                 const std::string& rt_addr) {
    std::stringstream ss;
    std::string textAddr;
    ss << "0x" << std::hex << addr;
    ss >> textAddr;
    auto result = addr2line(libName, textAddr);
    if (result.size() < 2) {
        return {};
    }
    if (result[0] == "??" && result[1] == "??:0") {
        return {};
    }
    return result[1] + "(" + result[0] + ") " + "[" + rt_addr + "]";
}

struct MatchedInfo {
    std::string libName;
    std::string symbol;
    std::string offset;
    std::string rtAddr;
};

static MatchedInfo parse_backtrace_line(const std::string& line) {
    MatchedInfo result;
    std::pair<const char*, const char*> libRange;
    std::pair<const char*, const char*> symbolRange;
    std::pair<const char*, const char*> offsetRange;
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
                    offsetRange.second = ptr;
                } else {
                    symbolRange.second = ptr;
                }
            } break;
            case '+': {
                if (symbolRange.first) {
                    symbolRange.second = ptr;
                    offsetRange.first = ptr + 1;
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
    check_and_legal(offsetRange);
    check_and_legal(rtAddrRange);
    result.libName = std::string(libRange.first, libRange.second);
    result.symbol = std::string(symbolRange.first, symbolRange.second);
    result.offset = std::string(offsetRange.first, offsetRange.second);
    result.rtAddr = std::string(rtAddrRange.first, rtAddrRange.second);
    return result;
}

bool BackTraceCollection::CallStackInfo::parse() {
    if (!getBaseAddr_) {
        return false;
    }
    // backtrace format lib_name(symbol_name(+add)?) [address]
    std::vector<std::string> tmp_backtrace;
    tmp_backtrace.reserve(backtrace_.size());
#define push_and_continue()        \
    tmp_backtrace.push_back(line); \
    continue;

    for (auto& line : backtrace_) {
        MatchedInfo matchedInfo;
#if 0
        try {
            std::smatch m;
            std::regex e(R"((.*?)\((.*?)\+?(0[x|X].*?)\) \[(.*?)\])");
            if (!std::regex_search(line, m, e)) {
                LOG(WARN) << "backtrace has ilegal format line:" << line;
                push_and_continue();
            }
            if (m.empty()) {
                push_and_continue();
            }
            if (m.size() - 1 < sizeof(matchedInfo) / sizeof(std::string)) {
                push_and_continue();
            }
            matchedInfo.libName = m[1].str();
            matchedInfo.symbol = m[2].str();
            matchedInfo.textAddr = m[3].str();
            matchedInfo.rtAddr = m[4].str();
        } catch (const std::exception& e) {
            std::cerr << e.what() << '\n';
            push_and_continue();
        }
#endif
        matchedInfo = parse_backtrace_line(line);
        auto baseAddr = getBaseAddr_(matchedInfo.libName);
        if (matchedInfo.rtAddr.empty() || !baseAddr) {
            push_and_continue();
        }
        std::string textAddr;

        size_t rtAddr = 0;
        std::stringstream ss;
        ss << std::hex << matchedInfo.rtAddr;
        ss >> rtAddr;
        if (reinterpret_cast<size_t>(baseAddr) >= rtAddr) {
            push_and_continue();
        }
        rtAddr -= reinterpret_cast<size_t>(baseAddr);
        ss.clear();
        ss.str("");

        auto parsed_line =
            get_addr2line(matchedInfo.libName, rtAddr, matchedInfo.rtAddr);
        if (!parsed_line.empty()) {
            tmp_backtrace.push_back(parsed_line);
        } else {
            tmp_backtrace.push_back(line);
        }

        if (!matchedInfo.offset.empty()) {
            size_t offset = 0;
            ss << std::hex << matchedInfo.offset;
            ss >> offset;
            rtAddr -= offset;
            ss.clear();
            ss.str("");
        }
        auto parsed_offset_line =
            get_addr2line(matchedInfo.libName, rtAddr, matchedInfo.rtAddr);
        if (!parsed_offset_line.empty()) {
            tmp_backtrace.push_back(parsed_offset_line);
        }
    }
    backtrace_.swap(tmp_backtrace);
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

    backtraces_.emplace_back(std::make_tuple(
        CallStackInfo(std::bind(&BackTraceCollection::getBaseAddr, this,
                                std::placeholders::_1)),
        size_t(0)));
    if (!std::get<0>(backtraces_.back()).snapshot()) {
        LOG(WARN) << "can't get backtrace symbol!";
    }
}

static std::vector<std::string> split_string(const std::string& str) {
    std::vector<std::string> result;
    std::stringstream ss;
    ss << str;
    std::copy(std::istream_iterator<std::string>(ss),
              std::istream_iterator<std::string>(), std::back_inserter(result));
    return result;
}

const void* BackTraceCollection::getBaseAddr(const std::string& name) {
#if 0
    auto iter = base_addrs_.find(name);
    if (iter == base_addrs_.end()) {
        return nullptr;
    }
    return iter->second;
#else
    auto it = link_maps_.begin();
    for (; it != link_maps_.end(); ++it) {
        if (it->find(name) != std::string::npos) {
            break;
        }
    }
    if (it == link_maps_.end()) {
        return nullptr;
    }
    // link map, first item is base address or second?
    // it -= 1;
    auto map_info = split_string(*it);
    if (map_info.empty()) {
        return nullptr;
    }
    auto pos = map_info[0].find("-");
    if (pos == std::string::npos) {
        return nullptr;
    }
    auto strBaseAddr = map_info[0].substr(0, pos);
    std::stringstream ss;
    ss << std::hex << strBaseAddr;
    size_t baseAddr = 0;
    ss >> baseAddr;
    return reinterpret_cast<void*>(baseAddr);
#endif
}

void BackTraceCollection::parse_link_map() {
    char cmd[64] = {0};
    sprintf(cmd, "cat /proc/%d/maps", getpid());
    link_maps_ = exec_shell(cmd);
}

void BackTraceCollection::dump() {
    if (backtraces_.empty()) {
        return;
    }
    parse_link_map();
    for (const auto& baseAddr : base_addrs_) {
        LOG(WARN) << baseAddr.first << " base address:" << baseAddr.second
                  << "\n";
    }
    for (auto& stack_info : backtraces_) {
        LOG(WARN) << "ignore:[call " << std::get<1>(stack_info) << " times"
                  << "]\n";
        LOG(WARN) << std::get<0>(stack_info);
        if (!std::get<0>(stack_info).parse()) {
            LOG(WARN) << "parse fail!";
        }
        LOG(WARN) << "=========================parsed backtrace "
                     "symbol=========================";
        LOG(WARN) << std::get<0>(stack_info);
    }
}

std::ostream& operator<<(std::ostream& os,
                         const BackTraceCollection::CallStackInfo& info) {
    for (size_t i = 0; i < info.backtrace_.size(); ++i) {
        os << info.backtrace_[i] << "\n";
    }
    return os;
}

}  // namespace trace
