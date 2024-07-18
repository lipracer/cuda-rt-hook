#include "logger/logger.h"

#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <exception>
#include <future>
#include <iomanip>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "env_mgr.h"
#include "env_util.h"

static std::vector<std::string>& gLoggerLevelStringSet() {
    static std::vector<std::string> instance = {"INFO", "WARN", "ERROR",
                                                "FATAL"};
    return instance;
}

namespace hook {

template <>
struct str2value_impl<logger::LogLevel> {
    void operator()(logger::LogLevel& lvl, adt::StringRef str) {
        auto iter = std::find(std::begin(gLoggerLevelStringSet()),
                              std::end(gLoggerLevelStringSet()), str);
        if (iter != std::end(gLoggerLevelStringSet())) {
            lvl = static_cast<logger::LogLevel>(
                std::distance(std::begin(gLoggerLevelStringSet()), iter));
        } else {
            // default warning
            lvl = logger::LogLevel::warning;
        }
    }
};
}  // namespace hook

namespace logger {

inline size_t alignUp(size_t x, size_t align) {
    return (x + align - 1) & ~(align - 1);
}

inline size_t alignStringSize(size_t size) {
    return alignUp(size + 1 + SimpleStringRef::headerSize(),
                   SimpleStringRef::alignSize());
}

std::string getFileName(const std::string& path) {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << path;
    if (path.back() != '/') {
        ss << '/';
    }
    tm structTime;
    ::localtime_r(&in_time_t, &structTime);
    ss << "hook-";
    ss << getpid() << '-' << std::this_thread::get_id() << "-"
       << std::put_time(&structTime, "%Y-%m-%d-%X") << ".log";
    auto result = ss.str();
    for (auto& it : result) {
        if (it == ':') {
            it = '-';
        }
    }
    return result;
}

SimpleStringRef::SimpleStringRef(const char* str, size_t size) : size_(size) {
    if (size == 0) {
        memset(reinterpret_cast<char*>(this) + SimpleStringRef::headerSize(), 0,
               SimpleStringRef::alignSize());
    } else {
        memcpy(reinterpret_cast<char*>(this) + SimpleStringRef::headerSize(),
               str, objSize() - SimpleStringRef::headerSize());
    }
}

size_t SimpleStringRef::objSize() const { return alignStringSize(size_); }

size_t SimpleStringRef::invalidSize() const {
    return SimpleStringRef::headerSize() + size_ + 1;
}

SimpleStringRef* SimpleStringRef::create(StringPool& pool, const char* str,
                                         size_t size) {
    auto buf = pool.allocStringBuf();
    auto strRef = new (buf) logger::SimpleStringRef(str, size);
    assert(pool.pool_begin() <= buf && buf < pool.pool_end() &&
           "buf not in range!");
    return strRef;
}

#define CHECK_BEGIN_IN_RANGE()                                 \
    assert(pool_begin() <= reinterpret_cast<char*>(front()) && \
           reinterpret_cast<char*>(front()) < pool_end() &&    \
           "begin not in pool range!")

StringPool::StringRefIterator StringPool::StringRefIterator::operator++() {
    ptr_ = ptr_->next();
    return ptr_;
}
StringPool::StringRefIterator StringPool::StringRefIterator::operator++(int) {
    auto cur = ptr_;
    ptr_ = ptr_->next();
    return cur;
}

StringPool::StringPool(size_t pageSzie, const FlushFunc& flush)
    : flushFunc_(flush), pageSize_(pageSzie) {
    pool_ = static_cast<char*>(malloc(pageSize_));
    currentPoolBegin_ = pool_;
    currentPoolEnd_ = currentPoolBegin_ + pageSize_;
    begin_ = reinterpret_cast<SimpleStringRef*>(pool_);
}

StringPool::~StringPool() { free(pool_); }

char* StringPool::allocStringBuf() { return currentPoolBegin_; }

void StringPool::flushPool() {
#if 0
    auto iter = this->begin();
    for (; iter != this->end();) {
        auto preIter = iter++;
        memset(reinterpret_cast<char*>(&*preIter), '*', SimpleStringRef::headerSize());
    }
    flushFunc_(pool_, pageSize_);
#else
    for (auto& curStr : *this) {
        assert(strlen(curStr.c_str()) < pageSize_ && "ilegal str!");
        assert(curStr.c_str() && "flush null string!");
        flushFunc_(curStr.c_str(), curStr.size());
    }
#endif
}

void StringPool::push_back(const std::string& str) {
    size_t alignSize = alignStringSize(str.size());
    if (currentPoolBegin_ + alignSize > currentPoolEnd_) {
        flushPool();
        currentPoolBegin_ = pool_;
        assert(currentPoolBegin_ == pool_begin() && "flush reset error!");
        begin_ = reinterpret_cast<SimpleStringRef*>(pool_);
        if (LOGGER_UNLIKELY(currentPoolBegin_ + alignSize > currentPoolEnd_)) {
            flushFunc_(str.c_str(), str.size());
            size_ = 0;
            return;
        }
        size_ = 1;
    } else {
        ++size_;
    }
    (void)SimpleStringRef::create(*this, str.c_str(), str.size());
    currentPoolBegin_ += alignSize;
    assert(size() == debugSize() && "error size!");
}

void StringPool::pop_front() {
    --size_;
    ++begin_;
    CHECK_BEGIN_IN_RANGE();
}

bool StringPool::hasEnoughSpace(size_t size) {
    size_t asize = alignStringSize(size);
    return currentPoolBegin_ + asize <= currentPoolEnd_;
}

size_t StringPool::debugSize() {
    CHECK_BEGIN_IN_RANGE();
    size_t i = 0;
    auto iter = begin();
    while (iter != end()) {
        ++iter;
        ++i;
    }
    assert(reinterpret_cast<char*>(&*iter) <= pool_begin() + pageSize_ &&
           "offset error!");
    return i;
}

using StlDeque = std::deque<std::string>;

#ifdef USE_STL_QUEUE
using StringQueue = StlDeque;
#else
using StringQueue = StringPool;
#endif

inline void fwriteString(const std::string& str, std::FILE* fh) {
    fwrite(str.c_str(), str.size(), 1, fh);
}

inline void fwriteString(SimpleStringRef* str, std::FILE* fh) {
    fwrite(str->c_str(), str->size(), 1, fh);
}

class LogStreamCollection {
   public:
    static LogStreamCollection& instance();
    void collect(LogStream* stream);
    std::shared_ptr<LogConsumer> collect_consumer(
        const std::shared_ptr<LogConfig>& cfg);
    void release_all_stream();
    std::shared_ptr<LogConsumer>&& release_consumer() {
        return std::move(consumer_);
    }

    void regist_on_exit(const std::function<void(void)>& on_exit) {
        on_exit_ = on_exit;
    }

    const std::function<void(void)>& on_exit() const { return on_exit_; }

   private:
    std::mutex stream_mtx;
    std::mutex consumer_mtx;
    std::unordered_set<LogStream*> stream_set;
    std::shared_ptr<LogConsumer> consumer_;
    std::function<void(void)> on_exit_;
};

void regist_on_exit(const std::function<void(void)>& OnExit) {
    LogStreamCollection::instance().regist_on_exit(OnExit);
}

static bool checkPathNotExistCreateIt(const std::string& path) {
    if (access(path.c_str(), F_OK) == -1) {
        fprintf(stderr, "%s not exist try create it!\n", path.c_str());
        std::string cmd = ("mkdir -p ");
        cmd += path;
        if (std::system(cmd.c_str())) {
            return false;
        }
    }
    return true;
}

const struct {
    int number;
    const char* name;
} kFailureSignals[] = {
    {SIGSEGV, "SIGSEGV"}, {SIGILL, "SIGILL"},   {SIGFPE, "SIGFPE"},
    {SIGABRT, "SIGABRT"}, {SIGTERM, "SIGTERM"},
};

class LogConsumer : public std::enable_shared_from_this<LogConsumer> {
   public:
    LogConsumer(const std::shared_ptr<LogConfig>& cfg)
        : std::enable_shared_from_this<LogConsumer>(),
#ifndef USE_STL_QUEUE
          buf_(cfg->pageSize),
#endif
          exit_(false),
          cfg_(cfg) {
        tmpBuffer_.resize(256);
        auto path = hook::get_env_value<std::string>(env_mgr::LOG_OUTPUT_PATH);
        if (!path.empty()) {
            if (!checkPathNotExistCreateIt(path)) {
                fprintf(stderr, "%s not exist and try create fail!\n",
                        path.c_str());
            } else {
                path = getFileName(path);
                cfg_->stream = fopen(path.c_str(), "wt+");
                if (!cfg_->stream) {
                    fprintf(stderr, "can't open file:%s\n", path.c_str());
                    cfg_->stream = stdout;
                }
            }
        }
        if (cfg->mode == LogConfig::kAsync) {
            // future_ = promise_.get_future();
            th_ = std::make_unique<std::thread>(&LogConsumer::print, this);
        }
    }

    void pushLog(std::stringstream& ss) {
        if (LOGGER_UNLIKELY(exit_.load())) {
            return;
        }
        auto str = ss.str();
        {
            std::lock_guard<std::mutex> guard(mtx_);
            buf_.push_back(std::move(str));
        }
        ss.clear();
        ss.str("");
    }

    void notify() {
        {
            std::lock_guard<std::mutex> gl(mtx_);
            started_ = true;
        }
        cv_.notify_one();
    }

    void print() {
        // sometimes new threads start during the logconsumer build process,
        // which can cause a crash
        {
            std::unique_lock<std::mutex> ul(mtx_);
            cv_.wait(ul, [this]() { return this->started_; });
        }
        // increase the ref count avoid other thread release self
        auto self = this->shared_from_this();
        do {
            if (buf_.empty()) {
                goto LOOP_END;
            } else {
#if 1
                mtx_.lock();
                if (buf_.empty()) {
                    mtx_.unlock();
                    goto LOOP_END;
                }

#ifdef USE_STL_QUEUE
                auto str = std::move(buf_.front());
#else
                auto str = buf_.front();
#endif

                buf_.pop_front();
                mtx_.unlock();
                fwriteString(str, cfg_->stream);
#else
                mtx_.lock();
                size_t consumeSize = buf_.size() <= tmpBuffer_.size()
                                         ? buf_.size()
                                         : tmpBuffer_.size();
                std::copy(buf_.begin(), buf_.begin() + consumeSize,
                          tmpBuffer_.begin());
                for (size_t i = 0; i < consumeSize; ++i) {
                    buf_.pop_front();
                }
                mtx_.unlock();
                for (size_t i = 0; i < consumeSize; ++i) {
                    fwriteString(tmpBuffer_[i].c_str(), tmpBuffer_[i].size(),
                                 cfg_->stream);
                }
#endif
            }
        LOOP_END:
            if (self.use_count() > 1) {
                // https://stackoverflow.com/questions/17325888/c11-thread-waiting-behaviour-stdthis-threadyield-vs-stdthis-thread
                // yield always takes up 100 cpu time
                // std::this_thread::yield();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        } while (!exit_.load());
    }

    void sync_pause_loop(int signum) {
        static std::once_flag onceFlag;
        std::call_once(onceFlag, [this, signum = signum]() {
            exit_.store(true);
            if (cfg_->mode == LogConfig::kAsync) {
                if (th_ && th_->joinable()) th_->join();
            }
            flush_queue();
            auto iter = std::find_if(
                std::begin(kFailureSignals), std::end(kFailureSignals),
                [signum](auto& it) { return it.number == signum; });
            if (iter != std::end(kFailureSignals)) {
                fprintf(cfg_->stream, "[LOG END reason:%s]\n", iter->name);
            }
            fflush(cfg_->stream);
        });
    }

    void report_fatal() {
        sync_pause_loop(SIGUSR1);
        // write nullptr statement maybe be motion to front
        int n = 0;
        *reinterpret_cast<int*>(n) = 0;
    }

    void flush_queue() {
        while (buf_.size()) {
            auto str = buf_.front();
            buf_.pop_front();
            fwriteString(str, cfg_->stream);
        }
    }

    StringQueue& queue() { return buf_; }

   private:
    std::mutex mtx_;
    StringQueue buf_;
    std::atomic<bool> exit_;
    std::unique_ptr<std::thread> th_;
    std::vector<std::string> tmpBuffer_;
    std::shared_ptr<LogConfig> cfg_;
    std::condition_variable cv_;
    bool started_{false};
    // std::promise<int> promise_;
    // std::future<int> future_;
};

size_t StringLiteralBase::MN = 0;

std::thread::id LogStream::threadId() {
    static thread_local std::thread::id _id = std::this_thread::get_id();
    return _id;
}

void LogStreamCollection::collect(LogStream* stream) {
    std::lock_guard<std::mutex> lg(stream_mtx);
    stream_set.insert(stream);
}

std::shared_ptr<LogConsumer> LogStreamCollection::collect_consumer(
    const std::shared_ptr<LogConfig>& cfg) {
    if (consumer_) {
        return consumer_;
    }
    std::lock_guard<std::mutex> lg(consumer_mtx);
    if (!consumer_)
        consumer_ = std::shared_ptr<LogConsumer>(new LogConsumer(cfg));
    return consumer_;
}

void LogStreamCollection::release_all_stream() {
    for (auto stream : stream_set) {
        delete stream;
    }
}

LogStreamCollection& LogStreamCollection::instance() {
    static LogStreamCollection* __instance = new LogStreamCollection();
    return *__instance;
}

void destroy_logger();

void core_dump_handler(int signum) {
    auto consumer = LogStreamCollection::instance().release_consumer();
    // TODO: move to destructor
    consumer->sync_pause_loop(signum);
    auto on_exit = LogStreamCollection::instance().on_exit();
    if (on_exit) on_exit();
    exit(signum);
}

LogStream& LogStream::instance(const LogConfig& cfg) {
    auto sp_cfg = std::make_shared<LogConfig>(cfg);
    std::shared_ptr<LogConsumer> gLogConsumer =
        LogStreamCollection::instance().collect_consumer(sp_cfg);
    // static thread_local std::unique_ptr<LogStream> __instance =
    //     std::make_unique<LogStream>(gLogConsumer);

    static thread_local LogStream* __instance =
        new LogStream(gLogConsumer, sp_cfg);
    gLogConsumer->notify();

    for (auto& sig : kFailureSignals) {
        signal(sig.number, core_dump_handler);
    }
    return *__instance;
}

void setLoggerLevel(
    std::array<LogLevel, static_cast<size_t>(LogModule::last) + 1>& module_set_,
    LogLevel& level_) {
    adt::StringRef envValue =
        hook::get_env_value<adt::StringRef>(env_mgr::LOG_LEVEL);
    adt::StringRef mainLeveleStr, modulesStr;
    for (auto m = LogModuleHelper::begin(); m != LogModuleHelper::end(); ++m) {
        if (envValue.startsWith(*m)) {
            modulesStr = envValue;
            break;
        }
    }
    if (modulesStr.empty()) {
        size_t index = 0;
        for (auto chr : envValue) {
            if (chr == ',') {
                modulesStr = envValue.slice(index + 1);
                mainLeveleStr = envValue.slice(0, index);
                break;
            }
            index++;
        }
    }

    auto modules = hook::str2value<
        std::vector<std::pair<std::string, logger::LogLevel>>>()(modulesStr);

    std::fill(std::begin(module_set_), std::end(module_set_),
              logger::LogLevel::warning);
    for (auto name : LogModuleHelper::enum_strs()) {
        auto iter = std::find_if(
            modules.begin(), modules.end(),
            [&](const auto& env_v) { return env_v.first == name; });
        if (modules.end() != iter) {
            size_t IntModule =
                static_cast<size_t>(LogModuleHelper::strToEnum(name));
            assert(IntModule < sizeof(module_set_) / sizeof(module_set_[0]));
            module_set_[IntModule] = iter->second;
        }
    }
    level_ = hook::str2value<logger::LogLevel>()(mainLeveleStr);
}

LogStream::LogStream(std::shared_ptr<LogConsumer>& logConsumer,
                     const std::shared_ptr<LogConfig>& cfg)
    : logConsumer_(logConsumer), cfg_(cfg) {
#ifdef DEBUG_LOGER
    std::cout << "parse level_:" << static_cast<int>(level_) << std::endl;
#endif
    setLoggerLevel(module_set_, level_);
    LogStreamCollection::instance().collect(this);
    {
        std::stringstream ss;
        ss << "[PID:" << getpid() << "]"
           << "[TID:" << threadId() << "]";
        logHeader_ = ss.str();
    }
}

LogStream::~LogStream() {}

void LogStream::flush() {
    ss_ << "\n";
    if (cfg_->mode == LogConfig::kSync) {
        printf("%s", ss_.str().c_str());
        // fwriteString(ss_.str(), gLogConfig.stream);
        ss_.str("");
    } else {
        logConsumer_->pushLog(ss_);
    }
}

void LogStream::log_fatal() { logConsumer_->report_fatal(); }

void initLogger(const LogConfig& cfg) { (void)LogStream::instance(cfg); }

void destroy_logger() {
    LogStreamCollection::instance().release_all_stream();
    auto consumer = LogStreamCollection::instance().release_consumer();
    // TODO: move to destructor
    consumer->sync_pause_loop(0);
}

thread_local std::chrono::high_resolution_clock::duration
    LogWrapper::totalDur{};

}  // namespace logger
