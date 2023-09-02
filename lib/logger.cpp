#include "logger.h"

#include <string.h>

#include <atomic>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace logger {

inline size_t alignUp(size_t x, size_t align) {
    return (x + align - 1) & ~(align - 1);
}

inline size_t alignStringSize(size_t size) {
    return alignUp(size + 1 + sizeof(size_t), sizeof(size_t));
}

SimpleStringRef::SimpleStringRef(const char* str, size_t size)
    : size_(size + 1) {
    if (size == 0) {
        memset(reinterpret_cast<char*>(this) + sizeof(size_t), 0,
               sizeof(size_t));
        reinterpret_cast<char*>(this)[sizeof(size_t)] = '\n';
    } else {
        memcpy(reinterpret_cast<char*>(this) + sizeof(size_t), str, size);
        reinterpret_cast<char*>(this)[sizeof(size_t) + size] = '\n';
        reinterpret_cast<char*>(this)[sizeof(size_t) + size + 1] = '\0';
    }
    memset(reinterpret_cast<char*>(this) + sizeof(size_t) + size + 2, 'e',
           objSize() - invalidSize());
}

size_t SimpleStringRef::objSize() const { return alignStringSize(size_); }

size_t SimpleStringRef::invalidSize() const {
    return sizeof(size_t) + size_ + 1;
}

SimpleStringRef* SimpleStringRef::create(StringPool& pool, const char* str,
                                         size_t size) {
    auto buf = pool.allocStringBuf(size);
    auto strRef = new (buf) logger::SimpleStringRef(str, size);
    return strRef;
}

StringPool::StringPool(const FlushFunc& flush) : flushFunc_(flush) {
    pool_ = static_cast<char*>(malloc(kPoolSize()));
    currentPoolBegin_ = pool_;
    currentPoolEnd_ = currentPoolBegin_ + kPoolSize();
    begin_ = reinterpret_cast<SimpleStringRef*>(pool_);
}

StringPool::~StringPool() { free(pool_); }

char* StringPool::allocStringBuf(size_t size) {
    size_t asize = alignStringSize(size + 1);
    ++size_;
    if (currentPoolBegin_ + asize > currentPoolEnd_) {
        flushPool();
        size_ = 1;
        assert(currentPoolBegin_ >= pool_begin());
        return currentPoolBegin_;
    }
    auto ptr = currentPoolBegin_;
    currentPoolBegin_ += asize;
    assert(ptr >= pool_begin());
    return ptr;
}

void StringPool::flushPool() {
#if 0
    auto iter = this->begin();
    for (; iter != this->end();) {
        auto preIter = iter++;
        memset(reinterpret_cast<char*>(&*preIter), '*', sizeof(size_t));
    }
    flushFunc_(pool_, kPoolSize());
#else
    for (auto& curStr : *this) {
        assert(strlen(curStr.c_str()) < kPoolSize() && "ilegal str!");
        flushFunc_(curStr.c_str(), curStr.size());
    }

#endif
    currentPoolBegin_ = pool_;
}

class LogConsumer {
   public:
    LogConsumer()
        : th_(&LogConsumer::print, this),
          exit_(false),
          pool_(([](const char* str, size_t size) {
              fwrite(str, size, 1, stdout);
          })) {
        tmpBuffer_.resize(256);
    }

    void pushLog(std::stringstream& ss) {
        auto str = ss.str();
        {
            std::lock_guard<std::mutex> guard(mtx_);
            buf_.push_back(std::move(str));
            ++queueSzie_;

            // auto buf = pool_.allocStringBuf(str.size());
            // (void)new (buf) logger::SimpleStringRef(str.size(), str.c_str());
        }

        ss.clear();
        ss.str("");
    }

    void print() {
        std::cout << "start print" << std::endl;
        while (!exit_ || queueSzie_.load()) {
            if (buf_.empty()) {
                std::this_thread::yield();
                continue;
            } else {
#if 1
                mtx_.lock();
                std::string str = std::move(buf_.front());
                buf_.pop_front();
                --queueSzie_;
                mtx_.unlock();
                std::cout << str << std::endl;
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
                queueSzie_ -= consumeSize;
                mtx_.unlock();
                for (size_t i = 0; i < consumeSize; ++i) {
                    std::cout << tmpBuffer_[i] << std::endl;
                }
#endif
            }
            // std::this_thread::yield();
        }
        std::cout << "exit consuer loop" << std::endl;
    }

    ~LogConsumer() {
        exit_.store(true);
        th_.join();
    }

   private:
    std::mutex mtx_;
    std::deque<std::string> buf_;
    std::thread th_;
    std::atomic<bool> exit_;
    std::atomic<size_t> queueSzie_{0};
    std::vector<std::string> tmpBuffer_;
    StringPool pool_;
};

size_t StringLiteralBase::MN = 0;

LogStream& LogStream::instance() {
    static std::shared_ptr<LogConsumer> gLogConsumer =
        std::make_shared<LogConsumer>();
    // static thread_local std::unique_ptr<LogStream> __instance =
    //     std::make_unique<LogStream>(gLogConsumer);

    static thread_local LogStream* __instance = new LogStream(gLogConsumer);
    return *__instance;
}

LogStream::LogStream(std::shared_ptr<LogConsumer>& logConsumer)
    : logConsumer_(logConsumer) {
    auto strLevel = std::getenv("LOG_LEVEL");
    if (strLevel) {
        level_ = static_cast<LogLevel>(atoi(strLevel));
    }
}

LogStream::~LogStream() {}

void LogStream::flush() { logConsumer_->pushLog(ss_); }

thread_local std::chrono::high_resolution_clock::duration
    LogWrapper::totalDur{};

}  // namespace logger