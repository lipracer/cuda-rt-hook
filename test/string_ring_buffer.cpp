
#include <stdlib.h>

#include <algorithm>
#include <deque>
#include <fstream>
#include <numeric>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "logger.h"

#define RANDOM(N) (rand() % (N))

constexpr size_t kLoopCount = 100000;

static void flushBuffer(const char* str, size_t size, std::FILE* fs) {
    fwrite(str, size, 1, fs);
}

TEST(RingBufferTest, rand64_stl) {
    auto fs = fopen("/tmp/rand64_stl.log", "w");
    std::deque<std::string> pool;
    const size_t kMaxSzie = 23280;
    for (size_t i = 0; i < kLoopCount; ++i) {
        auto len = RANDOM(64);
        std::string str(len, '1');
        str.push_back('\n');
        pool.push_back(str);
        if (pool.size() > kMaxSzie) {
            for (auto& str : pool) {
                fwrite(str.c_str(), str.size() + 1, 1, fs);
            }
            pool.clear();
        }
        EXPECT_EQ(len + 1, str.size());
        EXPECT_EQ(len + 1, strlen(str.c_str()));
    }
    fflush(fs);
    fclose(fs);
}

TEST(RingBufferTest, rand64) {
    auto fs = fopen("/tmp/rand64.log", "w");
    logger::StringPool pool(
        [&](const char* str, size_t size) { flushBuffer(str, size, fs); });

    for (size_t i = 0; i < kLoopCount; ++i) {
        auto len = RANDOM(64);
        std::string str(len, '1');
        auto strRef =
            logger::SimpleStringRef::create(pool, str.c_str(), str.size());

        EXPECT_EQ(len + 1, strRef->size());
        EXPECT_EQ(len + 1, strlen(strRef->c_str()));
    }
    fflush(fs);
    fclose(fs);
}

// TEST(RingBufferTest, rand100) {
//     logger::StringPool pool(flushBuffer);

//     for (size_t i = 0; i < kLoopCount; ++i) {
//         auto len = RANDOM(100);
//         std::string str(len, '1');
//         auto buf = pool.allocStringBuf(str.size());
//         auto strRef =
//             new (buf) logger::SimpleStringRef(str.size(), str.c_str());
//         EXPECT_EQ(len + 1, strRef->size());
//         EXPECT_EQ(len + 1, strlen(strRef->c_str()));
//         // fprintf(stdout, "%d %s", len, strRef->c_str());
//         // std::cout << "offset:" << reinterpret_cast<char*>(strRef) -
//         // pool.pool() << std::endl;
//     }
// }

// TEST(RingBufferTest, rand110) {
//     logger::StringPool pool(flushBuffer);

//     for (size_t i = 0; i < kLoopCount; ++i) {
//         auto len = RANDOM(110);
//         std::string str(len, '1');
//         auto buf = pool.allocStringBuf(str.size());
//         auto strRef =
//             new (buf) logger::SimpleStringRef(str.size(), str.c_str());
//         EXPECT_EQ(len + 1, strRef->size());
//         EXPECT_EQ(len + 1, strlen(strRef->c_str()));
//         // fprintf(stdout, "%d %s", len, strRef->c_str());
//         // std::cout << "offset:" << reinterpret_cast<char*>(strRef) -
//         // pool.pool() << std::endl;
//     }
// }