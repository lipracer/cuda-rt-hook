cmake_minimum_required(VERSION 3.18...3.22)

project(cuda_mock)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE Release)
endif()

set(compile_options -Wall -Wextra -Wno-unused-parameter 
-Wno-unused-function -Wno-unused-result -Wno-missing-field-initializers 
-Wno-unknown-pragmas -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas 
-Wno-strict-overflow -Wno-strict-aliasing -Wno-missing-braces -Wno-uninitialized
-Werror=return-type)

option(PUBLISH_BUILD "check is a publish build on ci" OFF)

if(NOT PUBLISH_BUILD)
  list(APPEND compile_options -Werror)
endif()

add_compile_options(${compile_options})

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
find_package(Options REQUIRED)

# set(Boost_INCLUDE_DIR "third_party/boost")
# list(APPEND CMAKE_PREFIX_PATH ${Boost_INCLUDE_DIR})
# find_package( Boost 1.82.0 COMPONENTS program_options REQUIRED )
# include_directories(${Boost_INCLUDE_DIR})

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake

# list(APPEND CMAKE_PREFIX_PATH "third_party/fmt")
# find_package(fmt)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/fmt EXCLUDE_FROM_ALL)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib)

set(DST_DIRECTORY_TO_INSTALL ${CMAKE_INSTALL_PREFIX}/cuda_mock)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION ${DST_DIRECTORY_TO_INSTALL})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fmt/include DESTINATION ${DST_DIRECTORY_TO_INSTALL})

if (ENABLE_BUILD_WITH_GTEST)

################################
# GTest
################################
enable_testing()
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest EXCLUDE_FROM_ALL)

file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test/**/*.cpp")
add_executable(UnitTest ${TEST_SOURCES})
target_link_libraries(UnitTest gtest gtest_main logger plthook)
add_test(NAME UnitTest COMMAND UnitTest)

target_include_directories(UnitTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/googletest/include)
target_include_directories(UnitTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fmt/include)
target_include_directories(UnitTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/lib)
target_include_directories(UnitTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

endif()
