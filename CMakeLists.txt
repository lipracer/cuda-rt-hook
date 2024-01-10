cmake_minimum_required(VERSION 3.18...3.22)

project(cuda_mock)

set(CMAKE_CXX_STANDARD 14)

if(NOT DEFINED CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wno-unused-parameter 
-Wno-unused-function -Wno-unused-result -Wno-missing-field-initializers 
-Wno-unknown-pragmas -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas 
-Wno-strict-overflow -Wno-strict-aliasing -Wno-missing-braces -Wno-uninitialized
-Werror=return-type -Werror)



find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

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


# Install directive for scikit-build-core
install(TARGETS cuda_mock_impl LIBRARY DESTINATION cuda_mock)

#set(DIRECTORY_TO_INSTALL ${CMAKE_CURRENT_SOURCE_DIR}/include)
#install(DIRECTORY ${DIRECTORY_TO_INSTALL} DESTINATION ${CMAKE_BINARY_DIR}/lib/cuda_mock)

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
