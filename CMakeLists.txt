project(cuda_mock)
cmake_minimum_required(VERSION 3.18...3.22)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build-core'.
  Running it directly will almost certainly not produce the desired
  result. If you are a user trying to install this package, use the
  command below, which will install all necessary build dependencies,
  compile the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to rerun the above
  after editing C++ files.")
endif()

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
list(APPEND CMAKE_PREFIX_PATH "third_party/nanobind")
find_package(nanobind CONFIG REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/nanobind)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib)


# Install directive for scikit-build-core
install(TARGETS cuda_mock_impl LIBRARY DESTINATION cuda_mock)
target_link_libraries(cuda_mock_impl LINK_PUBLIC ${Boost_LIBRARIES})
