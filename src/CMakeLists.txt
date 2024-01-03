
add_library(cuda_mock_impl SHARED cuda_mock_impl.cpp)

target_link_libraries(cuda_mock_impl PRIVATE cuda_mock)

include_directories(PRIVATE ${CMAKE_SOURCE_DIR}/include)

set_target_properties(cuda_mock_impl
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/cuda_mock
)
