
add_library(cuda_mock_impl SHARED cuda_mock_impl.cpp)

target_link_libraries(cuda_mock_impl PRIVATE cuda_mock)

include_directories(PRIVATE ${CMAKE_SOURCE_DIR}/include)

set_target_properties(cuda_mock_impl
    PROPERTIES
    LINK_OPTIONS "-static-libstdc++"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX}/cuda_mock
)
