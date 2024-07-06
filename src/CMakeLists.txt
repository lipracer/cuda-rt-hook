
add_library(cuda_mock_impl SHARED cuda_mock_impl.cpp)

target_link_libraries(cuda_mock_impl PRIVATE cuda_mock)

include_directories(PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(cuda_mock_impl PRIVATE fmt::fmt-header-only)

set_target_properties(cuda_mock_impl
    PROPERTIES
    # LINK_OPTIONS "-static-libstdc++"
    LIBRARY_OUTPUT_DIRECTORY ${DST_DIRECTORY_TO_INSTALL}
    BUILD_WITH_INSTALL_RPATH OFF
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN"
)
