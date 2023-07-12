
nanobind_add_module(cuda_mock_impl STABLE_ABI cuda_mock_impl.cpp)

target_link_libraries(cuda_mock_impl PRIVATE cuda_mock)

include_directories(PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)