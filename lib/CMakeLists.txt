add_library(cuda_mock STATIC 
    cuda_mock.cpp 
    hook.cpp 
    cuda_op_tracer.cpp 
    logger.cpp
    backtrace.cpp
)

include_directories(PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/Backward-cpp)

set_target_properties(cuda_mock PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(cuda_mock PRIVATE fmt::fmt-header-only)
include_directories(PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/fmt/include)

add_library(plthook STATIC
    cuda_mock.cpp 
    hook.cpp 
    cuda_op_tracer.cpp 
    logger.cpp
    backtrace.cpp
)
target_compile_definitions(plthook
    PRIVATE FMT_HEADER_ONLY
)

target_link_libraries(plthook PRIVATE pthread)
target_link_libraries(plthook PRIVATE dl)
