add_library(cuda_mock STATIC cuda_mock.cpp hook.cpp cuda_op_tracer.cpp logger.cpp)

include_directories(PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/Backward-cpp)

set_target_properties(cuda_mock PROPERTIES POSITION_INDEPENDENT_CODE ON)
