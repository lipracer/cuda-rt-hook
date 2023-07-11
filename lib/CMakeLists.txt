add_library(hook_lib STATIC hook.cpp)

include_directories(PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/Backward-cpp)

set_target_properties(hook_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
