add_library(hook_lib STATIC hook.cpp)

include_directories(PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

set_target_properties(hook_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(hook_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/Backward-cpp)