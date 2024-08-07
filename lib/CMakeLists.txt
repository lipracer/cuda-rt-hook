# add_library(logger STATIC 
#     logger.cpp
# )

# Although we used memory pool, it doesn't seem to have a significant advantage over stl.
# Perhaps we need to implement multi-level pooling, 
# introduce lock free queues, and avoid using stringstream
#target_compile_definitions(logger
#    PRIVATE USE_STL_QUEUE
#)

find_package(PythonLibs REQUIRED)

add_library(cuda_mock STATIC
    util.cpp
    env_util.cpp
    GlobalVarMgr.cpp
    logger.cpp
    env_mgr.cpp
    backtrace.cpp
    statistic.cpp
    elf_parser.cpp
    hook_context.cpp
    hook.cpp
    cuda_op_tracer.cpp
    cuda_mock.cpp
    xpu_mock.cpp
    any_hook.cpp
)

add_library(xpu_mock_improve SHARED hooks/print_hook.cpp hooks/skip_kernel_hook.cpp)
target_include_directories(xpu_mock_improve PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}../include)
set_target_properties(xpu_mock_improve
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${DST_DIRECTORY_TO_INSTALL}
)

if (NOT PUBLISH_BUILD)
    find_package(PythonLibs REQUIRED)
    target_include_directories(cuda_mock PUBLIC ${Python_INCLUDE_DIRS})
    target_link_libraries(cuda_mock PRIVATE ${PYTHON_LIBRARIES} xpu_mock_improve)
else ()
    message(STATUS "Use PUBLISH_BUILD")
    message(STATUS "PYTHON_INCLUDE_DIR is ${PYTHON_INCLUDE_DIR}")
    message(STATUS "PYTHON_LIBRARY is ${PYTHON_LIBRARY}")
    target_include_directories(cuda_mock PUBLIC ${PYTHON_INCLUDE_DIR})
    target_link_libraries(cuda_mock PRIVATE ${PYTHON_LIBRARY} xpu_mock_improve)
endif ()

set_target_properties(cuda_mock PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(cuda_mock PRIVATE fmt::fmt-header-only)
include_directories(PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/fmt/include)


# add_library(plthook STATIC
#     env_util.cpp
#     GlobalVarMgr.cpp
#     cuda_mock.cpp
#     hook.cpp
#     cuda_op_tracer.cpp
#     backtrace.cpp
#     statistic.cpp
#     hook_context.cpp
#     env_mgr.cpp
# )
# 
# target_compile_definitions(plthook
#     PRIVATE FMT_HEADER_ONLY
# )
# target_include_directories(plthook PUBLIC ${Python_INCLUDE_DIRS})
# target_link_libraries(plthook PRIVATE ${PYTHON_LIBRARIES})
# target_link_libraries(plthook PRIVATE pthread dl)

