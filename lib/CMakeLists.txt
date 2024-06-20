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
    GlobalVarMgr.cpp
    logger.cpp
    env_mgr.cpp
    backtrace.cpp
    statistic.cpp
    hook.cpp
    cuda_op_tracer.cpp
    cuda_mock.cpp
    xpu_mock.cpp
    triton_mock.cpp
)

add_library(xpu_mock_improve SHARED hooks/print_hook.cpp)
target_include_directories(xpu_mock_improve PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}../include)
set_target_properties(xpu_mock_improve
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX}/cuda_mock
)

target_include_directories(cuda_mock PUBLIC ${Python_INCLUDE_DIRS})
target_link_libraries(cuda_mock PRIVATE ${PYTHON_LIBRARIES} xpu_mock_improve)


set_target_properties(cuda_mock PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(cuda_mock PRIVATE fmt::fmt-header-only)
include_directories(PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/fmt/include)


# add_library(plthook STATIC
#     GlobalVarMgr.cpp
#     cuda_mock.cpp
#     hook.cpp
#     cuda_op_tracer.cpp
#     backtrace.cpp
#     statistic.cpp
#     env_mgr.cpp
# )
# 
# target_compile_definitions(plthook
#     PRIVATE FMT_HEADER_ONLY
# )
# target_include_directories(plthook PUBLIC ${Python_INCLUDE_DIRS})
# target_link_libraries(plthook PRIVATE ${PYTHON_LIBRARIES})
# target_link_libraries(plthook PRIVATE pthread dl)

