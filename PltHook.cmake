add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/lib)
install(TARGETS plthook DESTINATION ${CMAKE_BINARY_DIR}/lib)
set(plthook "${CMAKE_BINARY_DIR}/lib/libplthook.a")