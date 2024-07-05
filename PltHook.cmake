add_library(cuda_mock SHARED IMPORTED) 
set_target_properties(cuda_mock PROPERTIES IMPORTED_LOCATION ${CMAKE_MODULE_PATH}/libcuda_mock_impl.so)
