# ggml tests
add_executable(test-conv-1d-grouped test-conv-1d-grouped.cpp)
target_link_libraries(test-conv-1d-grouped PRIVATE ggml ggml-cpu)
target_include_directories(test-conv-1d-grouped PRIVATE ${CMAKE_SOURCE_DIR}/ggml/include)
