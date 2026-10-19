# Loaded by CMAKE_PROJECT_LIBJXL_INCLUDE in the unmodified upstream project.
# Target resolution happens after its jxl/jxl_threads/hwy definitions exist.
get_filename_component(adapter_dir "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
add_executable(iqa-ssim2-primary
  "${adapter_dir}/ssimulacra2_primary_compute.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/ssimulacra2.cc")
target_include_directories(iqa-ssim2-primary PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(iqa-ssim2-primary jxl jxl_threads hwy)
