cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dcgan)

find_package(Torch REQUIRED)

add_executable(dcgan dcgan.cpp)
target_link_libraries(dcgan "${TORCH_LIBRARIES}")
set_property(TARGET dcgan PROPERTY CXX_STANDARD 14)


add_executable(gemm gemm.cu)
target_link_libraries(gemm "${TORCH_LIBRARIES}")
set_property(TARGET gemm PROPERTY CXX_STANDARD 14)
