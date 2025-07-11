cmake_minimum_required(VERSION 3.10)
project(seal_wrapper)

set(CMAKE_CXX_STANDARD 17)


find_package(SEAL REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(seal_wrapper bindings.cpp)
target_link_libraries(seal_wrapper PRIVATE SEAL::seal)