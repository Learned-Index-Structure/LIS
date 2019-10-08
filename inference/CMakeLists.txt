cmake_minimum_required (VERSION 2.6)
project (Inference)
add_executable(inference main.cpp)

install (TARGETS inference DESTINATION ${CMAKE_INSTALL_PREFIX})