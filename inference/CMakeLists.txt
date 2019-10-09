cmake_minimum_required(VERSION 2.6)
project(Inference)
#set(CMAKE_CXX_COMPLIER /usr/local/bin/g++-9)
add_executable(inference main.cpp)

install(TARGETS inference DESTINATION ${CMAKE_INSTALL_PREFIX})