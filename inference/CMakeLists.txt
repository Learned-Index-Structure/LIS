cmake_minimum_required(VERSION 2.6)
project(Inference)
set(CMAKE_CXX_COMPLIER /usr/local/bin/g++)

IF (CMAKE_VERSION VERSION_LESS "3.8")
    SET(CMAKE_CXX_STANDARD 14)
ELSEIF (CMAKE_VERSION VERSION_LESS "3.11")
    SET(CMAKE_CXX_STANDARD 17)
ELSE() # CMake 3.11 or higher:
    SET(CMAKE_CXX_STANDARD 20)
ENDIF()

# Typically, you'll also want to turn off compiler-specific extensions:
SET(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS -O3)
set(CMAKE_CXX_FLAGS -mavx2)

#add_executable(infer benchmark.cpp inference.cpp inference.hpp btree.hpp lms_algo.hpp)
add_executable(infer inference.cpp inference.hpp)
#add_executable(infer btree.hpp btree.cpp)
install(TARGETS infer DESTINATION ${CMAKE_INSTALL_PREFIX})