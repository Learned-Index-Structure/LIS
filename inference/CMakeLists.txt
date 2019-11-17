cmake_minimum_required(VERSION 2.6)
project(Inference)
set(CMAKE_CXX_COMPLIER /usr/local/bin/g++-9)

IF (CMAKE_VERSION VERSION_LESS "3.8")
    SET(CMAKE_CXX_STANDARD 14)
ELSEIF (CMAKE_VERSION VERSION_LESS "3.11")
    SET(CMAKE_CXX_STANDARD 17)
ELSE() # CMake 3.11 or higher:
    SET(CMAKE_CXX_STANDARD 20)
ENDIF()

# Typically, you'll also want to turn off compiler-specific extensions:
SET(CMAKE_CXX_EXTENSIONS OFF)
#add_executable(inference main.cpp avx_test.cpp avx_test.h)
#add_executable(inference main.cpp avx_new.cpp avx_new.h)
set(CMAKE_CXX_FLAGS -O3)
set(CMAKE_CXX_FLAGS -mavx2)

#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -mavx2)
#add_executable(inference avx_test.cpp avx_test.h)
#add_executable(inference avx_new.cpp avx_new.h)
#add_executable(inference main.cpp)
#add_executable(inference avx_mm.cpp avx_mm.h)
#add_executable(bench benchmark.cpp inference.cpp inference.hpp btree.hpp lms_algo.hpp)
#add_executable(infer inference.cpp inference.hpp)
add_executable(infer btree.hpp btree.cpp)
#add_executable(infer inference64.cpp )
install(TARGETS infer DESTINATION ${CMAKE_INSTALL_PREFIX})