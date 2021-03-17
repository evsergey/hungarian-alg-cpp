cmake_minimum_required(VERSION 3.5)
project(hungarian VERSION 1.0.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (MSVC)
    # warning level 4 and all warnings as errors
    add_compile_options(/W4 /WX)
else()
    # lots of warnings and all warnings as errors
    add_compile_options(-Wall -Wextra -pedantic -Werror)
    add_compile_options(-fPIC)
endif()

include_directories(include)

include(CTest)
if(BUILD_TESTING)
    add_executable(test_hungarian test/test.cpp include/hungarian.h)
    add_test(NAME HungarianAlgTest COMMAND test_hungarian WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/test")
endif()