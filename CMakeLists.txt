cmake_minimum_required(VERSION 3.16)
project(rf C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CORE_SOURCES csrc/rf.c csrc/wl_api.c)

# Shared library
add_library(rf SHARED ${CORE_SOURCES})
target_include_directories(rf PUBLIC csrc)
target_compile_definitions(rf PRIVATE _POSIX_C_SOURCE=200809L)
target_compile_options(rf PRIVATE -Wall -Wextra -Wpedantic -O2)
target_link_libraries(rf m)
set_target_properties(rf PROPERTIES OUTPUT_NAME rf)

# Test executable
if(BUILD_TESTING)
    add_executable(test_rf test/test_c.c)
    target_link_libraries(test_rf rf)
    target_include_directories(test_rf PRIVATE csrc)
    target_link_libraries(test_rf m)
endif()
