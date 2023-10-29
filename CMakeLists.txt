cmake_minimum_required(VERSION 3.10)
project(LIDA_ML VERSION 0.1)

add_library(lida_ml STATIC "lida_ml.c" "lida_ml.h" "lida_ml_basic.c" "lida_ml_basic.h" "lida_ml.hpp")

target_include_directories(lida_ml PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

set_target_properties(lida_ml PROPERTIES
  # use C99
  C_STANDARD 99
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
  # use C++20
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)

# enable all warnings from compiler
target_compile_options(lida_ml PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>                         # MSVC
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Wno-unused-function> # g++ or clang
)

set(ENABLE_ASAN 1)

if (ENABLE_ASAN)
  target_compile_options(lida_ml PUBLIC
    -fsanitize=address -fno-omit-frame-pointer)
  target_link_options(lida_ml PUBLIC
    -fsanitize=address -fno-omit-frame-pointer -lrt)
endif ()

option(LIDA_ML_BUILD_SAMPLES "Enable building of samples." OFF)

if (${LIDA_ML_BUILD_SAMPLES})
  add_subdirectory(samples)
endif ()
