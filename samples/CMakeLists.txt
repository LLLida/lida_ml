project(samples)

function(add_sample TARGET)
  add_executable(${TARGET} ${TARGET}.cpp)
  target_link_libraries(${TARGET} PRIVATE lida_ml)
  set_target_properties(${TARGET} PROPERTIES
    # use C++20
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
endfunction(add_sample)

add_sample(tensors)
add_sample(grad)
add_sample(mnist)
