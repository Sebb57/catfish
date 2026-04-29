cmake_minimum_required(VERSION 3.20)
project(john LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(ExternalProject)
set(LIBTORCH_INSTALL_DIR "${CMAKE_BINARY_DIR}/libtorch_sdk")
set(TORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.2.2%2Bcpu.zip")

ExternalProject_Add(libtorch_external
    URL ${TORCH_URL}
    PREFIX ${CMAKE_BINARY_DIR}/external/libtorch
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/ ${LIBTORCH_INSTALL_DIR}
)

set(TORCH_INCLUDE_DIRS 
    "${LIBTORCH_INSTALL_DIR}/include"
    "${LIBTORCH_INSTALL_DIR}/include/torch/csrc/api/include"
)
set(TORCH_LIB_DIR "${LIBTORCH_INSTALL_DIR}/lib")

add_executable(${PROJECT_NAME} 
    src/main.cpp
)
target_include_directories(${PROJECT_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_directories(${PROJECT_NAME} PRIVATE ${TORCH_LIB_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${TORCH_LIB_DIR}/libtorch.so 
    ${TORCH_LIB_DIR}/libtorch_cpu.so 
    ${TORCH_LIB_DIR}/libc10.so
)
add_dependencies(${PROJECT_NAME} libtorch_external)
set_target_properties(${PROJECT_NAME} PROPERTIES 
    INSTALL_RPATH "${TORCH_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)
