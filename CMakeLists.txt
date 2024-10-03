cmake_minimum_required(VERSION 3.15...3.26)
if(NOT DEFINED SKBUILD_PROJECT_NAME)
  set(SKBUILD_PROJECT_NAME "fast_crossing")
endif()
if(NOT DEFINED PROJECT_VERSION)
  set(PROJECT_VERSION "dev")
endif()
# https://scikit-build-core.readthedocs.io/en/latest/cmakelists.html#accessing-information
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE STRING "" FORCE)
  message(STATUS "Set build type to default: ${CMAKE_BUILD_TYPE}")
else()
  message(STATUS "Your build type: ${CMAKE_BUILD_TYPE}")
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -ggdb")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -ggdb")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

include_directories(${PROJECT_SOURCE_DIR}/headers/include
                    ${PROJECT_SOURCE_DIR}/headers/include/cubao)

set(PYBIND11_CPP_STANDARD -std=c++17)

# https://scikit-build-core.readthedocs.io/en/latest/getting_started.html
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
include_directories(headers/include)

file(GLOB SRCS src/*.cpp)
python_add_library(_core MODULE ${SRCS} WITH_SOABI)
target_link_libraries(_core PRIVATE pybind11::headers)
target_include_directories(_core PRIVATE src)
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})
install(TARGETS _core DESTINATION ${PROJECT_NAME})
