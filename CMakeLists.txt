cmake_minimum_required(VERSION 3.16)
project(sota_fer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Apple Silicon (M1/M2/M3/M4) – prefer native arm64
if(APPLE)
  set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "" FORCE)
endif()

# OpenCV – Homebrew path on Apple Silicon is typically /opt/homebrew
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui dnn objdetect imgcodecs videoio)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")

add_executable(sota_fer
  src/demo.cpp
  src/face_detector.cpp
  src/fer_model.cpp
)

target_include_directories(sota_fer PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(sota_fer PRIVATE ${OpenCV_LIBS})

# Nice warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(sota_fer PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Install / RPATH niceties for macOS
if(APPLE)
  set_target_properties(sota_fer PROPERTIES
    BUILD_RPATH "@executable_path"
    INSTALL_RPATH "@executable_path"
  )
endif()

install(TARGETS sota_fer RUNTIME DESTINATION bin)
