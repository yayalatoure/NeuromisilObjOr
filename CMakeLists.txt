cmake_minimum_required(VERSION 3.5)
project(NeuroObjOr)

find_package(OpenCV REQUIRED)
find_package(Qt5Widgets REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

file(GLOB SOURCES *.h *.cpp)

set(SOURCE_FILES main.cpp foot.cpp foot.h)

add_executable(NeuroObjOr ${SOURCE_FILES} ${SOURCES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(NeuroObjOr ${OpenCV_LIBS})

target_link_libraries(NeuroObjOr Qt5::Widgets)