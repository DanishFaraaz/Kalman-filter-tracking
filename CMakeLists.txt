project (CMT)

cmake_minimum_required (VERSION 2.6)

## load in pkg-config support
find_package(PkgConfig)
## use pkg-config to get hints for 0mq locations
pkg_check_modules(PC_ZeroMQ QUIET zmq)

## use the hint from above to find where 'zmq.hpp' is located
find_path(ZeroMQ_INCLUDE_DIR
        NAMES zmq.hpp
        PATHS ${PC_ZeroMQ_INCLUDE_DIRS}
        )

## use the hint from about to find the location of libzmq
find_library(ZeroMQ_LIBRARY
        NAMES zmq
        PATHS ${PC_ZeroMQ_LIBRARY_DIRS}
        )


find_package(OpenCV REQUIRED)
find_library(HAS_LIBUDEV udev)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

#add_definitions ("-Wall")
set( CMAKE_CXX_FLAGS "-Wall -pthread -std=c++11 -lv4l2 -udev" )
set(USB_INCLUDE_DIR ${LIBUDEV_INCLUDE_DIR})
set(V4L2_LIBRARIES v4l2)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${LIBUDEV_INCLUDE_DIR})
include_directories(${V4L2_LIBRARIES})


add_executable(contours contours.cpp withrobot_camera.cpp withrobot_utility.cpp)

target_link_libraries(contours ${OpenCV_LIBS} udev ${V4L2_LIBRARIES} ${ZeroMQ_INCLUDE_DIR})


