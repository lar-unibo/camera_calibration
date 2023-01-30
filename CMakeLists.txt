cmake_minimum_required(VERSION 3.0.2)
project(camera_calibration)


set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  sensor_msgs
  tf2_ros
  rospy
)


catkin_package(
  CATKIN_DEPENDS
  geometry_msgs
  sensor_msgs
  tf2_ros
)

include_directories(${catkin_INCLUDE_DIRS})
