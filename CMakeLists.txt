cmake_minimum_required(VERSION 3.8)
project(cnn_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

include_directories(
  include  
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

link_directories(
  ${OpenCV_LIBRARY_DIRS}
  ${PCL_LIBRARY_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

# bounding component
ament_auto_add_library(bounding_component SHARED
  src/bounding_component.cpp 
  src/easydnn.cpp
)
target_link_libraries(bounding_component
  ${OpenCV_LIBRARIES}
)
rclcpp_components_register_nodes(bounding_component
  "cnn_ros::Bounding"
)
ament_auto_add_executable(bounding_node
  src/bounding_node.cpp
)

# bounding to cloud component
ament_auto_add_library(bounding_to_cloud_component SHARED
  src/bounding_to_cloud_component.cpp
)
target_include_directories(bounding_to_cloud_component
  PUBLIC
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(bounding_to_cloud_component
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARY_DIRS}
)
rclcpp_components_register_nodes(bounding_to_cloud_component
  "cnn_ros::BoundingToCloud"
)
ament_auto_add_executable(bounding_to_cloud_node
  src/bounding_to_cloud_node.cpp
)

# Install launch files.
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()
