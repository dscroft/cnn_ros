#include <rclcpp/rclcpp.hpp>

#include <memory>

#include "bounding_component.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.use_intra_process_comms(true);
  rclcpp::spin(std::make_shared<cnn_ros::Bounding>(options));
  rclcpp::shutdown();
  return 0;
}