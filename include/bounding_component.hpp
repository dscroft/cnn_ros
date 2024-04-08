#ifndef BOUNDING_HPP

#include <algorithm>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "easydnn.h"

namespace cnn_ros
{

class Bounding : public rclcpp::Node
{
public:
    explicit Bounding(const rclcpp::NodeOptions &options);

private:
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pubDebug_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;

    EasyDNN dnn_;
    cv::Mat image_;
    std::vector<std::string> labels_;

    void image_callback(const sensor_msgs::msg::Image &msg);
};

} // namespace cnn_ros

#endif // BOUNDING_HPP