#ifndef BOUNDING_HPP

#include <algorithm>
#include <memory>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.hpp>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_geometry/pinhole_camera_model.h>


namespace cnn_ros
{

class BoundingToCloud : public rclcpp::Node
{
public:
    explicit BoundingToCloud(const rclcpp::NodeOptions &options);

private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;

    message_filters::Subscriber<sensor_msgs::msg::Image> imageSub_;
    message_filters::Subscriber<vision_msgs::msg::Detection2DArray> detectionSub_;

    using SyncPolicy = 
        message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, vision_msgs::msg::Detection2DArray>;
    using Sync =
        message_filters::Synchronizer<SyncPolicy>;
    std::unique_ptr<Sync> sync_;

    // image processing
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr infoSub_;

    image_geometry::PinholeCameraModel pinmodel_;

    void info_callback(const sensor_msgs::msg::CameraInfo &infoMsg)
    {
        RCLCPP_DEBUG(this->get_logger(), "Received camera info");
        this->pinmodel_.fromCameraInfo(infoMsg);
    }

    void callback(const sensor_msgs::msg::Image::ConstSharedPtr &imageMsg, 
        const vision_msgs::msg::Detection2DArray::ConstSharedPtr &detectionMsg);
};

} // namespace cnn_ros

#endif // BOUNDING_HPP