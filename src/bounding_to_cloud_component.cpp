#include "bounding_to_cloud_component.hpp"

#include <cv_bridge/cv_bridge.h>

#include <functional>
#include <string>

namespace cnn_ros
{

BoundingToCloud::BoundingToCloud(const rclcpp::NodeOptions &options) : Node("bounding", options)
{
    this->infoSub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>("camera_info", 1,
        std::bind(&BoundingToCloud::info_callback, this, std::placeholders::_1));

    this->imageSub_.subscribe(this, "depth");
    this->detectionSub_.subscribe(this, "detections");

    this->sync_ = std::make_unique<Sync>( SyncPolicy(10), this->imageSub_, this->detectionSub_ );
    this->sync_->registerCallback(
        std::bind(&BoundingToCloud::callback, this, std::placeholders::_1, std::placeholders::_2));

    this->pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("cloud", 1);
}

void BoundingToCloud::callback(const sensor_msgs::msg::Image::ConstSharedPtr &imageMsg, 
        const vision_msgs::msg::Detection2DArray::ConstSharedPtr &detectionMsg)
{
    //RCLCPP_INFO(this->get_logger(), "Received image and detections");

    if( this->pinmodel_.initialized() == false )
    {
        RCLCPP_ERROR(this->get_logger(), "Camera model not initialized");
        return;
    }

    if( imageMsg->encoding != sensor_msgs::image_encodings::TYPE_32FC1 )
    {
        RCLCPP_ERROR( this->get_logger(), "Unsupported depth image encoding format" );
        return;
    }

    cv::Mat image;

    try
    {
        //BGR8 image
        image = cv_bridge::toCvCopy(imageMsg)->image;
    }
    catch( cv_bridge::Exception& e )
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    using point_t = pcl::PointXYZRGBA;
    using cloud_t = pcl::PointCloud<point_t>;

    cloud_t cloud;
    cloud.points.resize( detectionMsg->detections.size() );

    auto distance = [&]( const vision_msgs::msg::BoundingBox2D &bbox ) -> float
    {
        /* check horizontal strip of pixels at vertical centre of bb 
            for the closest pixel */
        const float* row = image.ptr<float>( bbox.center.position.y );

        const int leftx = bbox.center.position.x - bbox.size_x / 2;
        const int rightx = bbox.center.position.x + bbox.size_x / 2;
        
        return *std::min_element( row + leftx, row + rightx +1 );
    };

    std::transform( detectionMsg->detections.begin(),
                    detectionMsg->detections.end(),
                    cloud.points.begin(),
                    [&]( const auto &detection )
                    {
                        const float z = distance( detection.bbox );

                        if( std::isnan(z) )
                        {
                            RCLCPP_ERROR( this->get_logger(), "NAN value detected in depth image" );
                            return point_t();
                        }

                        cv::Point3d ray = this->pinmodel_.projectPixelTo3dRay( 
                            cv::Point2d( detection.bbox.center.position.x, detection.bbox.center.position.y ) );

                        point_t point;
                        point.x = ray.x * z;
                        point.y = ray.y * z;
                        point.z = ray.z * z;

                        return point;
                    } );

    sensor_msgs::msg::PointCloud2 cloudMsg;
    pcl::toROSMsg( cloud, cloudMsg );
    cloudMsg.header = imageMsg->header;
    this->pub_->publish( cloudMsg );
}


} // namespace cnn_ros