#include "bounding_component.hpp"

#include <cv_bridge/cv_bridge.h>

#include <functional>
#include <string>

namespace cnn_ros
{

Bounding::Bounding(const rclcpp::NodeOptions &options) : Node("bounding", options)
{
    //parameters
    const std::string cfg = 
        this->declare_parameter<std::string>("cfg", "model.cfg");
    const std::string weights = 
        this->declare_parameter<std::string>("weights", "model.weights");
    const bool cpu = 
        this->declare_parameter<bool>("cpu", false);

    this->labels_ = 
        this->declare_parameter<std::vector<std::string>>("labels", std::vector<std::string>() );

    RCLCPP_INFO(this->get_logger(), "cfg: %s", cfg.c_str());
    RCLCPP_INFO(this->get_logger(), "weights: %s", weights.c_str());
    RCLCPP_INFO(this->get_logger(), "labels: %ld", this->labels_.size() );

    // declare target hardware for cnn processing, default to CUDA
    std::pair<cv::dnn::Backend, cv::dnn::Target> backend 
        { cv::dnn::Backend::DNN_BACKEND_CUDA, cv::dnn::Target::DNN_TARGET_CUDA };

    if( cpu )
        backend = { cv::dnn::Backend::DNN_BACKEND_OPENCV, cv::dnn::Target::DNN_TARGET_CPU };

    this->dnn_.setup( cfg, weights, backend );

    if( this->dnn_._classes < this->labels_.size() )
    {
        RCLCPP_ERROR(this->get_logger(), "Number of classes in the model is less then number of labels provided %ld vs %ld",
            this->dnn_._classes, this->labels_.size() );
    }
    else if( this->dnn_._classes > this->labels_.size() )
    {
        RCLCPP_WARN(this->get_logger(), "Number of classes in the model is more then number of labels provided %ld vs %ld",
            this->dnn_._classes, this->labels_.size() );
        
        this->labels_.resize( this->dnn_._classes );
    }

    this->sub_ = this->create_subscription<sensor_msgs::msg::Image>("image", 1,
        std::bind(&Bounding::image_callback, this, std::placeholders::_1));

    this->pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("detections", 1);
    this->pubDebug_ = this->create_publisher<sensor_msgs::msg::Image>("~/debug", 1);
}

void Bounding::image_callback(const sensor_msgs::msg::Image &msg)
{   
    try
    {
        //BGR8 image
        this->image_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
    }
    catch( cv_bridge::Exception& e )
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    this->dnn_.pre_process( this->image_ );
    this->dnn_.process();
    this->dnn_.post_process();

    //RCLCPP_INFO(this->get_logger(), "Detections: %ld", this->dnn_.detections().size() );

    // convert detections to ROS message
    vision_msgs::msg::Detection2DArray detectionMsg;
    detectionMsg.header = msg.header;

    detectionMsg.detections.resize( this->dnn_.detections().size() );

    std::transform( this->dnn_.detections().begin(), this->dnn_.detections().end(),
                    detectionMsg.detections.begin(),
                    [&](const Detection& in)
                    {
                        vision_msgs::msg::Detection2D out;
                        out.bbox.center.position.x = in.box.x + (in.box.width*0.5);
                        out.bbox.center.position.y = in.box.y + (in.box.height*0.5);
                        //out.bbox.center.theta = 0.0;
                        out.bbox.size_x = in.box.width;
                        out.bbox.size_y = in.box.height;

                        out.results = { vision_msgs::msg::ObjectHypothesisWithPose() };
                        out.results[0].hypothesis.class_id = this->labels_[in.classId];
                        out.results[0].hypothesis.score = in.confidence;

                        return out;
                    } );

    this->pub_->publish( detectionMsg );

    // publish debug image
    if( this->pubDebug_->get_subscription_count() == 0 )
        return;

    std::for_each( this->dnn_.detections().begin(), this->dnn_.detections().end(),
        [&](const Detection& in)
        {
            cv::rectangle( this->image_, in.box, cv::Scalar(0, 255, 0), 2 );
            cv::putText( this->image_, this->labels_[in.classId], cv::Point(in.box.x, in.box.y-10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2 );

        } );

    cv_bridge::CvImage debugMsg( msg.header, "bgr8", this->image_ );//.toImageMsg();
    sensor_msgs::msg::Image outMsg;
    debugMsg.toImageMsg( outMsg );

    this->pubDebug_->publish( outMsg );
}

} // namespace cnn_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(cnn_ros::Bounding)
