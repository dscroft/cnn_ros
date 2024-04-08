#ifndef VISION_FSAI_EASYDNN_H
#define VISION_FSAI_EASYDNN_H

#include <fstream>
#include <regex>
#include <string>

//! system headers
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/dnn/dnn.hpp>

class Detection
{
public:
    int classId;
    float confidence;
    cv::Rect box;

    Detection() {}

    Detection( int i, float c, cv::Rect b )
        : classId( i ), confidence( c ), box( b ) {}
};

class EasyDNN
{
public:
    size_t _classes;
    cv::Size _size, _frameSize;
    cv::dnn::Net _nn;
    std::vector<cv::String> _outputNames;
    std::vector<cv::Mat> _output;
    std::vector<Detection> _detections;
    std::pair<cv::dnn::Backend, cv::dnn::Target> _backend;

    std::vector<std::vector<float>> _confidences;
    std::vector<std::vector<cv::Rect>> _boxes;

    /**
    * Read the network width and height from the config file.
    *
    * Values are read from the first lines that match
    *   width=?
    * or
    *   height=?
    *
    * @param filename Path for the configutation files.
    * @return Width, height and classes.
    */
    static std::tuple<cv::Size,size_t> read_params_from_cfg( const std::string &filename );

    inline void detection_network_post_process( const float confThreshold );

    inline void region_network_post_process( const float confThreshold );

    static std::vector<std::string> get_outputs_names( const cv::dnn::Net& net ) noexcept
    {
        std::vector<std::string> names;
        if( names.empty() )
        {
            //! Get the indices of the output layers, i.e. the layers with unconnected outputs
            std::vector<int> outLayers = net.getUnconnectedOutLayers();

            //! get the names of all the layers in the network
            std::vector<std::string> layersNames = net.getLayerNames();

            //! Get the names of the output layers in names
            names.resize(outLayers.size());

            for (size_t i = 0; i < outLayers.size(); ++i)
            {
                names[i] = layersNames[outLayers[i] - 1];
            }
        }
        return names;
    }

public:
    EasyDNN();

    /**
    * Sets up the detector.
    *
    * Calculates the arithmetic sum of two integers.
    *
    * @param cfgFilename Network architecture configuration file.
    * @param weightsFilename Pre-trained weights file.
    * @param backend Preferred processing backend to use.
    */
    void setup( const std::string cfgFilename, const std::string weightsFilename,
                std::pair<cv::dnn::Backend, cv::dnn::Target> backend );

    /**
    * Pre-process the image.
    *
    * Consists of creating the temporary storage and loading to the gfx
    *   card as appropriate.
    *
    * @pre Call setup().
    * @param frame Image to process.
    */
    void pre_process( const cv::Mat& frame ) noexcept;

    /**
    * Process the image.
    *
    * Consists of putting through the network.
    *
    * @pre Call pre_process().
    * @param frame Image to process.
    */
    void process();

    /**
    * Process the image.
    *
    * Consists of putting through the network.
    *
    * @pre Call process().
    * @param frame Image to process.
    * @return Reference to the bounding boxes.
    */
    const decltype(_detections)& post_process( const float confThreshold=0.5f,
                                               const float nmsThreshold=0.4f,
                                               std::function<void(cv::Rect&)> transform=nullptr );

    const decltype(_detections)& detections();
};

#endif // VISION_FSAI_EASY_DNN_H
