#include "easydnn.h"

std::tuple<cv::Size,size_t> EasyDNN::read_params_from_cfg( const std::string &filename )
{
    //const std::regex reg( "^\\s*(width|height)\\s*=\\s*([0-9]{1,})" );
    const std::regex reg( "(width|height|classes)=([0-9]{1,})" );

    std::tuple<cv::Size,size_t> result { {0, 0}, 0 };

    std::ifstream ifs( filename );
    std::string line;
    while( std::getline( ifs, line ) 
        /*&& ( !result.width || !result.height ||  )*/ )
    {
        std::smatch match;
        if( std::regex_match( line, match, reg ) )
        {
            const int value = std::stoi( match[2] );

            if( match[1] == "width" )
                std::get<0>(result).width = value;
            else if( match[1] == "height")
                std::get<0>(result).height = value;
            else
                std::get<1>(result) = value;
        }
    }

    return result;
}

inline void EasyDNN::detection_network_post_process( const float confThreshold )
{
    // Network produces output blob with a shape 1x1xNx7 where N is a number of
    // detections and an every detection is a vector of values
    // [batchId, classId, confidence, left, top, right, bottom]
    assert( _output.size() > 0 );
    for (size_t k = 0; k < _output.size(); k++)
    {
        float* data = (float*)_output[k].data;
        for (size_t i = 0; i < _output[k].total(); i += 7)
        {
            float confidence = data[i + 2];
            if (confidence > confThreshold)
            {
                int left   = static_cast<int>( data[i + 3] );
                int top    = static_cast<int>( data[i + 4] );
                int right  = static_cast<int>( data[i + 5] );
                int bottom = static_cast<int>( data[i + 6] );
                int width  = right - left + 1;
                int height = bottom - top + 1;
                if (width <= 2 || height <= 2)
                {
                        left   = static_cast<int>( data[i + 3] * _frameSize.width );
                        top    = static_cast<int>( data[i + 4] * _frameSize.height );
                        right  = static_cast<int>( data[i + 5] * _frameSize.width );
                        bottom = static_cast<int>( data[i + 6] * _frameSize.height );
                        width  = right - left + 1;
                        height = bottom - top + 1;
                }

                const size_t classId = (int)(data[i+1]-1);
                _confidences[classId].emplace_back( confidence );
                _boxes[classId].emplace_back( cv::Rect(left, top, width, height) );
            }
        }
    }
}

inline void EasyDNN::region_network_post_process( const float confThreshold )
{
    for (size_t i = 0; i < _output.size(); ++i)
    {
        // Network produces output blob with a shape NxC where N is a number of
        // detected objects and C is a number of classes + 4 where the first 4
        // numbers are [center_x, center_y, width, height]
        const float* data = reinterpret_cast<float*>(_output[i].data);
        for (int j = 0; j < _output[i].rows; ++j, data += _output[i].cols)
        {
            cv::Mat scores = _output[i].row(j).colRange(5, _output[i].cols);
            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

            if( confidence >= confThreshold )
            {
                const int centerX = static_cast<int>( data[0] * _frameSize.width );
                const int centerY = static_cast<int>( data[1] * _frameSize.height );
                const int width   = static_cast<int>( data[2] * _frameSize.width );
                const int height  = static_cast<int>( data[3] * _frameSize.height );
                const int left = centerX - width / 2;
                const int top = centerY - height / 2;

                _confidences[ classIdPoint.x ].emplace_back( confidence );
                _boxes[ classIdPoint.x ].emplace_back( cv::Rect( left, top, width, height ) );
            }
        }
    }
}

EasyDNN::EasyDNN()
{
}

//void setup( int width, int height, std::pair<cv::dnn::Backend, cv::dnn::Target> backend )
void EasyDNN::setup( const std::string cfgFilename, const std::string weightsFilename, 
            std::pair<cv::dnn::Backend, cv::dnn::Target> backend )
{
    const auto result = read_params_from_cfg( cfgFilename );
    _size = std::get<0>(result);
    _classes = std::get<1>(result);

    assert( _size.width > 0 );
    assert( _size.height > 0 );
    assert( _classes > 0 );
    
    const auto backends = cv::dnn::getAvailableBackends();
    _backend = backend;
    if( std::find( backends.begin(), backends.end(), _backend ) == backends.end() )
    {
        assert( false );
    }

    _nn = cv::dnn::readNetFromDarknet( cfgFilename, weightsFilename );

    _outputNames = get_outputs_names( _nn );

    _nn.setPreferableBackend( _backend.first );
    _nn.setPreferableTarget( _backend.second );   
}

void EasyDNN::pre_process( const cv::Mat& frame ) noexcept 
{
    assert( !frame.empty() );
    assert( _size.width > 0 && _size.height > 0 );
    assert( !_nn.empty() );

    static cv::Mat blob;

    _frameSize = frame.size();

    //! Create a 4D blob from a frame.
    cv::dnn::blobFromImage( frame, blob, 1. / 255., _size, cv::Scalar(), true, false );
    _nn.setInput( blob );
}

void EasyDNN::process()
{
    assert( !_nn.empty() );
    assert( !_outputNames.empty() );

    _nn.forward( _output, _outputNames );
}

const decltype(EasyDNN::_detections)& EasyDNN::post_process( const float confThreshold, 
                                            const float nmsThreshold,
                                            std::function<void(cv::Rect&)> transform )
{   
    assert( !_nn.empty() ); 

    static std::vector<int> outLayers = _nn.getUnconnectedOutLayers();
    static std::string outLayerType = _nn.getLayer(outLayers[0])->type;

    assert( outLayerType == "Region" || outLayerType == "DetectionOutput" );

    _confidences.resize( _classes );
    _boxes.resize( _classes );
            
    for( size_t i=0; i<_classes; ++i )
    {
        _confidences[i].clear();
        _boxes[i].clear();
    }

    const bool region = outLayerType == "Region";
    
    if( region )
        region_network_post_process( confThreshold );
    else // DetectionOutput
        detection_network_post_process( confThreshold );
    
    /* apply bounding box transformation function, useful processing an
        input image with funky structure */
    if( transform != nullptr )
        for( auto &c : _boxes )
            for( auto &b : c ) transform( b );

    _detections.clear();

    if( outLayers.size() > 1 || ( region && _backend.first != cv::dnn::Backend::DNN_BACKEND_OPENCV ) )
    {
        std::vector<int> nmsIndices;

        for( size_t c=0; c<_classes; ++c )
        {
            nmsIndices.clear();
            cv::dnn::NMSBoxes(_boxes[c], _confidences[c], confThreshold, nmsThreshold, nmsIndices);
            
            for( const int idx : nmsIndices )
                _detections.emplace_back( c, _confidences[c][idx], _boxes[c][idx] );
        }
    }
    else
    {
        for( size_t c=0; c<_classes; ++c )
        {
            for( size_t i=0; i<_confidences[c].size(); ++i )
                _detections.emplace_back( c, _confidences[c][i], _boxes[c][i] );
        }
    }

    return _detections;
}

const decltype(EasyDNN::_detections)& EasyDNN::detections()
{
    return _detections;
}
