#include <iostream>
#include <string>
#include <vector>
#include <exception>
#include <stdexcept>
#include <sstream>
#include <filesystem>
#include <algorithm> //std::all_of
#include <cctype> //std::isdigit
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;
using namespace cv::dnn;

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const float confidenceThreshold = 0.7;

#if 1
#   define USE_MODEL_FROM_2020_MAY_25
#endif

#ifdef USE_MODEL_FROM_2020_MAY_25
const cv::Scalar meanVal(0, 0, 0);
#else
const cv::Scalar meanVal(104.0, 177.0, 123.0);
#endif

#if 0
#define CAFFE
#endif

const std::string caffeConfigFile = "./models/deploy.prototxt";
const std::string caffeWeightFile = "./models/res10_300x300_ssd_iter_140000_fp16.caffemodel";

#ifdef USE_MODEL_FROM_2020_MAY_25
const std::string tensorflowConfigFile = "./models/2020-05-25/ssd300_thermal_face_detection_v1.pbtxt";
const std::string tensorflowWeightFile = "./models/2020-05-25/ssd300_thermal_face_detection_v1.pb";
#else
const std::string tensorflowConfigFile = "./models/opencv_face_detector.pbtxt";
const std::string tensorflowWeightFile = "./models/opencv_face_detector_uint8.pb";
#endif

void detectFaceOpenCVDNN(Net net, Mat &frameOpenCVDNN)
{
    int frameHeight = frameOpenCVDNN.rows;
    int frameWidth = frameOpenCVDNN.cols;
#ifdef CAFFE
        cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, false, false);
#else
        cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, true, false);
#endif

#ifdef USE_MODEL_FROM_2020_MAY_25
    net.setInput(inputBlob, "image_tensor");
#else
    net.setInput(inputBlob, "data");
#endif
    
    cv::Mat detection = net.forward("detection_out");

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    for(int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if(confidence > confidenceThreshold)
        {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

            cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),2, 4);
        }
    }

}

class BadCommandLineArgs : public invalid_argument
{
public:
  BadCommandLineArgs( const string& msg )
    : invalid_argument{ msg }
  {}
};

bool isAllDigits( const string& s )
{
  return all_of(
    begin( s ),
    end( s ),
    []( const auto c ) {
      return isdigit( c );
    }
  );
}

bool isANonNegativeInteger( const string& s )
{
  return isAllDigits( s );
}

bool isAVidCapDeviceId( const string& s )
{
  return isANonNegativeInteger( s );
}

bool isNotAVidCapDeviceId( const string& s )
{
  return ! isAVidCapDeviceId( s );
}

void validateArgs( const int argc, const char** argv )
{
  if ( argc == 1 ) return;

  if ( argc > 2 )
  {
    ostringstream ss;
    ss << "Too many command line arguments.";
    throw BadCommandLineArgs{ ss.str() };
  }
  if ( isNotAVidCapDeviceId( argv[ 1 ] ) )
  {
    ostringstream ss;
    ss << argv[ 1 ] << " isn't a valid video capture device ID.";
    throw BadCommandLineArgs{ ss.str() };
  }
}

VideoCapture makeVideoCaptureSource( const int argc, const char** argv )
{
  VideoCapture source;

  const auto deviceId = argc == 1 ? 0 : stoi( argv[1] );

  const auto successfullyOpened = source.open( deviceId );

  if ( ! successfullyOpened )
  {
      ostringstream ss;
      ss << "failed to open device with ID " << deviceId << ".";
      throw BadCommandLineArgs{ ss.str() };
  }
  return source;
}

namespace
{
  string getCvMatDepthAndDataTypeAsString( const int cvMatType )
  {
    //https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv/17820615#17820615
    const uchar depthAndDataType = cvMatType & CV_MAT_DEPTH_MASK;

    switch ( depthAndDataType )
    {
      case CV_8U:  return "8U";
      case CV_8S:  return "8S";
      case CV_16U: return "16U";
      case CV_16S: return "16S";
      case CV_32S: return "32S";
      case CV_32F: return "32F";
      case CV_64F: return "64F";
      default:     return "User";
    }
  }

  int getNumChannels( const int cvMatType )
  {
        //https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv/17820615#17820615
    return 1 + ( cvMatType >> CV_CN_SHIFT );
  }
} //namespace

string cvMatTypeToString( const int cvMatType )
{
    const auto depthAndDataTypeAsString = getCvMatDepthAndDataTypeAsString( cvMatType );
    const auto numChannels = getNumChannels( cvMatType );
    
    return depthAndDataTypeAsString + "C" + std::to_string( numChannels );
}

void main_( const int argc, const char** argv )
{
  validateArgs( argc, argv );
#ifdef CAFFE
  Net net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
#else
  Net net = cv::dnn::readNetFromTensorflow(tensorflowWeightFile, tensorflowConfigFile);
#endif

  net.setPreferableTarget( cv::dnn::DNN_TARGET_OPENCL );

  auto source = makeVideoCaptureSource( argc, argv );
  
  Mat frame;

  double tt_opencvDNN = 0;
  double fpsOpencvDNN = 0;
  bool matTypeWasPrintedOnce = false;
  while(1)
  {
      source >> frame;
      if(frame.empty())
          break;

      if ( ! matTypeWasPrintedOnce )
      {
        cout << "cv::Mat type = " << cvMatTypeToString( frame.type() ) << "\n";
        matTypeWasPrintedOnce = true;
      }
      double t = cv::getTickCount();
      detectFaceOpenCVDNN ( net, frame );
      tt_opencvDNN = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
      fpsOpencvDNN = 1/tt_opencvDNN;
      putText(frame, format("OpenCV DNN ; FPS = %.2f",fpsOpencvDNN), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.4, Scalar(0, 0, 255), 4);
      imshow( "OpenCV - DNN Face Detection", frame );
      int k = waitKey(5);
      if(k == 27)
      {
        destroyAllWindows();
        break;
      }
    }
}

string getFilenamePartOfPath( const string& somePath )
{
  return std::filesystem::path{ somePath }.filename().string();
}

void usage( const string& fullPathToExecutable )
{
  const auto executableName = getFilenamePartOfPath( fullPathToExecutable );
  constexpr auto dotSlash = "./";
  cout << "Usage:\n";
  cout << "\n";
  cout << "  " << dotSlash << executableName << " [device_ID]\n";
  cout << "\n";
  cout << "    where device_ID is the numeric device ID of the video capturing device to use.  This arg is optional.  If no arg is provided, device ID of 0 is used.\n";
  cout << "\n";
  cout << "Examples:\n";
  cout << "\n";
  cout << " Use the first camera implicitly:\n";
  cout << "\n";
  cout << "  " << dotSlash << executableName << "\n";
  cout << "\n";
  cout << " Use the first camera explicitly (device IDs are 0-indexed):\n";
  cout << "\n";
  cout << "  " << dotSlash << executableName << " 0\n";
  cout << "\n";
  cout << " Use the second camera (device IDs are 0-indexed):\n";
  cout << "\n";
  cout << "  " << dotSlash << executableName << " 1\n";
  cout << "\n";
  cout << " Use the third camera (device IDs are 0-indexed):\n";
  cout << "\n";
  cout << "  " << dotSlash << executableName << " 2\n";
  cout << "\n";
}

int main( int argc, const char** argv )
{
  try
  {
      main_( argc, argv );
      cout << "\nEnd of program.\n\n";
      return 0;
  }
  catch ( const BadCommandLineArgs& e )
  {
      cerr << "\n";
      cerr << __func__ << ": error: " << e.what() << "\n";
      cerr << "\n";
      usage( argv[ 0 ] );
  }
  catch ( const exception& e )
  {
      cerr << "\n";
      cerr << __func__ << ": error: " << e.what() << "\n";
  }
  catch ( ... )
  {
      cerr << "\n";
      cerr << __func__ << ": error, unknown exception caught.\n";
  }
  return -1;
}
