//face detection using surf cascade classifier
//autho: Guotian Xie
//version: 1.0

#ifndef _SURF_FACE_DETECTION_H_
#define _SURF_FACE_DETECTION_H_

#include "opencv2\core\core.hpp"
#include "DebugConfig.h"
#include "SurfCascadeModel.h"

using namespace cv;

class SurfFaceDetection
{
public:
	bool DetectMultiScale(Mat &_grayImg, float _scaleFactor, Size winSize);
	bool DetectSingleScale(Mat &_grayImg, float _sacleFactor, Size winSize);

protected:
	SurfCasacdeModel model;
};

#endif