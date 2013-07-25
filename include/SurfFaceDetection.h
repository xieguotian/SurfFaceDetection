//face detection using surf cascade classifier
//autho: Guotian Xie
//version: 1.0

#ifndef _SURF_FACE_DETECTION_H_
#define _SURF_FACE_DETECTION_H_

#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"

#include "DebugConfig.h"
#include "SurfCascadeModel.h"
#include "Global.h"

using namespace cv;

class SurfFaceDetection
{
public:
	SurfFaceDetection();
	SurfFaceDetection(char *modelFileName);
	bool DetectMultiScale(Mat &_grayImg, float _scaleFactor, Size winSize);
	bool DetectSingleScale(Mat &_grayImg, float _sacleFactor, Size winSize);

protected:
	void Init();
	bool CalculateSurfSumImg(const Mat &_grayImg);

	SurfCascadeModel model;
	Ptr<FilterEngine> rowFilter;
	Ptr<FilterEngine> colFilter;
	
	Size maxImgSize;
	Mat sumCache
};

#endif