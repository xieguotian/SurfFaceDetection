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
	SurfFaceDetection(vector<char *> _stageFiles);

	bool DetectMultiScale(const Mat &_grayImg, float _scaleFactor, 
		float _stepFactor, Size _winSize, vector<Rect> &_faceList,bool _isScore = false, vector<double> *_scoreList = NULL);


	bool DetectSingleScale(const Mat &_grayImg, float _sacle,
		 float _stepFactor, Size _winSize, vector<Rect> &_faceList);

protected:
	void Init();
	bool CalculateSurfSumImg(const Mat &_grayImg);
	bool DetectSingleScale(Size _winSize,float _scaleFactor, int _step,
		 vector<Rect> &_faceList, vector<double> *_scoreList = NULL);

	SurfCascadeModel model;
	Ptr<FilterEngine> rowFilter;
	Ptr<FilterEngine> colFilter;
	
	Size maxImgSize;
	Mat sumCache;
	Size imgSize;
	Mat imgOrg;
	double srcScale;
};

#endif