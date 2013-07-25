//Loregresion weak classifier
//autho: Guotian Xie
//version: 1.0

#ifndef _LogRegression_H_
#define _LogRegression_H_

#include "opencv2\core\core.hpp"
#include "DebugConfig.h"

#include "Feature.h"
#include "Global.h"

using namespace cv;

class LogRegression
{
public:
	LogRegression();
	LogRegression(Mat &_weight, SurfFeature &_feature);
	LogRegression(Mat &_weight, Rect &_feature);

	double Predict(const Mat &_sumImg, float _scale);
	void SetWeight(Mat &_weight);
	void SetFeature(SurfFeature &_feature);
	void SetFeature(Rect &_feature);

	bool LoadWeak(FileNode *node);
protected:
	Mat weight;
	SurfFeature feature;
};

#endif