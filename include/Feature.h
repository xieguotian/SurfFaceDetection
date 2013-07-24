//surf feature
//autho: Guotian Xie
//version: 1.0

#ifndef _SURF_FEATURE_H_
#define _SURF_FEATURE_H_

#include "opencv2\core\core.hpp"
#include "DebugConfig.h"

using namespace cv;

class SurfFeature
{
public:
	Mat FeatureEvaluate(Mat &_sumImg, float _scale);
	void SetFeature(Rect &_feature);
protected:
	Rect feature;
};

#endif