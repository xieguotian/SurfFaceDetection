//Loregresion weak classifier
//autho: Guotian Xie
//version: 1.0

#ifndef _LogRegression_H_
#define _LogRegression_H_

#include "opencv2\core\core.hpp"
#include "DebugConfig.h"

#include "Feature.h"

using namespace cv;

class LogRegression
{
public:
	float Predict(const Mat &sumImg);

protected:
	Mat weight;
	Feature feature;
};

#endif