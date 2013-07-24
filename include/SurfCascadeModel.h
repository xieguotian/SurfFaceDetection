//declaration of SurfCascadeModel
//autho:Guotian Xie
//version:1.0

#ifndef _SURF_CASCADE_MODEL_H_
#define _SURF_CASCADE_MODEL_H_

#include <vector>
#include "DebugConfig.h"
#include "opencv2\core\core.hpp

#include "LogRegression.h"

using namespace cv;
using namespace std;


class SurfCascadeModel
{
public:	//API extern
	bool LoadSurfCascadeModel(char *_fileName);
	int JudgeWindow(Mat &sumImg, float sacle);

protected:
	//API intern

	//data
	vector<CascadeStage>  Stages;
};

class CascadeStage
{
public:
	int Predict(Mat &sumImg, float scale);

protected:
	vector<LogRegression> weak;
	float threshold;
};

#endif
