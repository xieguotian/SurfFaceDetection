//declaration of SurfCascadeModel
//autho:Guotian Xie
//version:1.0

#ifndef _SURF_CASCADE_MODEL_H_
#define _SURF_CASCADE_MODEL_H_

#include <vector>
#include "DebugConfig.h"
#include "opencv2\core\core.hpp"

#include "LogRegression.h"
#include "Global.h"

using namespace cv;
using namespace std;

class CascadeStage
{
public:
	int Predict(Mat &_sumImg, float _scale, bool _sumRes = false);

	bool LoadStage(FileNode *node);
	bool LoadStage(FileStorage *file);

protected:
	vector<LogRegression> weak;
	float threshold;
};

class SurfCascadeModel
{
public:	//API extern
	bool LoadSurfCascadeModel(char *_fileName);
	bool LoadSurfCascadeModel(FileStorage *_file);
	int JudgeWindow(Mat &_sumImg, float _sacle);

protected:
	//API intern

	//data
	vector<CascadeStage>  Stages;
};

#endif
