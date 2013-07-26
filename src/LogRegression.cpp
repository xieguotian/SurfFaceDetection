#include "LogRegression.h"
LogRegression::LogRegression()
{
	weight = Mat();
	feature = SurfFeature();
}

LogRegression::LogRegression(Mat &_weight, SurfFeature &_feature)
{
	SetWeight(_weight);
	SetFeature(_feature);
}
LogRegression::LogRegression(Mat &_weight, Rect &_feature)
{
	SetWeight(_weight);
	SetFeature(_feature);
}

//todo:
void LogRegression::SetWeight(Mat &_weight)
{
	CV_Assert(_weight.cols == 1 && _weight.rows == FEATURE_SIZE && _weight.type() == CV_64FC1);

	weight = _weight;
}

void LogRegression::SetFeature(SurfFeature &_feature)
{
	feature = _feature;
}

void LogRegression::SetFeature(Rect &_feature)
{
	feature.SetFeature(_feature);
}

double LogRegression::Predict(const Mat &_sum, float _scale)
{
	Mat fstVal = feature.FeatureEvaluate(_sum, _scale);
	
	double y  = 0;
	for(int idx=0; idx<fstVal.rows; idx++)
		y += weight.at<double>(idx,0) * fstVal.at<double>(idx,0);

	y = 1 / (1 + cv::exp(-y));

	return y;
}

bool LogRegression::LoadWeak(FileNode *node)
{
	string str = (*node).name();
	if(node == NULL || (*node).empty() ||
		(*node)["Feature"].empty() || (*node)["LogregressionWeight"].empty() )
	{
		CV_Error(CV_StsParseError, "Weak classifier format wrong in the model file");
		return false;
	}

	
	FileNode fstNode = (*node)["Feature"] ;
	feature.LoadFeature(&fstNode);

	Mat wei(FEATURE_SIZE, 1, CV_32FC1);
	(*node)["LogregressionWeight"] >> wei;
	weight = Mat_<double>(wei);

	return true;
}