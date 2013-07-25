#include "Feature.h"
#include "Global.h"

SurfFeature::SurfFeature()
{
	feature = Rect();
}

SurfFeature::SurfFeature(const Rect &_feature)
{
	SetFeature(_feature);
}

void SurfFeature::SetFeature(const Rect &_feature)
{
	CV_Assert( _feature.x >= 0 && _feature.y >= 0 
		&& _feature.br().x <= MIN_WIND_WIDTH && _feature.br().y <= MIN_WIND_HEIGHT );

	feature = Rect(_feature);
}

Mat SurfFeature::FeatureEvaluate(const Mat &_sumImg, float _scale)
{
	CV_Assert( _scale < 1 );
	CV_Assert( _sumImg.type() == CV_64FC(8) );
	CV_Assert( feature == Rect() );

	int x = (int)(feature.x * _scale + 0.5);
	int y = (int)(feature.y * _scale + 0.5);
	int width = (int)(feature.width * _scale + 0.5);
	int height = (int)(feature.width * _scale + 0.5);

	Rect scaleFst(x,y,width,height);
	if( scaleFst.br().x >= _sumImg.rows || scaleFst.br().y >= _sumImg.cols )
		CV_Error(CV_StsOutOfRange,"Scale feature size is larger than given window size!");

#define SumType Vec<double,8>

	Point mid((int)(scaleFst.x + scaleFst.width * 0.5 + 0.5),
		(int)(scaleFst.y + scaleFst.height * 0.5 + 0.5));

	Vec<double,8> res[4];
	res[0] = _sumImg.at<SumType>(mid) + _sumImg.at<SumType>(scaleFst.tl())
		- _sumImg.at<SumType>(mid.y, scaleFst.x) - _sumImg.at<SumType>(scaleFst.y, mid.x);

	res[1] = _sumImg.at<SumType>(mid.y,scaleFst.br().x) + _sumImg.at<SumType>(scaleFst.y,mid.x)
		- _sumImg.at<SumType>(mid) -  _sumImg.at<SumType>(scaleFst.y,scaleFst.br().x);

	res[2] = _sumImg.at<SumType>(scaleFst.br().y,mid.x) + _sumImg.at<SumType>(mid.y,scaleFst.x)
		- _sumImg.at<SumType>(mid) - _sumImg.at<SumType>(scaleFst.br().y,scaleFst.x);


	res[3] = _sumImg.at<SumType>(scaleFst.br()) + _sumImg.at<SumType>(mid)
		- _sumImg.at<SumType>(mid.y,scaleFst.br().x) - _sumImg.at<SumType>(scaleFst.br().y, mid.x);

	double sumRes = 1e-10;
	Mat result(FEATURE_SIZE, 1, CV_64FC1);

	for(int i = 0; i < 4; i++)
	{
		for(int j = 0; j < 8; j++)
		{
			result.at<double>(i * 8 + j,0) = res[i][j];
			sumRes += res[i][j]*res[i][j];
		}
	}

	CV_Assert(sumRes == 0);
	result = result / cv::sqrt(sumRes);
}