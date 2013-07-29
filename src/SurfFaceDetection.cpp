#include "SurfFaceDetection.h"
#include "opencv2\objdetect\objdetect.hpp"

SurfFaceDetection::SurfFaceDetection()
{
	Init();
}

SurfFaceDetection::SurfFaceDetection(char *modelFileName)
{
	Init();
	model.LoadSurfCascadeModel(modelFileName);
}

SurfFaceDetection::SurfFaceDetection(vector<char *> _stageFiles)
{
	Init();
	model.LoadSurfCascadeModelByStages(_stageFiles);
}
void SurfFaceDetection::Init()
{
	maxImgSize = Size(2000,2000);

	Mat rowKernel(1, 3, CV_32F);
	rowKernel.at<float>(0,0) = -1;
	rowKernel.at<float>(0,1) = 0;
	rowKernel.at<float>(0,2) = 1;

	Mat colKernel(3, 1, CV_32F);
	colKernel.at<float>(0,0) = -1;
	colKernel.at<float>(1,0) = 0;
	colKernel.at<float>(2,0) = 1;

	rowFilter = createLinearFilter(CV_8UC1, CV_16SC1, rowKernel, Point(-1,-1),
		0.0, BORDER_REFLECT, BORDER_REFLECT);
	colFilter = createLinearFilter(CV_8UC1, CV_16SC1, colKernel, Point(-1,-1),
		0.0, BORDER_REFLECT, BORDER_REFLECT);

	sumCache = Mat(maxImgSize + Size(1,1), CV_64FC(8));
	srcScale = 1.0;
	//img.reserve(maxImgSize);
}

bool SurfFaceDetection::CalculateSurfSumImg(const Mat &_grayImg)
{
	CV_Assert(_grayImg.type() == CV_8U && !_grayImg.empty());
	imgOrg = _grayImg;
	Mat img;
	if(_grayImg.size().width > maxImgSize.width || _grayImg.size().height > maxImgSize.height)
	{
		double scale = cv::min( (double)maxImgSize.height / _grayImg.size().height, (double)maxImgSize.width / _grayImg.size().width);
		srcScale = scale;
		Size reSize((int)(_grayImg.size().width * scale + 0.5), (int)(_grayImg.size().height * scale + 0.5));
		resize(_grayImg, img, reSize);
	}else
	{
		img = _grayImg;
		srcScale = 1.0;
	}

	imgSize = img.size();

	cv::Mat rowImg(img.size(), CV_16SC1);
	cv::Mat colImg(img.size(), CV_16SC1);

	rowFilter->apply(img, rowImg);
	colFilter->apply(img, colImg);

	std::vector<cv::Mat> splitImg(8);

	cv::Mat mergeImg(img.size(), CV_32SC(8));
	//Mat sumImg(img.size() + Size(1,1), CV_64FC(8), sumCache.ptr(0,0));
	Mat sumImg(sumCache,Rect(0,0,imgSize.width+1, imgSize.height+1));

	cv::Mat dyM = cv::Mat_<short>((colImg < 0) / 255);
	cv::Mat dxM = cv::Mat_<short>((rowImg < 0) / 255);

	splitImg[0] = rowImg.mul(dyM);    //dx when dy<0
	splitImg[1] = rowImg.mul(1 - dyM); //dx when dy>=0
	splitImg[2] = cv::abs(rowImg).mul(dyM); //abs(dx) when dy<0
	splitImg[3] = cv::abs(rowImg).mul(1-dyM); //abs(dx) when dy>=0

	splitImg[4] = colImg.mul(dxM);   //dy when dx<0
	splitImg[5] = colImg.mul(1 - dxM); //dy when dx>=0
	splitImg[6] = cv::abs(colImg).mul(dxM); //abs(dy) when dx<0
	splitImg[7] = cv::abs(colImg).mul(1 - dxM); //abs(dy) when dx>=0

	cv::merge(splitImg, mergeImg);
	mergeImg = cv::Mat_<cv::Vec<float,8>>(mergeImg);

	
	cv::integral(mergeImg, sumImg);
	
	return true;
}

bool SurfFaceDetection::DetectMultiScale(const Mat &_grayImg, float _scaleFactor, 
	float _stepFactor, Size _winSize, vector<Rect> &_faceList, bool _isScore, vector<double> *_scoreList)
{
	if(!CalculateSurfSumImg(_grayImg))
		return false;

	Size actualSize(_winSize);
	//Size imgSize = _grayImg.size();

	float scale = 1.0;
	int step = cv::min((int)(actualSize.height * _stepFactor + 0.5),
		(int)(actualSize.width * _stepFactor + 0.5));

	while( actualSize.width <= imgSize.width && actualSize.height <= imgSize.height )
	{
		DetectSingleScale(actualSize, scale, step, _faceList,_scoreList);

		scale = scale * _scaleFactor;

		int height = (int)(scale * _winSize.height + 0.5);
		int width = (int)(scale * _winSize.width + 0.5);
		actualSize = Size(width, height);

		step = cv::min((int)(actualSize.height * _stepFactor + 0.5),
		(int)(actualSize.width * _stepFactor + 0.5));
	}
	vector<int> weights(_faceList.size(),2);
	//groupRectangles(_faceList,1,0.2,&weights,_scoreList);
	groupRectangles(_faceList,weights, *_scoreList,1);
	if( srcScale != 1.0 )
	{
		for(int idx=0; idx<_faceList.size(); idx++)
		{
			_faceList[idx].x = (int)(_faceList[idx].x * srcScale + 0.5);
			_faceList[idx].y = (int)(_faceList[idx].y * srcScale + 0.5);
			_faceList[idx].width = (int)(_faceList[idx].width * srcScale + 0.5);
			_faceList[idx].height = (int)(_faceList[idx].height * srcScale + 0.5);
		}
	}
	return true;
}

bool SurfFaceDetection::DetectSingleScale(const Mat &_grayImg, float _scale,
	float _stepFactor, Size _winSize, vector<Rect> &_faceList)
{
	if(!CalculateSurfSumImg(_grayImg))
		return false;

	int height = (int)(_scale * _winSize.height + 0.5);
	int width = (int)(_scale * _winSize.width + 0.5);

	Size actualSize(width,height);
	int step = cv::min((int)(actualSize.height * _stepFactor + 0.5),
		(int)(actualSize.width * _stepFactor + 0.5));
	DetectSingleScale(actualSize, _scale, step, _faceList);
	groupRectangles(_faceList, 2);

	return true;
}

bool SurfFaceDetection::DetectSingleScale(Size _winSize, float _scaleFactor, 
	int _step, vector<Rect> &_faceList, vector<double> *_scoreList)
{
	Size sumWinSize = _winSize + Size(1,1);
	
	for(int offsetY = 0; offsetY + _winSize.height <= imgSize.height; offsetY += _step)
	{
		for(int offsetX = 0; offsetX + _winSize.width <= imgSize.width; offsetX += _step)
		{
			double score = 0.0;
			Rect roi = Rect(offsetX,offsetY,sumWinSize.width,sumWinSize.height);
			Mat sum(sumCache,roi);

			bool flag = false;
			if( model.JudgeWindow(sum, _scaleFactor, score) == 1 )
			{
				flag = true;
				if(_scoreList!=NULL)
					_scoreList->push_back(score);
				Rect rect(offsetX, offsetY, _winSize.width, _winSize.height);
				_faceList.push_back(rect);
			}
#undef MY_DEBUG
#ifdef MY_DEBUG
			Mat tmp;
			cvtColor(imgOrg,tmp,CV_GRAY2RGB);
			if(!flag)
				rectangle(tmp,roi,Scalar(0,255,0));
			else
				rectangle(tmp,roi,Scalar(255,0,0));
			imshow("tmpShow",tmp);
			waitKey(_step*2);
#endif
		}
	}
	return true;
}