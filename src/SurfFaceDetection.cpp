#include "SurfFaceDetection.h"

SurfFaceDetection::SurfFaceDetection()
{
	Init();
}

SurfFaceDetection::SurfFaceDetection(char *modelFileName)
{
	Init();
	model.LoadSurfCascadeModel(modelFileName);
}

void SurfFaceDetection::Init()
{
	maxImgSize = Size(4000,4000);

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

}

bool SurfFaceDetection::CalculateSurfSumImg(const Mat &_grayImg)
{
	CV_Assert(_grayImg.type() == CV_8U && !_grayImg.empty());

	Mat img;
	if(_grayImg.size().width > maxImgSize.width || _grayImg.size().height > maxImgSize.height)
	{
		resize(_grayImg, img, maxImgSize);
	}else
	{
		img = _grayImg;
	}

	Size imgSize = img.size();
	cv::Mat rowImg(img.size(), CV_16SC1);
	cv::Mat colImg(img.size(), CV_16SC1);

	rowFilter->apply(img, rowImg);
	colFilter->apply(img, colImg);

	std::vector<cv::Mat> splitImg(8);

	cv::Mat mergeImg(img.size(), CV_32SC(8));
	Mat sumImg(img.size() + Size(1,1), CV_64FC(8), sumCache.ptr(0,0));

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
}
