#include "DebugConfig.h"
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <iostream>
#include "SurfFaceDetection.h"

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
	vector<char *> stageFiles;

	for(int i=0; i<5; i++)
	{
		char *file = new char[_MAX_PATH];
		sprintf(file, "../model/stage%d.xml", i);
		stageFiles.push_back(file);
	}

	SurfFaceDetection detection(stageFiles);
	
	Mat img = imread("../data/test.jpg",0);
	vector<Rect> faces;
	detection.DetectMultiScale(img,1.2,10,Size(40,40),faces);

	return 0;
}