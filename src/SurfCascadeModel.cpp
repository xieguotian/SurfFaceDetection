#include "SurfCascadeModel.h"

bool CascadeStage::LoadStage(FileNode *node)
{
	if(node == NULL || (*node).empty() || (*node)["Threshold"].empty() 
		|| (*node)["NumWeaks"].empty() || (*node)["WeakSequence"].empty())
	{
		CV_Error(CV_StsParseError, "Stage classifier format wrong in the model file");
		return false;
	}
	(*node)["Threshold"] >> threshold;
	
	int numWeaks = 0;
	(*node)["NumWeaks"] >> numWeaks;
	weak.reserve(numWeaks);

	FileNode weaksNode = (*node)["WeakSequence"];
	int idx = 0;
	for(FileNodeIterator iter = weaksNode.begin(); iter != weaksNode.end(); iter++, idx++)
	{
		weak[idx].LoadWeak(&(*iter));
	}

	return true;
}

bool CascadeStage::LoadStage(FileStorage *file)
{
	if( file == NULL || !(*file).isOpened() )
	{
		CV_Error(CV_StsNullPtr, "File is NULL or is not opened.");
		return false;
	}

	FileNode firstNode  = (*file).getFirstTopLevelNode();
	if( firstNode.empty() )
	{
		CV_Error(CV_StsParseError, "Wrong format of model file.");
		return false;
	}

	return LoadStage(&firstNode);
}

double CascadeStage::Predict(Mat &_sumImg, float _scale, bool _sumRes)
{
	double sum = 0.0;

	for(int idx=0; idx<weak.size(); idx++)
	{
		sum += weak[idx].Predict(_sumImg, _scale);
	}

	if(!_sumRes)
		sum = (sum<threshold? 0:1);

	return sum;
}

bool SurfCascadeModel::LoadSurfCascadeModel(char *_fileName)
{
	FileStorage file(_fileName,FileStorage::READ);

	return LoadSurfCascadeModel(&file);

}

bool SurfCascadeModel::LoadSurfCascadeModel(FileStorage *_file)
{
	if( _file == NULL || !(*_file).isOpened() )
		return false;

	int numStages = 0;
	if( (*_file)["NumStages"].empty() || (*_file)["WeakSequence"].empty() )
	{
		CV_Error(CV_StsParseError, "Wrong format of model file.");
		return false;
	}
}

int SurfCascadeModel::JudgeWindow(Mat &_sumImg, float _scale)
{
	for(int idx = 0; idx<Stages.size(); idx++)
	{
		if( Stages[idx].Predict(_sumImg, _scale) == 0.0 )
			return 0;
	}
	return 1;
}