#include "SurfCascadeModel.h"

bool CascadeStage::LoadStage(FileNode *node)
{
	Clear();

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
		LogRegression ww;
		if(!ww.LoadWeak(&(*iter)))
			return false;

		weak.push_back(ww);
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

	FileNode firstNode  = (*file).root();

	if( firstNode.empty() )
	{
		CV_Error(CV_StsParseError, "Wrong format of model file.");
		return false;
	}

	return LoadStage(&firstNode);
}

double CascadeStage::Predict(const Mat &_sumImg, float _scale, bool _sumRes)
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
	Clear();
	if( _file == NULL || !(*_file).isOpened() )
		return false;

	int numStages = 0;
	if( (*_file)["NumStages"].empty() || (*_file)["WeakSequence"].empty() )
	{
		CV_Error(CV_StsParseError, "Wrong format of model file.");
		return false;
	}

	(*_file)["NumStages"] >> numStages;
	stages.reserve(numStages);

	FileNode weakSeq = (*_file)["WeakSequence"];
	int idx = 0;
	for(FileNodeIterator weak = weakSeq.begin(); weak != weakSeq.end(); weak++, idx++)
	{
		CascadeStage stage;

		if(!stage.LoadStage(&(*weak)))
		{
			return false;
		}
		stages.push_back(stage);
	}

	return true;
}

bool SurfCascadeModel::LoadSurfCascadeModelByStages(vector<char *> _stageFiles)
{
	Clear();

	for(int idx=0; idx<_stageFiles.size(); idx++)
	{
		FileStorage file(_stageFiles[idx], FileStorage::READ);
		CascadeStage stage;
		if(!stage.LoadStage(&file))
			return false;

		stages.push_back(stage);

	}
	return true;
}

int SurfCascadeModel::JudgeWindow(Mat &_sumImg, float _scale)
{
	for(int idx = 0; idx<stages.size(); idx++)
	{
		if( stages[idx].Predict(_sumImg, _scale) == 0.0 )
			return 0;
	}
	return 1;
}