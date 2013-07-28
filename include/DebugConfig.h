//header file of debug configure
//author:Guotian Xie
//version:1.0

#ifndef _DEBUG_CONFIG_H_
#define _DEBUG_CONFIG_H_

#include <assert.h>
#include <iostream>
#include "engine.h"
#include "opencv2\highgui\highgui.hpp"
#define MY_DEBUG
#ifdef MY_DEBUG
extern engine *eg;
#endif

#endif