
/**
 * @file kaze_features.h
 * @brief Main program for detecting and computing descriptors in a nonlinear
 * scale space
 * @date Jan 20, 2012
 * @author Pablo F. Alcantarilla
 */

#ifndef KAZE_FEATURES_H_
#define KAZE_FEATURES_H_

//*************************************************************************************
//*************************************************************************************

// Includes
#include "KAZE.h"
#include "config.h"
#include "utils.h"

//*************************************************************************************
//*************************************************************************************

// Declaration of functions
int Parse_Input_Options(toptions &options, char *img_name, char *kfile, int argc, char *argv[] );
void Save_Image_with_Features(cv::Mat img);
void Show_Input_Options_Help(void);

//*************************************************************************************
//*************************************************************************************

#endif // KAZE_FEATURES_H
