
/**
 * @file kaze_match.h
 * @brief Main program for matching two images with KAZE features
 * scale space
 * @date Jan 20, 2012
 * @author Pablo F. Alcantarilla
 */

#ifndef KAZE_MATCH_H_
#define KAZE_MATCH_H_

//*************************************************************************************
//*************************************************************************************

// Includes
#include "KAZE.h"
#include "config.h"
#include "utils.h"

//*************************************************************************************
//*************************************************************************************

// Declaration of functions
void Show_Input_Options_Help(void);
int Parse_Input_Options(toptions &options, char *img_name, char *img_name2, char *hfile, char *kfile, int argc, char *argv[] );
void Save_Matching_Image(cv::Mat img);

//*************************************************************************************
//*************************************************************************************

#endif // KAZE_MATCH_H
