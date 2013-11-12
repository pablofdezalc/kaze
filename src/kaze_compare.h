
/**
 * @file kaze_compare.h
 * @brief Simple image matching program that compares KAZE against other
 * features implemented in OpenCV such as SIFT and SURF
 * @date Oct 24, 2012
 * @author Pablo F. Alcantarilla
 */

#ifndef KAZE_COMPARE_H
#define KAZE_COMPARE_H

//*************************************************************************************
//*************************************************************************************

// Includes
#include "KAZE.h"
#include "config.h"
#include "utils.h"

// OpenCV Features Includes
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

//*************************************************************************************
//*************************************************************************************

// Declaration of functions
void Show_Input_Options_Help(void);
int Parse_Input_Options(toptions &options, char *img_name, char *img_name2, char *hfile, char *kfile, int argc, char *argv[] );
void Save_Matching_Image(cv::Mat img);
void Display_Text(cv::Mat &img_rgb, int npoints1, int npoints2, int nmatches, int ninliers, float ratio, int index);

//*************************************************************************************
//*************************************************************************************

#endif // KAZE_COMPARE_H
