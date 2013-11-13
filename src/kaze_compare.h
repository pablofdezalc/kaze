
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
void show_input_options_help(void);
int parse_input_options(toptions &options, char *img_name, char *img_name2, char *hfile, char *kfile, int argc, char *argv[] );
void save_matching_image(const cv::Mat& img);
void display_text(cv::Mat &img_rgb, const int& npoints1, const int& npoints2, const int& nmatches, const int& ninliers,
                  const float& ratio, const int& index);

//*************************************************************************************
//*************************************************************************************

#endif // KAZE_COMPARE_H
