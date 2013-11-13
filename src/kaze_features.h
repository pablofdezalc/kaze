
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
int parse_input_options(toptions &options, char *img_name, char *kfile, int argc, char *argv[]);
void save_image_with_features(const cv::Mat& img);
void show_input_options_help(void);

//*************************************************************************************
//*************************************************************************************

#endif // KAZE_FEATURES_H
