
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
void show_input_options_help(void);
int parse_input_options(toptions &options, char *img_name, char *img_name2, char *hfile, char *kfile, int argc, char *argv[] );
void save_matching_image(const cv::Mat& img);

//*************************************************************************************
//*************************************************************************************

#endif // KAZE_MATCH_H
