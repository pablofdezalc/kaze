
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

// Some image matching options
const bool COMPUTE_HOMOGRAPHY = false;	// 0->Use ground truth homography, 1->Estimate homography with RANSAC
const float MAX_H_ERROR = 5.0;	// Maximum error in pixels to accept an inlier
const float DRATIO = .60;		// NNDR Matching value

//*************************************************************************************
//*************************************************************************************

// Declaration of functions
int parse_input_options(KAZEOptions& options, std::string& img_path1, std::string& img_path2,
                        std::string& homography_path, int argc, char *argv[]);
void save_matching_image(const cv::Mat& img);

//*************************************************************************************
//*************************************************************************************

#endif // KAZE_MATCH_H
