
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

// Some image matching options
const float MAX_H_ERROR = 2.50;	// Maximum error in pixels to accept an inlier
const float DRATIO = 0.6;		// NNDR Matching value

//*************************************************************************************
//*************************************************************************************

// Declaration of functions
int parse_input_options(KAZEOptions& options, std::string& img_path1, std::string& img_path2,
                        std::string& homography_path, int argc, char *argv[]);

//*************************************************************************************
//*************************************************************************************

#endif // KAZE_COMPARE_H
