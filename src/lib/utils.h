
/**
 * @file utils.h
 * @brief Some useful functions
 * @date Dec 29, 2011
 * @author Pablo F. Alcantarilla
 */

#ifndef UTILS_H_
#define UTILS_H_

//******************************************************************************
//******************************************************************************

// OPENCV Includes
#include <cv.h>
#include <highgui.h>
#include <cvaux.h>
#include <ml.h>

// System Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <assert.h>
#include <math.h>

//*************************************************************************************
//*************************************************************************************

// Declaration of Functions
void Compute_min_32F(const cv::Mat &src, float &value);
void Compute_max_32F(const cv::Mat &src, float &value);

void Convert_Scale(cv::Mat &src);
void Copy_and_Convert_Scale(const cv::Mat &src, cv::Mat &dst);

void DrawKeyPoints(cv::Mat &img, const std::vector<cv::KeyPoint> &kpts);
int SaveKeyPoints(char *sFileName, const std::vector<cv::KeyPoint> &kpts, const cv::Mat &desc, bool bVerbose);

int fRound(float flt);

void findmatches_nndr(std::vector<cv::KeyPoint> &kpts1, cv::Mat &desc1,std::vector<cv::KeyPoint> &kpts2, cv::Mat &desc2,
                      std::vector<cv::Point2f> &matches, float nndr);

float Compute_Descriptor_Distance(float *d1, float *d2, int dsize);

void Compute_Inliers_RANSAC(const std::vector<cv::Point2f> &matches, std::vector<cv::Point2f> &inliers, float error, bool use_fund);
void Compute_Inliers_Homography(const std::vector<cv::Point2f> &matches, std::vector<cv::Point2f> &inliers, float error, const cv::Mat &H);

void Composite_Image_with_Line(cv::Mat &img1, cv::Mat &img2, cv::Mat &img_com, const std::vector<cv::Point2f> &inliers);
void Composite_Image_with_Line(cv::Mat &img1, cv::Mat &img2, cv::Mat &img_com, const std::vector<cv::Point2f> &inliers, int index);

void Read_Homography(const char *hFile, cv::Mat &H1toN);

//*************************************************************************************
//*************************************************************************************

#endif // UTILS_H_
