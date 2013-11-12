
/**
 * @file nldiffusion_functions.h
 * @brief Functions for non-linear diffusion applications:
 * 2D Gaussian Derivatives
 * Perona and Malik conductivity equations
 * Perona and Malik evolution
 * @date Dec 27, 2011
 * @author Pablo F. Alcantarilla
 */

#ifndef NLDIFFUSION_FUNCTIONS_H_
#define NLDIFFUSION_FUNCTIONS_H_

//******************************************************************************
//******************************************************************************

// Includes
#include "config.h"

//*************************************************************************************
//*************************************************************************************

// Declaration of functions
void Gaussian_2D_Convolution(const cv::Mat &src, cv::Mat &dst, unsigned int ksize_x, unsigned int ksize_y, float sigma);
void PM_G1(const cv::Mat &Lx, const cv::Mat &Ly, cv::Mat &dst, float k);
void PM_G2(const cv::Mat &Lx, const cv::Mat &Ly, cv::Mat &dst, float k);
void Weickert_Diffusivity(const cv::Mat &Lx, const cv::Mat &Ly, cv::Mat &dst, float k);
float Compute_K_Percentile(const cv::Mat &img, float perc, float gscale, unsigned int nbins, unsigned int ksize_x, unsigned int ksize_y);
void Compute_Scharr_Derivatives(const cv::Mat &src, cv::Mat &dst, int xorder, int yorder, int scale);
void Compute_Deriv_Kernels(cv::OutputArray _kx, cv::OutputArray _ky, int dx, int dy, int scale);
void NLD_Step_Scalar(cv::Mat &Ld2, const cv::Mat &Ld1, const cv::Mat &c, float stepsize);
bool Check_Maximum_Neighbourhood(cv::Mat &img, int dsize, float value, int row, int col, bool same_img );
bool Check_Minimum_Neighbourhood(cv::Mat &img, int dsize, float value, int row, int col, bool same_img );

//*************************************************************************************
//*************************************************************************************

#endif // NLDIFFUSION_FUNCTIONS_H_
