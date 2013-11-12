
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
void gaussian_2D_convolution(const cv::Mat& src, cv::Mat& dst, const size_t& ksize_x,
                             const size_t& ksize_y, const float& sigma);
void pm_g1(const cv::Mat& Lx, const cv::Mat& Ly, cv::Mat& dst, const float& k);
void pm_g2(const cv::Mat &Lx, const cv::Mat& Ly, cv::Mat& dst, const float& k);
void weickert_diffusivity(const cv::Mat& Lx, const cv::Mat& Ly, cv::Mat& dst, const float& k);
float compute_k_percentile(const cv::Mat& img, const float& perc, const float& gscale,
                           const size_t& nbins, const size_t& ksize_x, const size_t& ksize_y);
void compute_scharr_derivatives(const cv::Mat& src, cv::Mat& dst,
                                const size_t& xorder, const size_t& yorder, const size_t& scale);
void compute_derivative_kernels(cv::OutputArray _kx, cv::OutputArray _ky,
                                const size_t& dx, const size_t& dy, const size_t& scale_);
void nld_step_scalar(cv::Mat& Ld, const cv::Mat& c, cv::Mat& Lstep, const float& stepsize);
bool check_maximum_neighbourhood(const cv::Mat& img, const int& dsize, const float& value,
                                 const int& row, const int& col, const bool& same_img);

//*************************************************************************************
//*************************************************************************************

#endif // NLDIFFUSION_FUNCTIONS_H_
