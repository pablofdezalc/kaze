
/**
 * @file KAZEConfig.h
 * @brief KAZE configuration file
 * @date Apr 13, 2014
 * @author Pablo F. Alcantarilla
 */

#pragma once

/* ************************************************************************* */
// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

// OpenMP
#ifdef _OPENMP
# include <omp.h>
#endif

// System Includes
#include <string>
#include <vector>
#include <cmath>

/* ************************************************************************* */
/// KAZE Descriptor Type
enum DESCRIPTOR_TYPE {
  SURF_UPRIGHT = 0,            ///< Not rotation invariant descriptor, SURF grid, length 64
  SURF = 1,                    ///< Rotation invariant descriptor, SURF grid, length 64
  SURF_EXTENDED_UPRIGHT = 2,   ///< Not rotation invariant descriptor, SURF grid, length 128
  SURF_EXTENDED = 3,           ///< Rotation invariant descriptor, SURF grid, length 128
  MSURF_UPRIGHT = 4,           ///< Not rotation invariant descriptor, M-SURF grid, length 64
  MSURF = 5,                   ///< Rotation invariant descriptor, M-SURF grid, length 64
  MSURF_EXTENDED_UPRIGHT = 6,  ///< Not rotation invariant descriptor, M-SURF grid, length 128
  MSURF_EXTENDED = 7,          ///< Rotation invariant descriptor, M-SURF grid, length 128
  GSURF_UPRIGHT = 8,           ///< Not rotation invariant descriptor, G-SURF grid, length 64
  GSURF = 9,                   ///< Rotation invariant descriptor, G-SURF grid, length 64
  GSURF_EXTENDED_UPRIGHT = 10, ///< Not rotation invariant descriptor, G-SURF grid, length 128
  GSURF_EXTENDED = 11          ///< Rotation invariant descriptor, G-SURF grid, length 128
};

/* ************************************************************************* */
/// KAZE Diffusivities
enum DIFFUSIVITY_TYPE {
  PM_G1 = 0,
  PM_G2 = 1,
  WEICKERT = 2,
  CHARBONNIER = 3
};

/* ************************************************************************* */
/// KAZE configuration options struct
struct KAZEOptions {

  KAZEOptions() {
    soffset = 1.60;
    omax = 4;
    nsublevels = 4;
    dthreshold = 0.001;
    use_fed = true;
    descriptor = MSURF;
    diffusivity = PM_G2;
    sderivatives = 1.0;
    kcontrast = 0.001f;
    kcontrast_percentile = 0.7f;
    kcontrast_nbins = 300;
    save_scale_space = false;
    save_keypoints = false;
    verbosity = false;
  }

  float soffset;
  int omax;
  int nsublevels;
  int img_width;
  int img_height;

  DIFFUSIVITY_TYPE diffusivity;   ///< Diffusivity type
  float kcontrast;                ///< The contrast factor parameter
  float kcontrast_percentile;     ///< Percentile level for the contrast factor
  size_t kcontrast_nbins;         ///< Number of bins for the contrast factor histogram
  float sderivatives;
  float dthreshold;
  bool use_fed;

  DESCRIPTOR_TYPE descriptor;

  bool save_scale_space;
  bool save_keypoints;
  bool verbosity;
};

/* ************************************************************************* */
/// KAZE nonlinear diffusion filtering evolution
struct TEvolution {
  cv::Mat Lx, Ly;	// First order spatial derivatives
  cv::Mat Lxx, Lxy, Lyy;	// Second order spatial derivatives
  cv::Mat Lflow;	// Diffusivity image
  cv::Mat Lt;	// Evolution image
  cv::Mat Lsmooth; // Smoothed image
  cv::Mat Lstep; // Evolution step update
  cv::Mat Ldet; // Detector response
  float etime;	// Evolution time
  float esigma;	// Evolution sigma. For linear diffusion t = sigma^2 / 2
  float octave;	// Image octave
  float sublevel;	// Image sublevel in each octave
  int sigma_size;	// Integer esigma. For computing the feature detector responses
};

