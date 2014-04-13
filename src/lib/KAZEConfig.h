
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
// Some defines
#define NMAX_CHAR 400

const float DEFAULT_MIN_DETECTOR_THRESHOLD = 0.00001;     // Minimum Detector response threshold to accept point
const int DEFAULT_DESCRIPTOR_MODE = 1; // Descriptor Mode 0->SURF, 1->M-SURF

const bool DEFAULT_UPRIGHT = false;  // Upright descriptors, not invariant to rotation
const bool DEFAULT_EXTENDED = false; // Extended descriptor, dimension 128
const bool DEFAULT_SAVE_SCALE_SPACE = false; // For saving the scale space images
const bool DEFAULT_VERBOSITY = false; // Verbosity level (0->no verbosity)
const bool DEFAULT_SHOW_RESULTS = true; // For showing the output image with the detected features plus some ratios
const bool DEFAULT_SAVE_KEYPOINTS = false; // For saving the list of keypoints

// Some important configuration variables
const float DEFAULT_SIGMA_SMOOTHING_DERIVATIVES = 1.0;
const float DEFAULT_KCONTRAST = .01;
const float KCONTRAST_PERCENTILE = 0.7;
const int KCONTRAST_NBINS = 300;
const bool COMPUTE_KCONTRAST = true;
const int DEFAULT_DIFFUSIVITY_TYPE = 1;  // 0 -> PM G1, 1 -> PM G2, 2 -> Weickert
const bool USE_CLIPPING_NORMALIZATION = false;
const float CLIPPING_NORMALIZATION_RATIO = 1.6;
const int CLIPPING_NORMALIZATION_NITER = 5;

/* ************************************************************************* */
/// KAZE Descriptor Type
enum DESCRIPTOR_TYPE {
  MSURF_UPRIGHT = 0,          ///< Not rotation invariant descriptor, M-SURF grid, length 64
  MSURF = 1,                  ///< Rotation invariant descriptor, M-SURF grid, length 64
  MSURF_EXTENDED_UPRIGHT = 2, ///< Not rotation invariant descriptor, M-SURF grid, length 128
  MSURF_EXTENDED = 3,         ///< Rotation invariant descriptor, M-SURF grid, length 128
  GSURF_UPRIGHT = 4,          ///< Not rotation invariant descriptor, G-SURF grid, length 64
  GSURF = 5,                  ///< Rotation invariant descriptor, G-SURF grid, length 64
  GSURF_EXTENDED_UPRIGHT = 6, ///< Not rotation invariant descriptor, G-SURF grid, length 128
  GSURF_EXTENDED = 7          ///< Rotation invariant descriptor, G-SURF grid, length 128
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
    upright = false;
    extended = false;
    descriptor = MSURF;
    diffusivity = PM_G2;
    sderivatives = 1.0;

    kcontrast = 0.001f;
    kcontrast_percentile = 0.7f;
    kcontrast_nbins = 300;

    save_scale_space = DEFAULT_SAVE_SCALE_SPACE;
    save_keypoints = DEFAULT_SAVE_KEYPOINTS;
    verbosity = DEFAULT_VERBOSITY;
    show_results = DEFAULT_SHOW_RESULTS;
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
  bool upright;
  bool extended;
  int descriptor;
  bool save_scale_space;
  bool save_keypoints;
  bool verbosity;
  bool show_results;
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



