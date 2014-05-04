
/**
 * @file KAZE.h
 * @brief Main program for detecting and computing descriptors in a nonlinear
 * scale space
 * @date Jan 21, 2012
 * @author Pablo F. Alcantarilla
 */

#pragma once

/* ************************************************************************* */
#include "KAZEConfig.h"
#include "nldiffusion_functions.h"
#include "fed.h"
#include "utils.h"

/* ************************************************************************* */
/// KAZE Timing structure
struct KAZETiming {

  KAZETiming() {
    kcontrast = 0.0;
    scale = 0.0;
    derivatives = 0.0;
    detector = 0.0;
    extrema = 0.0;
    subpixel = 0.0;
    descriptor = 0.0;
  }

  double kcontrast;       ///< Contrast factor computation time in ms
  double scale;           ///< Nonlinear scale space computation time in ms
  double derivatives;     ///< Multiscale derivatives computation time in ms
  double detector;        ///< Feature detector computation time in ms
  double extrema;         ///< Scale space extrema computation time in ms
  double subpixel;        ///< Subpixel refinement computation time in ms
  double descriptor;      ///< Descriptors computation time in ms
};

/* ************************************************************************* */
/// KAZE Class Declaration
class KAZE {

private:

  KAZEOptions options_;                ///< Configuration options for AKAZE
  std::vector<TEvolution> evolution_;	/// Vector for nonlinear diffusion evolution

  /// Vector of keypoint vectors for finding extrema in multiple threads
  std::vector<std::vector<cv::KeyPoint> > kpts_par_;

  /// FED parameters
  int ncycles_;                  ///< Number of cycles
  bool reordering_;              ///< Flag for reordering time steps
  std::vector<std::vector<float > > tsteps_;  ///< Vector of FED dynamic time steps
  std::vector<int> nsteps_;      ///< Vector of number of steps per cycle

  // Some auxiliary variables used in the AOS step
  cv::Mat Ltx_, Lty_, px_, py_, ax_, ay_, bx_, by_, qr_, qc_;

  /// Computation times variables in ms
  KAZETiming timing_;

public:

  /// Constructor
  KAZE(KAZEOptions& options);

  /// Destructor
  ~KAZE(void);

  /// Public methods for KAZE interface
  void Allocate_Memory_Evolution(void);
  int Create_Nonlinear_Scale_Space(const cv::Mat& img);
  void Feature_Detection(std::vector<cv::KeyPoint>& kpts);
  void Compute_Descriptors(std::vector<cv::KeyPoint>& kpts, cv::Mat& desc);

  /// Methods for saving the scale space set of images and detector responses
  void Save_Nonlinear_Scale_Space();
  void Save_Detector_Responses();
  void Save_Flow_Responses();

private:

  /// Feature Detection Methods
  void Compute_KContrast(const cv::Mat& img);
  void Compute_Multiscale_Derivatives();
  void Compute_Detector_Response();
  void Determinant_Hessian_Parallel(std::vector<cv::KeyPoint>& kpts);
  void Find_Extremum_Threading(const int level);
  void Do_Subpixel_Refinement(std::vector<cv::KeyPoint>& kpts);
  void Feature_Suppression_Distance(std::vector<cv::KeyPoint>& kpts, const float mdist);

  /// AOS Methods
  void AOS_Step_Scalar(cv::Mat &Ld, const cv::Mat &Ldprev, const cv::Mat &c, const float& stepsize);
  void AOS_Rows(const cv::Mat &Ldprev, const cv::Mat &c, const float& stepsize);
  void AOS_Columns(const cv::Mat &Ldprev, const cv::Mat &c, const float& stepsize);
  void Thomas(const cv::Mat &a, const cv::Mat &b, const cv::Mat &Ld, cv::Mat &x);

  /// Feature Description methods
  void Compute_Main_Orientation(cv::KeyPoint& kpt);

  /// SURF Rectangular Grid 64
  void Get_SURF_Upright_Descriptor_64(const cv::KeyPoint& kpt, float* desc);
  void Get_SURF_Descriptor_64(const cv::KeyPoint& kpt, float* desc);

  /// SURF Rectangular Grid 128
  void Get_SURF_Upright_Descriptor_128(const cv::KeyPoint& kpt, float* desc);
  void Get_SURF_Descriptor_128(const cv::KeyPoint& kpt, float* desc);

  /// M-SURF Rectangular Grid 64
  void Get_MSURF_Upright_Descriptor_64(const cv::KeyPoint& kpt, float* desc);
  void Get_MSURF_Descriptor_64(const cv::KeyPoint& kpt, float* desc);

  /// M-SURF Rectangular Grid 128
  void Get_MSURF_Upright_Descriptor_128(const cv::KeyPoint& kpt, float* desc);
  void Get_MSURF_Descriptor_128(const cv::KeyPoint& kpt, float *desc);

  /// G-SURF Rectangular Grid 64
  void Get_GSURF_Upright_Descriptor_64(const cv::KeyPoint& kpt, float* desc);
  void Get_GSURF_Descriptor_64(const cv::KeyPoint& kpt, float *desc);

  /// Descriptor Mode -> 2 G-SURF Rectangular Grid 128
  void Get_GSURF_Upright_Descriptor_128(const cv::KeyPoint& kpt, float* desc);
  void Get_GSURF_Descriptor_128(const cv::KeyPoint& kpt, float* desc);

public:

  /// Return the computation times
  KAZETiming Get_Computation_Times() const {
    return timing_;
  }
};

/* ************************************************************************* */
/// Inline functions
float getAngle(const float& x, const float& y);
float gaussian(const float& x, const float& y, const float& sig);
void checkDescriptorLimits(int& x, int& y, const int width, const int height);
int fRound(const float& flt);

