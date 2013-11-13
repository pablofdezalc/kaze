
//=============================================================================
//
// kaze_compare.cpp
// Author: Pablo F. Alcantarilla
// Institution: University d'Auvergne
// Address: Clermont Ferrand, France
// Date: 24/10/2012
// Email: pablofdezalc@gmail.com
//
// KAZE Features Copyright 2012, Pablo F. Alcantarilla
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file kaze_compare.cpp
 * @brief Simple image matching program that compares KAZE against other
 * features implemented in OpenCV such as SIFT and SURF
 * @date Oct 24, 2012
 * @author Pablo F. Alcantarilla
 */

#include "kaze_compare.h"

// Namespaces
using namespace std;
using namespace cv;

// Some image matching options
const float MAX_H_ERROR = 5.0;	// Maximum error in pixels to accept an inlier
const float DRATIO = 0.6;		// NNDR Matching value

//*************************************************************************************
//*************************************************************************************

/** Main Function 																	 */
int main(int argc, char *argv[]) {

  // Variables
  toptions options;
  Mat img1, img1_32, img2, img2_32;

  char img_name1[NMAX_CHAR], img_name2[NMAX_CHAR], hfile[NMAX_CHAR];
  char rfile[NMAX_CHAR];
  int key = 0;

  // Variables for measuring computation times
  float t1 = 0.0, t2 = 0.0;
  float tsift = 0.0, tsurf = 0.0, tkaze = 0.0;

  // SIFT Variables
  Mat desc_sift1, desc_sift2;
  vector<KeyPoint> kpts_sift1, kpts_sift2;
  int nkpts_sift1 = 0, nkpts_sift2 = 0;
  int nmatches_sift = 0, ninliers_sift = 0, noutliers_sift = 0;
  float ratio_sift = 0.0;
  vector<vector<DMatch> > dmatches_sift;
  vector<Point2f> matches_sift, inliers_sift;
  Mat img1_rgb_sift, img2_rgb_sift, img_com_sift;

  // SURF Variables
  Mat desc_surf1, desc_surf2;
  vector<KeyPoint> kpts_surf1, kpts_surf2;
  int nkpts_surf1 = 0, nkpts_surf2 = 0;
  int nmatches_surf = 0, ninliers_surf = 0, noutliers_surf = 0;
  float ratio_surf = 0.0;
  vector<vector<DMatch> > dmatches_surf;
  vector<Point2f> matches_surf, inliers_surf;
  Mat img1_rgb_surf, img2_rgb_surf, img_com_surf;
  SURF dsurf(10,4,2,false,false);

  // KAZE Variables
  Mat desc_kaze1, desc_kaze2;
  vector<KeyPoint> kpts_kaze1, kpts_kaze2;
  int nkpts_kaze1 = 0, nkpts_kaze2 = 0;
  int nmatches_kaze = 0, ninliers_kaze = 0, noutliers_kaze = 0;
  float ratio_kaze = 0.0;
  vector<vector<DMatch> > dmatches_kaze;
  vector<Point2f> matches_kaze, inliers_kaze;
  Mat img1_rgb_kaze, img2_rgb_kaze, img_com_kaze;

  // L2 matcher
  Ptr<DescriptorMatcher> matcher_l2 = DescriptorMatcher::create("BruteForce");

  // Parse the input command line options
  if (parse_input_options(options,img_name1,img_name2,hfile,rfile,argc,argv)) {
    return -1;
  }

  // Read the image, force to be grey scale
  img1 = imread(img_name1,0);

  if (img1.data == NULL) {
    cout << "Error loading image: " << img_name1 << endl;
    return -1;
  }

  // Read the image, force to be grey scale
  img2 = imread(img_name2,0);

  if (img2.data == NULL) {
    cout << "Error loading image: " << img_name2 << endl;
    return -1;
  }

  // Convert the images to float
  img1.convertTo(img1_32,CV_32F,1.0/255.0,0);
  img2.convertTo(img2_32,CV_32F,1.0/255.0,0);

  // Color images for results visualization
  img1_rgb_sift = cv::Mat(Size(img1.cols,img1.rows),CV_8UC3);
  img2_rgb_sift = cv::Mat(Size(img2.cols,img2.rows),CV_8UC3);
  img_com_sift = cv::Mat(Size(img1.cols*2,img1.rows),CV_8UC3);
  img1_rgb_surf = cv::Mat(Size(img1.cols,img1.rows),CV_8UC3);
  img2_rgb_surf = cv::Mat(Size(img2.cols,img2.rows),CV_8UC3);
  img_com_surf = cv::Mat(Size(img1.cols*2,img1.rows),CV_8UC3);
  img1_rgb_kaze = cv::Mat(Size(img1.cols,img1.rows),CV_8UC3);
  img2_rgb_kaze = cv::Mat(Size(img2.cols,img2.rows),CV_8UC3);
  img_com_kaze = cv::Mat(Size(img1.cols*2,img1.rows),CV_8UC3);

  // Read the homography file
  Mat H;
  read_homography(hfile,H);

  // OpenCV Windows for visualization
  namedWindow("SIFT",CV_WINDOW_NORMAL);
  namedWindow("SURF",CV_WINDOW_NORMAL);
  namedWindow("KAZE",CV_WINDOW_NORMAL);

  //*************************************************************************************
  //*************************************************************************************

  // Detect SIFT Features
  t1 = getTickCount();

  SIFT dsift(3000,3,0.004,10,1.6);
  dsift(img1,Mat(),kpts_sift1,desc_sift1,false);
  dsift(img2,Mat(),kpts_sift2,desc_sift2,false);

  t2 = getTickCount();
  tsift = 1000.0*(t2-t1) / getTickFrequency();

  nkpts_sift1 = kpts_sift1.size();
  nkpts_sift2 = kpts_sift2.size();

  // Matching Descriptors!!
  matcher_l2->knnMatch(desc_sift1,desc_sift2,dmatches_sift,2);
  matches2points_nndr(kpts_sift1,kpts_sift2,dmatches_sift,matches_sift,DRATIO);

  // Compute Inliers!!
  compute_inliers_homography(matches_sift,inliers_sift,H,MAX_H_ERROR);

  // Compute the inliers statistics
  nmatches_sift = matches_sift.size()/2;
  ninliers_sift = inliers_sift.size()/2;
  noutliers_sift = nmatches_sift - ninliers_sift;

  if (nmatches_sift != 0) {
    ratio_sift = 100.0*((float) ninliers_sift / (float) nmatches_sift);
  }

  // Prepare the visualization
  cvtColor(img1,img1_rgb_sift,CV_GRAY2BGR);
  cvtColor(img2,img2_rgb_sift,CV_GRAY2BGR);

  // Draw the list of detected points
  draw_keypoints(img1_rgb_sift,kpts_sift1);
  draw_keypoints(img2_rgb_sift,kpts_sift2);

  // Create the new image with a line showing the correspondences
  draw_inliers(img1_rgb_sift,img2_rgb_sift,img_com_sift,inliers_sift,0);
  display_text(img_com_sift,nkpts_sift1,nkpts_sift2,nmatches_sift,ninliers_sift,ratio_sift,0);

  //*************************************************************************************
  //*************************************************************************************

  // Detect SURF Features
  t1 = getTickCount();

  dsurf.hessianThreshold = 400.0;
  dsurf(img1,Mat(),kpts_surf1,desc_surf1,false);
  dsurf(img2,Mat(),kpts_surf2,desc_surf2,false);

  t2 = getTickCount();
  tsurf = 1000.0*(t2-t1) / getTickFrequency();

  nkpts_surf1 = kpts_surf1.size();
  nkpts_surf2 = kpts_surf2.size();

  // Matching Descriptors!!
  matcher_l2->knnMatch(desc_surf1,desc_surf2,dmatches_surf,2);
  matches2points_nndr(kpts_surf1,kpts_surf2,dmatches_surf,matches_surf,DRATIO);

  // Compute Inliers!!
  compute_inliers_homography(matches_surf,inliers_surf,H,MAX_H_ERROR);

  // Compute the inliers statistics
  nmatches_surf = matches_surf.size()/2;
  ninliers_surf = inliers_surf.size()/2;
  noutliers_surf = nmatches_surf - ninliers_surf;

  if (nmatches_surf != 0) {
    ratio_surf = 100.0*((float) ninliers_surf / (float) nmatches_surf);
  }

  // Prepare the visualization
  cvtColor(img1,img1_rgb_surf,CV_GRAY2BGR);
  cvtColor(img2,img2_rgb_surf,CV_GRAY2BGR);

  // Draw the list of detected points
  draw_keypoints(img1_rgb_surf,kpts_surf1);
  draw_keypoints(img2_rgb_surf,kpts_surf2);

  // Create the new image with a line showing the correspondences
  draw_inliers(img1_rgb_surf,img2_rgb_surf,img_com_surf,inliers_surf,1);
  display_text(img_com_surf,nkpts_surf1,nkpts_surf2,nmatches_surf,ninliers_surf,ratio_surf,1);

  //*************************************************************************************
  //*************************************************************************************

  // Create the first NLDiffusion object
  t1 = getTickCount();

  options.img_width = img1.cols;
  options.img_height = img1.rows;
  KAZE evolution1(options);

  // Create the nonlinear scale space
  // and perform feature detection and description for image 1
  evolution1.Create_Nonlinear_Scale_Space(img1_32);
  evolution1.Feature_Detection(kpts_kaze1);
  evolution1.Feature_Description(kpts_kaze1,desc_kaze1);

  // Create the second KAZE object
  options.img_width = img2.cols;
  options.img_height = img2.rows;
  KAZE evolution2(options);
  evolution2.Set_Detector_Threshold(options.dthreshold2);
  evolution2.Create_Nonlinear_Scale_Space(img2_32);
  evolution2.Feature_Detection(kpts_kaze2);
  evolution2.Feature_Description(kpts_kaze2,desc_kaze2);

  t2 = getTickCount();
  tkaze = 1000.0*(t2-t1) / getTickFrequency();

  nkpts_kaze1 = kpts_kaze1.size();
  nkpts_kaze2 = kpts_kaze2.size();

  // Matching Descriptors!!
  matcher_l2->knnMatch(desc_kaze1,desc_kaze2,dmatches_kaze,2);
  matches2points_nndr(kpts_kaze1,kpts_kaze2,dmatches_kaze,matches_kaze,DRATIO);

  // Compute Inliers!!
  compute_inliers_homography(matches_kaze,inliers_kaze,H,MAX_H_ERROR);

  // Compute the inliers statistics
  nmatches_kaze = matches_kaze.size()/2;
  ninliers_kaze = inliers_kaze.size()/2;
  noutliers_kaze = nmatches_kaze - ninliers_kaze;

  if (nmatches_kaze != 0) {
    ratio_kaze = 100.0*((float) ninliers_kaze / (float) nmatches_kaze);
  }

  // Prepare the visualization
  cvtColor(img1,img1_rgb_kaze,CV_GRAY2BGR);
  cvtColor(img2,img2_rgb_kaze,CV_GRAY2BGR);

  // Draw the list of detected points
  draw_keypoints(img1_rgb_kaze,kpts_kaze1);
  draw_keypoints(img2_rgb_kaze,kpts_kaze2);

  // Create the new image with a line showing the correspondences
  draw_inliers(img1_rgb_kaze,img2_rgb_kaze,img_com_kaze,inliers_kaze,2);
  display_text(img_com_kaze,nkpts_kaze1,nkpts_kaze2,nmatches_kaze,ninliers_kaze,ratio_kaze,2);

  //*************************************************************************************
  //*************************************************************************************

  // Show matching statistics
  if (options.show_results == true) {

    cout << endl;
    cout << "SIFT Results" << endl;
    cout << "**************************************" << endl;
    cout << "Number of Keypoints Image 1: " << nkpts_sift1 << endl;
    cout << "Number of Keypoints Image 2: " << nkpts_sift2 << endl;
    cout << "Number of Matches: " << nmatches_sift << endl;
    cout << "Number of Inliers: " << ninliers_sift << endl;
    cout << "Number of Outliers: " << noutliers_sift << endl;
    cout << "Inliers Ratio: " << ratio_sift << endl;
    cout << "SIFT Features Extraction Time (ms): " << tsift << endl << endl;

    cout << "SURF Results" << endl;
    cout << "**************************************" << endl;
    cout << "Number of Keypoints Image 1: " << nkpts_surf1 << endl;
    cout << "Number of Keypoints Image 2: " << nkpts_surf2 << endl;
    cout << "Number of Matches: " << nmatches_surf << endl;
    cout << "Number of Inliers: " << ninliers_surf << endl;
    cout << "Number of Outliers: " << noutliers_surf << endl;
    cout << "Inliers Ratio: " << ratio_surf << endl;
    cout << "SURF Features Extraction Time (ms): " << tsurf << endl << endl;

    cout << "KAZE Results" << endl;
    cout << "**************************************" << endl;
    cout << "Number of Keypoints Image 1: " << nkpts_kaze1 << endl;
    cout << "Number of Keypoints Image 2: " << nkpts_kaze2 << endl;
    cout << "Number of Matches: " << nmatches_kaze << endl;
    cout << "Number of Inliers: " << ninliers_kaze << endl;
    cout << "Number of Outliers: " << noutliers_kaze << endl;
    cout << "Inliers Ratio: " << ratio_kaze << endl;
    cout << "KAZE Features Extraction Time (ms): " << tkaze << endl;

    while (1) {
      imshow("SIFT",img_com_sift);
      imshow("SURF",img_com_surf);
      imshow("KAZE",img_com_kaze);

      key = waitKey(10);

      // If the user presses ESC exit the program
      if (key == 27) {
        break;
      }
    }

    // Destroy the windows
    destroyAllWindows();
  }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief  This function saves the input image with the correct matches
 * @param img Image to be saved
 */
void save_matching_image(const cv::Mat& img) {
  char outputFile[NMAX_CHAR];
  sprintf(outputFile,"../output/images/image_matching.jpg");
  imwrite(outputFile,img);
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function parses the command line arguments for setting KAZE parameters
 * and image matching between two input images
 * @param options Structure that contains KAZE settings
 * @param img_name1 Name of the first input image
 * @param img_name2 Name of the second input image
 * @param hom Name of the file that contains a ground truth homography
 * @param kfile Name of the file where the keypoints where be stored
 */
int parse_input_options(toptions &options, char *img_name1, char *img_name2, char *hom,
                        char *kfile, int argc, char *argv[]) {

  // If there is only one argument return
  if (argc == 1) {
    show_input_options_help();
    return -1;
  }
  // Set the options from the command line
  else if (argc >= 2) {

    // Load the default options
    options.soffset = DEFAULT_SCALE_OFFSET;
    options.omax = DEFAULT_OCTAVE_MAX;
    options.nsublevels = DEFAULT_NSUBLEVELS;
    options.dthreshold = DEFAULT_DETECTOR_THRESHOLD;
    options.dthreshold2 = DEFAULT_DETECTOR_THRESHOLD;
    options.diffusivity = DEFAULT_DIFFUSIVITY_TYPE;
    options.descriptor = DEFAULT_DESCRIPTOR_MODE;
    options.sderivatives = DEFAULT_SIGMA_SMOOTHING_DERIVATIVES;
    options.upright = DEFAULT_UPRIGHT;
    options.extended = DEFAULT_EXTENDED;
    options.save_scale_space = DEFAULT_SAVE_SCALE_SPACE;
    options.save_keypoints = DEFAULT_SAVE_KEYPOINTS;
    options.show_results = DEFAULT_SHOW_RESULTS;
    options.verbosity = DEFAULT_VERBOSITY;

    strcpy(img_name1,argv[1]);
    strcpy(img_name2,argv[2]);
    strcpy(hom,argv[3]);
    strcpy(kfile,"./results.txt");

    for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i],"--soffset")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.soffset = atof(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--omax")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.omax = atof(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--dthreshold")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.dthreshold = atof(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--dthreshold2")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.dthreshold2 = atof(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--sderivatives")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.sderivatives = atof(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--nsublevels")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.nsublevels = atoi(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--diffusivity")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.diffusivity = atoi(argv[i]);
          if (options.diffusivity > 2 || options.diffusivity < 0) {
            options.diffusivity = DEFAULT_DIFFUSIVITY_TYPE;
          }
        }
      }
      else if (!strcmp(argv[i],"--descriptor")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.descriptor = atoi(argv[i]);
          if (options.descriptor > 2 || options.descriptor < 0) {
            options.descriptor = DEFAULT_DESCRIPTOR_MODE;
          }
        }
      }
      else if (!strcmp(argv[i],"--save_scale_space")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.save_scale_space = (bool)atoi(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--show_results")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.show_results = (bool)atoi(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--kfile")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          strcpy(kfile,argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--upright")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.upright = (bool)atoi(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--extended"))
      {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.extended = (bool)atoi(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--verbose")) {
        options.verbosity = true;
      }
      else if (!strcmp(argv[i],"--help")) {
        show_input_options_help();
        return -1;
      }
    }
  }
  else {
    cout << "Error introducing input options!!" << endl;

    // Show the help!!
    show_input_options_help();
    return -1;
  }

  return 0;
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function shows the possible command line configuration options
 */
void show_input_options_help(void)
{
  fflush(stdout);

  cout << "KAZE Features" << endl;
  cout << "************************************************" << endl;
  cout << "For running the program you need to type in the command line the following arguments: " << endl;
  cout << "./kaze_compare img1.jpg img2.pgm homography.txt options" << endl;
  cout << "The options are not mandatory. In case you do not specify additional options, default arguments will be used"
       << endl << endl;
  cout << "In KAZE compare, the options are only used for KAZE features and not the other methods (SIFT,SURF)" << endl;
  cout << "Here is a description of the additional options: " << endl;
  cout << "--verbose " << "\t\t if verbosity is required" << endl;
  cout << "--help" << "\t\t for showing the command line options" << endl;
  cout << "--soffset" << "\t\t the base scale offset (sigma units)" << endl;
  cout << "--omax" << "\t\t maximum octave evolution of the image 2^sigma (coarsest scale)" << endl;
  cout << "--nsublevels" << "\t\t number of sublevels per octave" << endl;
  cout << "--dthreshold" << "\t\t Feature detector threshold response for accepting points (0.001 can be a good value)"
       << endl;
  cout << "--sderivatives" << "\t\t Standard deviation for the Gaussian derivatives in the nonlinear diffusion filtering"
       << endl;
  cout << "--descriptor" << "\t\t Descriptor Type 0 -> SURF, 1 -> M-SURF, 2 -> G-SURF" << endl;
  cout << "--upright" << "\t\t 0 -> Rotation Invariant, 1 -> No Rotation Invariant" << endl;
  cout << "--extended" << "\t\t 0 -> Normal Descriptor (64), 1 -> Extended Descriptor (128)" << endl;
  cout << "--show_results" << "\t\t 1 in case we want to show detection results. 0 otherwise" << endl;
  cout << endl;
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function displays text in the image with the matching statistics
 */
void display_text(cv::Mat &img_rgb, const int& npoints1, const int& npoints2, const int& nmatches, const int& ninliers,
                  const float& ratio, const int& index)
{
  char text[400];

  sprintf(text,"NNDR Matching %.2f",DRATIO);

  if (index == 0) {
    putText(img_rgb,text,Point(20,30),CV_FONT_HERSHEY_DUPLEX,.75,CV_RGB(255,255,0),2,8,false);
  }
  else if (index == 1) {
    putText(img_rgb,text,Point(20,30),CV_FONT_HERSHEY_DUPLEX,.75,CV_RGB(255,0,0),2,8,false);
  }
  else if (index == 2) {
    putText(img_rgb,text,Point(20,30),CV_FONT_HERSHEY_DUPLEX,.75,CV_RGB(0,255,0),2,8,false);
  }

  sprintf(text,"# Points Image 1: %d, # Points Image 2: %d",npoints1,npoints2);

  if (index == 0) {
    putText(img_rgb,text,Point(20,img_rgb.rows-70),CV_FONT_HERSHEY_DUPLEX,.75,CV_RGB(255,255,0),2,8,false);
  }
  else if (index == 1) {
    putText(img_rgb,text,Point(20,img_rgb.rows-70),CV_FONT_HERSHEY_DUPLEX,.75,CV_RGB(255,0,0),2,8,false);
  }
  else if (index == 2) {
    putText(img_rgb,text,Point(20,img_rgb.rows-70),CV_FONT_HERSHEY_DUPLEX,.75,CV_RGB(0,255,0),2,8,false);
  }

  sprintf(text,"# Matches: %d, # Inliers: %d, Ratio %.2f",nmatches,ninliers,ratio);

  if (index == 0) {
    putText(img_rgb,text,Point(20,img_rgb.rows-30),CV_FONT_HERSHEY_DUPLEX,.75,CV_RGB(255,255,0),2,8,false);
  }
  else if (index == 1) {
    putText(img_rgb,text,Point(20,img_rgb.rows-30),CV_FONT_HERSHEY_DUPLEX,.75,CV_RGB(255,0,0),2,8,false);
  }
  else if (index == 2) {
    putText(img_rgb,text,Point(20,img_rgb.rows-30),CV_FONT_HERSHEY_DUPLEX,.75,CV_RGB(0,255,0),2,8,false);
  }
}
