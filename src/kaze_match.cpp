
//=============================================================================
//
// kaze_match.cpp
// Author: Pablo F. Alcantarilla
// Institution: University d'Auvergne
// Address: Clermont Ferrand, France
// Date: 22/10/2012
// Email: pablofdezalc@gmail.com
//
// KAZE Features Copyright 2012, Pablo F. Alcantarilla
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file kaze_match.cpp
 * @brief Main program for matching two images with KAZE features
 * The two images can have different resolutions
 * @date Oct 22, 2012
 * @author Pablo F. Alcantarilla
 */

#include "kaze_match.h"

// Namespaces
using namespace std;
using namespace cv;

//*************************************************************************************
//*************************************************************************************

/** Main Function 																	 */
int main( int argc, char *argv[] ) {

  KAZEOptions options;
  Mat img1, img1_32, img2, img2_32, img1_rgb, img2_rgb, img_com, img_r;
  string img_path1, img_path2, homography_path;
  float ratio = 0.0, rfactor = .90;
  vector<KeyPoint> kpts1, kpts2;
  vector<vector<DMatch> > dmatches;
  Mat desc1, desc2, H;
  int nkpts1 = 0, nkpts2 = 0, nmatches = 0, ninliers = 0, noutliers = 0;

  // Variables for measuring computation times
  double t1 = 0.0, t2 = 0.0, tkaze = 0.0, tmatch = 0.0;

  // Parse the input command line options
  if (parse_input_options(options,img_path1,img_path2,homography_path,argc,argv)) {
    return -1;
  }

  // Read the image, force to be grey scale
  img1 = imread(img_path1,0);

  if (img1.data == NULL) {
    cerr << "Error loading image: " << img_path1 << endl;
    return -1;
  }

  // Read the image, force to be grey scale
  img2 = imread(img_path2,0);

  if (img2.data == NULL) {
    cout << "Error loading image: " << img_path2 << endl;
    return -1;
  }

  // Convert the images to float
  img1.convertTo(img1_32,CV_32F,1.0/255.0,0);
  img2.convertTo(img2_32,CV_32F,1.0/255.0,0);

  // Color images for results visualization
  img1_rgb = Mat(Size(img1.cols,img1.rows),CV_8UC3);
  img2_rgb = Mat(Size(img2.cols,img1.rows),CV_8UC3);
  img_com = Mat(Size(img1.cols*2,img1.rows),CV_8UC3);
  img_r = Mat(Size(img_com.cols*rfactor,img_com.rows*rfactor),CV_8UC3);

  // Read the homography file
  read_homography(homography_path,H);

  // Create the first KAZE object
  options.img_width = img1.cols;
  options.img_height = img1.rows;
  KAZE evolution1(options);

  t1 = getTickCount();

  // Create the nonlinear scale space
  // and perform feature detection and description for image 1
  evolution1.Create_Nonlinear_Scale_Space(img1_32);
  evolution1.Feature_Detection(kpts1);
  evolution1.Feature_Description(kpts1,desc1);

  // Create the second KAZE object
  options.img_width = img2.cols;
  options.img_height = img2.rows;
  KAZE evolution2(options);

  evolution2.Create_Nonlinear_Scale_Space(img2_32);
  evolution2.Feature_Detection(kpts2);
  evolution2.Feature_Description(kpts2,desc2);

  t2 = getTickCount();
  tkaze = 1000.0*(t2-t1) / getTickFrequency();

  nkpts1 = kpts1.size();
  nkpts2 = kpts2.size();

  // Matching Descriptors!!
  vector<Point2f> matches, inliers;
  Ptr<DescriptorMatcher> matcher_l2 = DescriptorMatcher::create("BruteForce");

  t1 = getTickCount();

  matcher_l2->knnMatch(desc1,desc2,dmatches,2);
  matches2points_nndr(kpts1,kpts2,dmatches,matches,DRATIO);

  t2 = getTickCount();
  tmatch = 1000.0*(t2-t1) / cv::getTickFrequency();

  // Compute Inliers!!
  if (COMPUTE_HOMOGRAPHY == false) {
    compute_inliers_homography(matches,inliers,H,MAX_H_ERROR);
  }
  else {
    compute_inliers_ransac(matches,inliers,MAX_H_ERROR,false);
  }

  // Compute the inliers statistics
  nmatches = matches.size()/2;
  ninliers = inliers.size()/2;
  noutliers = nmatches - ninliers;
  ratio = 100.0*((float) ninliers / (float) nmatches);

  // Prepare the visualization
  cvtColor(img1,img1_rgb,CV_GRAY2BGR);
  cvtColor(img2,img2_rgb,CV_GRAY2BGR);

  // Draw the list of detected points
  draw_keypoints(img1_rgb,kpts1);
  draw_keypoints(img2_rgb,kpts2);

  // Create the new image with a line showing the correspondences
  draw_inliers(img1_rgb,img2_rgb,img_com,inliers);
  resize(img_com,img_r,Size(img_r.cols,img_r.rows),0,0,CV_INTER_LINEAR);

  // Show matching statistics
  if (options.show_results == true) {

    cout << "Number of Keypoints Image 1: " << nkpts1 << endl;
    cout << "Number of Keypoints Image 2: " << nkpts2 << endl;
    cout << "KAZE Features Extraction Time (ms): " << tkaze << endl;
    cout << "Matching Descriptors Time (ms): " << tmatch << endl;
    cout << "Number of Matches: " << nmatches << endl;
    cout << "Number of Inliers: " << ninliers << endl;
    cout << "Number of Outliers: " << noutliers << endl;
    cout << "Inliers Ratio: " << ratio << endl << endl;

    // Show the images in OpenCV windows
    namedWindow("Image 1",CV_WINDOW_NORMAL);
    namedWindow("Image 2",CV_WINDOW_NORMAL);
    namedWindow("Matches",CV_WINDOW_NORMAL);

    imshow("Image 1",img1_rgb);
    imshow("Image 2",img2_rgb);
    imshow("Matches",img_com);

    waitKey(0);

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
  string outputFile = "./image_matching.jpg";
  imwrite(outputFile,img);
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function parses the command line arguments for setting KAZE parameters
 * and image matching between two input images
 * @param options Structure that contains KAZE settings
 * @param img_path1 Name of the first input image
 * @param img_path2 Name of the second input image
 * @param homography_path Name of the file that contains a ground truth homography
 */
int parse_input_options(KAZEOptions& options, std::string& img_path1, std::string& img_path2,
                        std::string& homography_path, int argc, char *argv[]) {

  // If there is only one argument return
  if (argc == 1) {
    show_input_options_help(1);
    return -1;
  }
  // Set the options from the command line
  else if (argc >= 2) {

    if (!strcmp(argv[1],"--help")) {
      show_input_options_help(1);
      return -1;
    }

    img_path1 = argv[1];
    img_path2 = argv[2];
    homography_path = argv[3];

    for (int i = 4; i < argc; i++) {
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
      else if (!strcmp(argv[i],"--use_fed")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.use_fed = (bool)atoi(argv[i]);
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
      else if (!strcmp(argv[i],"--extended")) {
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
        show_input_options_help(1);
        return -1;
      }
    }
  }
  else {
    cout << "Error introducing input options!!" << endl;
    show_input_options_help(1);
    return -1;
  }

  return 0;
}
