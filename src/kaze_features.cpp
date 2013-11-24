
//=============================================================================
//
// kaze_features.cpp
// Author: Pablo F. Alcantarilla
// Institution: University d'Auvergne
// Address: Clermont Ferrand, France
// Date: 20/01/2012
// Email: pablofdezalc@gmail.com
//
// KAZE Features Copyright 2012, Pablo F. Alcantarilla
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file kaze_features.cpp
 * @brief Main program for detecting and computing descriptors in a nonlinear
 * scale space
 * @date Jan 20, 2012
 * @author Pablo F. Alcantarilla
 */

#include "kaze_features.h"

using namespace std;
using namespace cv;

//*************************************************************************************
//*************************************************************************************

/** Main Function 																	 */
int main(int argc, char *argv[]) {

  KAZEOptions options;
  Mat img, img_32, img_rgb;
  string img_path, kpts_path;

  // Parse the input command line options
  if (parse_input_options(options,img_path,kpts_path,argc,argv)) {
    return -1;
  }

  // Read the image, force to be grey scale
  img = imread(img_path,0);

  if (img.data == NULL) {
    cerr << "Error loading image: " << img_path << endl;
    return -1;
  }

  // Convert the image to float
  img.convertTo(img_32,CV_32F,1.0/255.0,0);
  img_rgb = Mat(Size(img.cols,img.rows),CV_8UC3);

  options.img_width = img.cols;
  options.img_height = img.rows;

  // Create the KAZE object
  KAZE evolution(options);

  // Create the nonlinear scale space
  evolution.Create_Nonlinear_Scale_Space(img_32);

  vector<KeyPoint> kpts;
  Mat desc;

  evolution.Feature_Detection(kpts);
  evolution.Feature_Description(kpts,desc);

  // Save the nonlinear scale space images
  if (options.save_scale_space == true) {
    evolution.Save_Nonlinear_Scale_Space();
    evolution.Save_Flow_Responses();
  }

  if (options.show_results == true) {
    cout << "Time Scale Space: " << evolution.Get_Time_NLScale() << endl;
    cout << "Time Detector: " << evolution.Get_Time_Detector() << endl;
    cout << "Time Descriptor: " << evolution.Get_Time_Descriptor() << endl;
    cout << "Number of Keypoints: " << kpts.size() << endl;

    // Create the OpenCV window
    namedWindow("Image",CV_WINDOW_FREERATIO);

    // Copy the input image to the color one
    cvtColor(img,img_rgb,CV_GRAY2BGR);

    // Draw the list of detected points
    draw_keypoints(img_rgb,kpts);

    imshow("Image",img_rgb);
    waitKey(0);

    // Destroy the windows
    destroyAllWindows();

    if (options.save_scale_space == true) {
      // Copy the input image to the color one
      cvtColor(img,img_rgb,CV_GRAY2BGR);

      // Draw the list of detected points
      draw_keypoints(img_rgb,kpts);

      // Save the rgb image
      save_image_with_features(img_rgb);
    }
  }

  // Save the list of keypoints
  if (options.save_keypoints == true) {
    save_keypoints(kpts_path,kpts,desc,false);
  }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief  This function saves the input image with the detected features
 * @param img Image to be saved
 */
void save_image_with_features(cv::Mat& img) {
  string outputFile = "./image_features.jpg";
  imwrite(outputFile,img);
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function parses the command line arguments for setting KAZE parameters
 * @param options Structure that contains KAZE settings
 * @param img_path Path for the input image
 * @param kpts_path Path for the file where the keypoints where be stored
 */
int parse_input_options(KAZEOptions& options, std::string& img_path,
                        std::string& kpts_path, int argc, char *argv[]) {

  // If there is only one argument return
  if (argc == 1) {
    show_input_options_help(0);
    return -1;
  }
  // Set the options from the command line
  else if (argc >= 2) {

    if (!strcmp(argv[1],"--help")) {
      show_input_options_help(0);
      return -1;
    }

    img_path = argv[1];

    for (int i = 2; i < argc; i++) {
      if (!strcmp(argv[i],"--soffset")) {
        i = i+1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.soffset = atof(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--omax")) {
        i = i+1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.omax = atof(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--dthreshold")) {
        i = i+1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.dthreshold = atof(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--sderivatives")) {
        i = i+1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.sderivatives = atof(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--nsublevels")) {
        i = i+1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.nsublevels = atoi(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--diffusivity")) {
        i = i+1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
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
          cerr << "Error introducing input options!!" << endl;
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
          cerr << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.save_scale_space = (bool)atoi(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--show_results")) {
        i = i+1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.show_results = (bool)atoi(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--use_fed")) {
        i = i+1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.use_fed = (bool)atoi(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--upright")) {
        i = i+1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.upright = (bool)atoi(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--extended")) {
        i = i+1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.extended = (bool)atoi(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--verbose")) {
        options.verbosity = true;
      }
      else if (!strcmp(argv[i],"--output")) {
        options.save_keypoints = true;
        i = i+1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          kpts_path = argv[i];
        }
      }
      else if (!strcmp(argv[i],"--help")) {
        show_input_options_help(0);
        return -1;
      }
    }
  }

  return 0;
}
