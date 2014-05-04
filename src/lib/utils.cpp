//=============================================================================
//
// utils.cpp
// Author: Pablo F. Alcantarilla
// Institution: University d'Auvergne
// Address: Clermont Ferrand, France
// Date: 29/12/2011
// Email: pablofdezalc@gmail.com
//
// KAZE Features Copyright 2012, Pablo F. Alcantarilla
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file utils.cpp
 * @brief Some useful functions
 * @date Dec 29, 2011
 * @author Pablo F. Alcantarilla
 */

#include "utils.h"

// System
#include <fstream>

using namespace std;
using namespace cv;

/* ************************************************************************* */
/**
 * @brief This function computes the minimum value of a float image
 * @param src Input image
 * @param value Minimum value
 */
void compute_min_32F(const cv::Mat& src, float& value) {

  float aux = 1000.0;

  for (int i = 0; i < src.rows; i++) {
    for(int j = 0; j < src.cols; j++) {
      if (src.at<float>(i,j) < aux) {
        aux = src.at<float>(i,j);
      }
    }
  }

  value = aux;
}

/* ************************************************************************* */
/**
 * @brief This function computes the maximum value of a float image
 * @param src Input image
 * @param value Maximum value
 */
void compute_max_32F(const cv::Mat& src, float& value) {

  float aux = 0.0;

  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      if (src.at<float>(i,j) > aux) {
        aux = src.at<float>(i,j);
      }
    }
  }

  value = aux;
}

/* ************************************************************************* */
/**
 * @brief This function converts the scale of the input image prior to visualization
 * @param src Input/Output image
 * @param value Maximum value
 */
void convert_scale(cv::Mat& src) {

  float min_val = 0, max_val = 0;

  compute_min_32F(src,min_val);

  src = src - min_val;

  compute_max_32F(src,max_val);
  src = src / max_val;
}

/* ************************************************************************* */
/**
 * @brief This function copies the input image and converts the scale of the copied
 * image prior visualization
 * @param src Input image
 * @param dst Output image
 */
void copy_and_convert_scale(const cv::Mat& src, cv::Mat& dst) {

  float min_val = 0, max_val = 0;

  src.copyTo(dst);
  compute_min_32F(dst,min_val);

  dst = dst - min_val;

  compute_max_32F(dst,max_val);
  dst = dst / max_val;
}

/* ************************************************************************* */
/**
 * @brief This function draws a vector of Ipoints
 * @param img Input/Output Image
 * @param dst Vector of keypoints
 */
void draw_keypoints(cv::Mat& img, const std::vector<cv::KeyPoint>& kpts) {

  int x = 0, y = 0;
  float s = 0.0;

  for (size_t i = 0; i < kpts.size(); i++) {

    x = kpts[i].pt.x;
    y = kpts[i].pt.y;
    s = kpts[i].size*.5;

    // Draw a circle centered on the interest point
    circle(img,cv::Point(x,y),s,CV_RGB(0,0,255),1);
    circle(img,cv::Point(x,y),1.0,CV_RGB(0,255,0),-1);
  }
}

/* ************************************************************************* */
/**
 * @brief  This function saves the interest points to a regular ASCII file
 * @note The format is compatible with Mikolajczy and Schmid evaluation
 * @param sFileName Name of the output file where the points will be stored
 * @param kpts Vector of points of interest
 * @param desc Descriptors
 * @param bLaplacian Set to 1 if we want to write the sign of the Laplacian
 * into the descriptor information
 * @param bVerbose Set to 1 for some verbosity information
 */
int save_keypoints(std::string& keypointsFile, const std::vector<cv::KeyPoint>& kpts,
                   const cv::Mat &desc, bool bVerbose) {

  int length = 0, count = 0;
  float sc = 0.0;

  ofstream ipfile(keypointsFile.c_str());

  if (!ipfile) {
    cerr << "ERROR in save_keypoints: "
         << "Couldn't open file '" << keypointsFile << "'!" << endl;
    return -1;
  }

  length = desc.cols;
  count = desc.rows;

  // Write the file header
  ipfile << length << endl << count << endl;

  // In order to just save the interest points without descriptor, comment
  // the above and uncomment the following command.
  // ipfile << 1.0 << endl << count << endl;
  // Save interest point with descriptor in the format of Krystian Mikolajczyk
  // for reasons of comparison with other descriptors. As our interest points
  // are circular in any case, we use the second component of the ellipse to
  // provide some information about the strength of the interest point. This is
  // important for 3D reconstruction as only the strongest interest points are
  // considered. Replace the strength with 0.0 in order to perform Krystian's
  // comparisons.
  for (int i = 0; i < count; i++) {

    // circular regions with diameter 2*scale x 2*scale
    sc = kpts[i].size/2.0;
    sc*=sc;

    ipfile  << kpts[i].pt.x /* x-location of the interest point */
            << " " << kpts[i].pt.y /* y-location of the interest point */
            << " " << 1.0/sc /* 1/r^2 */
            << " " << 0.0
            << " " << 1.0/sc; /* 1/r^2 */

    // Here comes the descriptor
    for (int j = 0; j < length; j++) {
      ipfile << " " << desc.at<float>(i,j);
    }

    ipfile << endl;
  }

  ipfile.close();

  // Write message to terminal.
  if (bVerbose == true) {
    cout << count << " interest points found" << endl;
  }

  return 1;
}

/* ************************************************************************* */
/**
 * @brief This function converts matches to points using nearest neighbor distance
 * ratio matching strategy
 * @param train Vector of keypoints from the first image
 * @param query Vector of keypoints from the second image
 * @param matches Vector of nearest neighbors for each keypoint
 * @param pmatches Vector of putative matches
 * @param nndr Nearest neighbor distance ratio value
 */
void matches2points_nndr(const std::vector<cv::KeyPoint>& train, const std::vector<cv::KeyPoint>& query,
                         const std::vector<std::vector<cv::DMatch> >& matches,
                         std::vector<cv::Point2f>& pmatches, const float& nndr) {

  float dist1 = 0.0, dist2 = 0.0;
  for (size_t i = 0; i < matches.size(); i++) {
    cv::DMatch dmatch = matches[i][0];
    dist1 = matches[i][0].distance;
    dist2 = matches[i][1].distance;

    if (dist1 < nndr*dist2) {
      pmatches.push_back(train[dmatch.queryIdx].pt);
      pmatches.push_back(query[dmatch.trainIdx].pt);
    }
  }
}

/* ************************************************************************* */
/**
 * @brief This function computes the set of inliers estimating the fundamental matrix
 * or a planar homography in a RANSAC procedure
 * @param matches Vector of putative matches
 * @param inliers Vector of inliers
 * @param error The minimum pixelic error to accept an inlier
 * @param use_fund Set to true if you want to compute a fundamental matrix
 */
void compute_inliers_ransac(const std::vector<cv::Point2f> &matches, std::vector<cv::Point2f> &inliers,
                            const float& error, const bool& use_fund) {

  vector<Point2f> points1, points2;
  Mat H = Mat::zeros(3,3,CV_32F);
  int npoints = matches.size()/2;
  Mat status = Mat::zeros(npoints,1,CV_8UC1);

  for (size_t i = 0; i < matches.size(); i+=2) {
    points1.push_back(matches[i]);
    points2.push_back(matches[i+1]);
  }

  if (use_fund == true) {
    H = findFundamentalMat(points1,points2,CV_FM_RANSAC,error,0.99,status);
  }
  else {
    H = findHomography(points1,points2,CV_RANSAC,error,status);
  }

  for (int i = 0; i < npoints; i++) {
    if (status.at<unsigned char>(i) == 1) {
      inliers.push_back(points1[i]);
      inliers.push_back(points2[i]);
    }
  }
}

/* ************************************************************************* */
/**
 * @brief This function computes the set of inliers given a ground truth homography
 * @param matches Vector of putative matches
 * @param inliers Vector of inliers
 * @param H Ground truth homography matrix 3x3
 * @param min_error The minimum pixelic error to accept an inlier
 */
void compute_inliers_homography(const std::vector<cv::Point2f> &matches,
                                std::vector<cv::Point2f> &inliers, const cv::Mat &H, const float& min_error) {

  float h11 = 0.0, h12 = 0.0, h13 = 0.0;
  float h21 = 0.0, h22 = 0.0, h23 = 0.0;
  float h31 = 0.0, h32 = 0.0, h33 = 0.0;
  float x1 = 0.0, y1 = 0.0;
  float x2 = 0.0, y2 = 0.0;
  float x2m = 0.0, y2m = 0.0;
  float dist = 0.0, s = 0.0;

  h11 = H.at<float>(0,0);
  h12 = H.at<float>(0,1);
  h13 = H.at<float>(0,2);
  h21 = H.at<float>(1,0);
  h22 = H.at<float>(1,1);
  h23 = H.at<float>(1,2);
  h31 = H.at<float>(2,0);
  h32 = H.at<float>(2,1);
  h33 = H.at<float>(2,2);

  inliers.clear();

  for (size_t i = 0; i < matches.size(); i+=2) {
    x1 = matches[i].x;
    y1 = matches[i].y;
    x2 = matches[i+1].x;
    y2 = matches[i+1].y;

    s = h31*x1 + h32*y1 + h33;
    x2m = (h11*x1 + h12*y1 + h13) / s;
    y2m = (h21*x1 + h22*y1 + h23) / s;
    dist = sqrt( pow(x2m-x2,2) + pow(y2m-y2,2));

    if (dist <= min_error) {
      inliers.push_back(matches[i]);
      inliers.push_back(matches[i+1]);
    }
  }
}

/* ************************************************************************* */
/**
 * @brief This function draws the set of the inliers between the two images
 * @param img1 First image
 * @param img2 Second image
 * @param img_com Image with the inliers
 * @param ptpairs Vector of point pairs with the set of inliers
 */
void draw_inliers(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &img_com,
                  const std::vector<cv::Point2f> &ptpairs) {

  int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
  float rows1 = 0.0, cols1 = 0.0;
  float rows2 = 0.0, cols2 = 0.0;
  float ufactor = 0.0, vfactor = 0.0;

  rows1 = img1.rows;
  cols1 = img1.cols;
  rows2 = img2.rows;
  cols2 = img2.cols;
  ufactor = (float)(cols1)/(float)(cols2);
  vfactor = (float)(rows1)/(float)(rows2);

  // This is in case the input images don't have the same resolution
  Mat img_aux = Mat(Size(img1.cols,img1.rows),CV_8UC3);
  resize(img2,img_aux,Size(img1.cols,img1.rows),0,0,CV_INTER_LINEAR);

  for (int i = 0; i < img_com.rows; i++) {
    for (int j = 0; j < img_com.cols; j++) {
      if (j < img1.cols) {
        *(img_com.ptr<unsigned char>(i)+3*j) = *(img1.ptr<unsigned char>(i)+3*j);
        *(img_com.ptr<unsigned char>(i)+3*j+1) = *(img1.ptr<unsigned char>(i)+3*j+1);
        *(img_com.ptr<unsigned char>(i)+3*j+2) = *(img1.ptr<unsigned char>(i)+3*j+2);
      }
      else {
        *(img_com.ptr<unsigned char>(i)+3*j) = *(img2.ptr<unsigned char>(i)+3*(j-img_aux.cols));
        *(img_com.ptr<unsigned char>(i)+3*j+1) = *(img2.ptr<unsigned char>(i)+3*(j-img_aux.cols)+1);
        *(img_com.ptr<unsigned char>(i)+3*j+2) = *(img2.ptr<unsigned char>(i)+3*(j-img_aux.cols)+2);
      }
    }
  }

  for (size_t i = 0; i < ptpairs.size(); i+= 2) {
    x1 = (int)(ptpairs[i].x+.5);
    y1 = (int)(ptpairs[i].y+.5);
    x2 = (int)(ptpairs[i+1].x*ufactor+img1.cols+.5);
    y2 = (int)(ptpairs[i+1].y*vfactor+.5);
    line(img_com,Point(x1,y1),Point(x2,y2),CV_RGB(255,0,0),2);
  }
}

/* ************************************************************************* */
/**
 * @brief This function draws the set of the inliers between the two images
 * @param img1 First image
 * @param img2 Second image
 * @param img_com Image with the inliers
 * @param ptpairs Vector of point pairs with the set of inliers
 * @param color The color for each method
 */
void draw_inliers(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& img_com,
                  const std::vector<cv::Point2f>& ptpairs, int color) {

  int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
  float rows1 = 0.0, cols1 = 0.0;
  float rows2 = 0.0, cols2 = 0.0;
  float ufactor = 0.0, vfactor = 0.0;

  rows1 = img1.rows;
  cols1 = img1.cols;
  rows2 = img2.rows;
  cols2 = img2.cols;
  ufactor = (float)(cols1)/(float)(cols2);
  vfactor = (float)(rows1)/(float)(rows2);

  // This is in case the input images don't have the same resolution
  Mat img_aux = Mat(Size(img1.cols,img1.rows),CV_8UC3);
  resize(img2,img_aux,Size(img1.cols,img1.rows),0,0,CV_INTER_LINEAR);

  for (int i = 0; i < img_com.rows; i++) {
    for (int j = 0; j < img_com.cols; j++) {
      if ( j < img1.cols ) {
        *(img_com.ptr<unsigned char>(i)+3*j) = *(img1.ptr<unsigned char>(i)+3*j);
        *(img_com.ptr<unsigned char>(i)+3*j+1) = *(img1.ptr<unsigned char>(i)+3*j+1);
        *(img_com.ptr<unsigned char>(i)+3*j+2) = *(img1.ptr<unsigned char>(i)+3*j+2);
      }
      else {
        *(img_com.ptr<unsigned char>(i)+3*j) = *(img2.ptr<unsigned char>(i)+3*(j-img_aux.cols));
        *(img_com.ptr<unsigned char>(i)+3*j+1) = *(img2.ptr<unsigned char>(i)+3*(j-img_aux.cols)+1);
        *(img_com.ptr<unsigned char>(i)+3*j+2) = *(img2.ptr<unsigned char>(i)+3*(j-img_aux.cols)+2);
      }
    }
  }

  for (size_t i = 0; i < ptpairs.size(); i+= 2) {
    x1 = (int)(ptpairs[i].x+.5);
    y1 = (int)(ptpairs[i].y+.5);
    x2 = (int)(ptpairs[i+1].x*ufactor+img1.cols+.5);
    y2 = (int)(ptpairs[i+1].y*vfactor+.5);

    if (color == 0) {
      line(img_com,Point(x1,y1),Point(x2,y2),CV_RGB(255,255,0),2);
    }
    else if (color == 1) {
      line(img_com,Point(x1,y1),Point(x2,y2),CV_RGB(255,0,0),2);
    }
    else if (color == 2) {
      line(img_com,Point(x1,y1),Point(x2,y2),CV_RGB(0,255,0),2);
    }
  }
}

/* ************************************************************************* */
/**
 * @brief Function for reading the ground truth homography from a txt file
 * @param homography_file Path for the file that contains the ground truth homography
 * @param HG Matrix to store the ground truth homography
 */
void read_homography(const std::string& homography_path, cv::Mat& H1toN) {

  float h11 = 0.0, h12 = 0.0, h13 = 0.0;
  float h21 = 0.0, h22 = 0.0, h23 = 0.0;
  float h31 = 0.0, h32 = 0.0, h33 = 0.0;
  int  tmp_buf_size = 256;
  char tmp_buf[tmp_buf_size];
  string tmp_string;

  // Allocate memory for the OpenCV matrices
  H1toN = Mat::zeros(3,3,CV_32FC1);

  setlocale(LC_ALL,"C");

  ifstream infile;
  infile.exceptions(ifstream::eofbit | ifstream::failbit | ifstream::badbit );
  infile.open(homography_path.c_str(),ifstream::in);

  infile.getline(tmp_buf,tmp_buf_size);
  sscanf(tmp_buf,"%f %f %f",&h11,&h12,&h13);

  infile.getline(tmp_buf,tmp_buf_size);
  sscanf(tmp_buf,"%f %f %f",&h21,&h22,&h23);

  infile.getline(tmp_buf,tmp_buf_size);
  sscanf(tmp_buf,"%f %f %f",&h31,&h32,&h33);

  infile.close();

  H1toN.at<float>(0,0) = h11 / h33;
  H1toN.at<float>(0,1) = h12 / h33;
  H1toN.at<float>(0,2) = h13 / h33;

  H1toN.at<float>(1,0) = h21 / h33;
  H1toN.at<float>(1,1) = h22 / h33;
  H1toN.at<float>(1,2) = h23 / h33;

  H1toN.at<float>(2,0) = h31 / h33;
  H1toN.at<float>(2,1) = h32 / h33;
  H1toN.at<float>(2,2) = h33 / h33;
}

/* ************************************************************************* */
/**
 * @brief This function shows the possible command line configuration options
 */
void show_input_options_help(int example) {

  fflush(stdout);

  cout << endl;
  cout << endl;
  cout << "KAZE Features" << endl;
  cout << "***********************************************************" << endl;
  cout << "For running the program you need to type in the command line the following arguments: " << endl;

  if (example == 0) {
    cout << "./kaze_features img.jpg [options]" << endl;
  }
  else if (example == 1) {
    cout << "./kaze_match img1.jpg img2.pgm homography.txt [options]" << endl;
  }
  else if (example == 2) {
    cout << "./kaze_compare img1.jpg img2.pgm homography.txt [options]" << endl;
  }

  cout << endl;
  cout << "The options are not mandatory. In case you do not specify additional options, default arguments will be used" << endl << endl;
  cout << "Here is a description of the additional options: " << endl;
  cout << "--verbose " << "\t\t if verbosity is required" << endl;
  cout << "--help" << "\t\t for showing the command line options" << endl;
  cout << "--soffset" << "\t\t the base scale offset (sigma units)" << endl;
  cout << "--omax" << "\t\t maximum octave evolution of the image 2^sigma (coarsest scale)" << endl;
  cout << "--nsublevels" << "\t\t number of sublevels per octave" << endl;
  cout << "--dthreshold" << "\t\t Feature detector threshold response for accepting points (0.001 can be a good value)" << endl;
  cout << "--descriptor" << "\t\t Descriptor Type 0 -> SURF, 1 -> M-SURF, 2 -> G-SURF" << endl;
  cout << "--use_fed" "\t\t 1 -> Use FED, 0 -> Use AOS for the nonlinear diffusion filtering" << endl;
  cout << "--upright" << "\t\t 0 -> Rotation Invariant, 1 -> No Rotation Invariant" << endl;
  cout << "--extended" << "\t\t 0 -> Normal Descriptor (64), 1 -> Extended Descriptor (128)" << endl;
  cout << "--output keypoints.txt" << "\t\t For saving the detected keypoints into a .txt file" << endl;
  cout << "--save_scale_space" << "\t\t 1 in case we want to save the nonlinear scale space images. 0 otherwise" << endl;
  cout << "--show_results" << "\t\t 1 in case we want to show detection results. 0 otherwise" << endl;
  cout << endl;
}

/* ************************************************************************* */
/**
 * @brief This function displays text in the image with the matching statistics
 */
void display_text(cv::Mat& img_rgb, int npoints1, int npoints2, int nmatches,
                  int ninliers, float ratio, float dratio, int index) {

  char text[400];

  sprintf(text,"NNDR Matching %.2f",dratio);

  if (index == 0){
    putText(img_rgb,text,cv::Point(20,30),CV_FONT_HERSHEY_DUPLEX,.75,CV_RGB(255,255,0),2,8,false);
  }
  else if (index == 1) {
    putText(img_rgb,text,cv::Point(20,30),CV_FONT_HERSHEY_DUPLEX,.75,CV_RGB(255,0,0),2,8,false);
  }
  else if (index == 2) {
    putText(img_rgb,text,cv::Point(20,30),CV_FONT_HERSHEY_DUPLEX,.75,CV_RGB(0,255,0),2,8,false);
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
