
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

// Namespaces
using namespace std;

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes the minimum value of a float image
 * @param src Input image
 * @param value Minimum value
 */
void Compute_min_32F(const cv::Mat &src, float &value)
{
   float aux = 1000.0;
   
   for( int i = 0; i < src.rows; i++ )
   {
	   for( int j = 0; j < src.cols; j++ )
	   {
		   if( src.at<float>(i,j) < aux )
		   {
			   aux = src.at<float>(i,j);
		   }
	   }
   }	
   
   value = aux;
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes the maximum value of a float image
 * @param src Input image
 * @param value Maximum value
 */
void Compute_max_32F(const cv::Mat &src, float &value)
{
   float aux = 0.0;

   for( int i = 0; i < src.rows; i++ )
   {
	   for( int j = 0; j < src.cols; j++ )
	   {
		   if( src.at<float>(i,j) > aux )
		   {
			   aux = src.at<float>(i,j);
		   }
	   }
   }	
  
   value = aux;
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function converts the scale of the input image prior to visualization
 * @param src Input/Output image
 * @param value Maximum value
 */
void Convert_Scale(cv::Mat &src)
{
   float min_val = 0, max_val = 0;

   Compute_min_32F(src,min_val);

   src = src - min_val;

   Compute_max_32F(src,max_val);
   src = src / max_val;
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function copies the input image and converts the scale of the copied
 * image prior visualization
 * @param src Input image
 * @param dst Output image
 */
void Copy_and_Convert_Scale(const cv::Mat &src, cv::Mat dst)
{
   float min_val = 0, max_val = 0;

   src.copyTo(dst);
   Compute_min_32F(dst,min_val);

   dst = dst - min_val;

   Compute_max_32F(dst,max_val);
   dst = dst / max_val;
   
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function draws a vector of Ipoints
 * @param img Input/Output Image
 * @param dst Vector of keypoints
 */
void DrawKeyPoints(cv::Mat &img, const std::vector<cv::KeyPoint> &kpts)
{
    int x = 0, y = 0;
    float s = 0.0;
	
    for( unsigned int i = 0; i < kpts.size(); i++ )
	{
        x = kpts[i].pt.x;
        y = kpts[i].pt.y;
        s = kpts[i].size;
	
		// Draw a circle centered on the interest point
        cv::circle(img,cv::Point(x,y),s,CV_RGB(0,0,255),1);
        cv::circle(img,cv::Point(x,y),1.0,CV_RGB(0,255,0),-1);
	}
}

//*************************************************************************************
//*************************************************************************************

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
int SaveKeyPoints(char *sFileName, const std::vector<cv::KeyPoint> &kpts, const cv::Mat &desc, bool bVerbose)
{
   int length = 0, count = 0;
   float sc = 0.0;

   ofstream ipfile(sFileName);

   if( !ipfile )
   {
     cerr << "ERROR in loadIpoints(): "
          << "Couldn't open file '" << sFileName << "'!" << endl;
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
   for( int i = 0; i < count; i++)
   {
       // circular regions with diameter 2*scale x 2*scale
       sc = kpts[i].size/2.0;
       sc*=sc;

       ipfile  << kpts[i].pt.x /* x-location of the interest point */
                << " " << kpts[i].pt.y /* y-location of the interest point */
                << " " << 1.0/sc /* 1/r^2 */
                << " " << 0.0
                << " " << 1.0/sc; /* 1/r^2 */

      // Here comes the descriptor
      for( int j = 0; j < length; j++)
      {
          ipfile << " " << desc.at<float>(i,j);
      }

    ipfile << endl;
  }

  ipfile.close();

  // Write message to terminal.
  if( bVerbose == true )
  {
      cout << count << " interest points found" << endl;
  }

  return 1;
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This funtion rounds float to nearest integer
 * @param flt Input float
 * @return dst Nearest integer
 */
int fRound(float flt)
{
  return (int)(flt+0.5);
}

//*******************************************************************************
//*******************************************************************************

/**
 * @brief Function for finding matches using the nearest neighbor distance ratio
 * @param kpts1 First list of keypoints
 * @param desc1 First list of descriptors
 * @param kpts2 Second list of keypoints
 * @param desc2 Second list of descriptors
 * @param matches Vector of putative matches
 * @param nndr Nearest neighbor distance ratio factor
 */
void findmatches_nndr(std::vector<cv::KeyPoint> &kpts1, cv::Mat &desc1,
                      std::vector<cv::KeyPoint> &kpts2, cv::Mat &desc2,
                      std::vector<cv::Point2f> &matches, float nndr)
{
    float dist = 0.0, mind = 0.0, last_mind = 0.0;
    int nkpts1 = 0, nkpts2 = 0, dsize = 0, mindex = -1;
    bool first = false;

    nkpts1 = desc1.rows;
    nkpts2 = desc2.rows;
    dsize = desc1.cols;

    for( int i = 0; i < nkpts1; i++ )
    {
        mind = 10000.0;
        last_mind = 10000.0;
        mindex = -1;

        for( int j = 0; j < nkpts2; j++ )
        {
            dist = Compute_Descriptor_Distance(desc1.ptr<float>(i),desc2.ptr<float>(j),dsize);

            if( dist < mind )
            {
                if( first == false )
                {	mind = dist;
                    mindex = j;
                    first = true;
                }
                else
                {
                    last_mind = mind;
                    mind = dist;
                    mindex = j;
                }
            }
            else if( dist < last_mind )
            {
                 last_mind = dist;
            }
        }

        if( mind < nndr*last_mind )
        {
            matches.push_back(kpts1[i].pt);
            matches.push_back(kpts2[mindex].pt);
        }
    }
}

//*******************************************************************************
//*******************************************************************************

/**
 * @brief Function for computing the distance between two descriptors
 * @param p1 First keypoint
 * @param p2 Second keypoint
 * @param best Maximum distance to skip some computations
 * @param dsize Size fo the descriptor vector
 * @return Euclidean distance between the two descriptors
 */
float Compute_Descriptor_Distance(float *d1, float *d2, int dsize)
{
   float dist = 0.0;

   for(int i = 0; i < dsize; i++ )
   {
       dist += pow(d1[i]-d2[i],2);
   }

   return dist;
}

//*******************************************************************************
//*******************************************************************************

/**
 * @brief This function computes the set of inliers by estimating the fundamental matrix
 * or a planar homography in a RANSAC procedure
 * @param matches Vector of putative matches
 * @param inliers Vector of inliers
 * @param error Maximum error in the homography or fundamental matrix estimation
 * @param use_fund Set to 1 in case you want to estimate a fundamental matrix, 0 in case
 * you want to estimate a homography
 */
void Compute_Inliers_RANSAC(const std::vector<cv::Point2f> &matches, std::vector<cv::Point2f> &inliers, float error, bool use_fund)
{
   std::vector<cv::Point2f> points1, points2;
   cv::Mat H = cv::Mat::zeros(3,3,CV_32F);
   int npoints = matches.size()/2;
   cv::Mat status = cv::Mat::zeros(npoints,1,CV_8UC1);

   for( int i = 0; i < matches.size(); i+=2 )
   {
        points1.push_back(matches[i]);
        points2.push_back(matches[i+1]);
   }

   if( use_fund == true )
   {
       H = cv::findFundamentalMat(points1,points2,CV_FM_RANSAC,error,0.99,status);
   }
   else
   {
       H = cv::findHomography(points1,points2,CV_RANSAC,error,status);
   }

   for( int i = 0; i < npoints; i++ )
   {
       if( status.at<unsigned char>(i) == 1 )
       {
           inliers.push_back(points1[i]);
           inliers.push_back(points2[i]);
       }
   }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes the set of inliers given a ground truth homography
 * @param matches Vector of putative matches
 * @param inliers Vector of inliers
 * @param error Maximum pixel location error with respect to ground truth
 * @param H Ground truth homography
 */
void Compute_Inliers_Homography(const std::vector<cv::Point2f> &matches,
                                std::vector<cv::Point2f> &inliers, float error, const cv::Mat &H)
{
    float x1 = 0.0, y1 = 0.0;
    float x2 = 0.0, y2 = 0.0;
    float x2m = 0.0, y2m = 0.0;
    float dist = 0.0, s = 0.0;

    inliers.clear();

    for(unsigned int i = 0; i < matches.size(); i+=2)
    {
        x1 = matches[i].x;
        y1 = matches[i].y;
        x2 = matches[i+1].x;
        y2 = matches[i+1].y;

        s = H.at<float>(2,0)*x1+H.at<float>(2,1)*y1+H.at<float>(2,2);
        x2m = (H.at<float>(0,0)*x1+H.at<float>(0,1)*y1+H.at<float>(0,2))/s;
        y2m = (H.at<float>(1,0)*x1+H.at<float>(1,1)*y1+H.at<float>(1,2))/s;
        dist = sqrt(pow(x2m-x2,2) + pow(y2m-y2,2));

        if( dist <= error )
        {
            inliers.push_back(matches[i]);
            inliers.push_back(matches[i+1]);
        }
    }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function creates a new image that displays the inliers for the
 * set of correspondeces between the two images
 * @param img1 First image
 * @param img2 Second image
 * @param img_com Composite image
 * @param inliers Vector of inliers
 */
void Composite_Image_with_Line(cv::Mat &img1, cv::Mat &img2, cv::Mat &img_com, const std::vector<cv::Point2f> &inliers)
{
    int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    unsigned char *ptr, *ptri;

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
    cv::Mat img_aux = cv::Mat(cv::Size(img1.cols,img1.rows),CV_8UC3);
    cv::resize(img2,img_aux,cv::Size(img1.cols,img1.rows),0,0,CV_INTER_LINEAR);

    for( int i = 0; i < img_com.rows; i++ )
    {
        ptr = img_com.ptr<unsigned char>(i);
        for( int j = 0; j < img_com.cols; j++ )
         {
             if( j < img1.cols )
             {
                 ptri = img1.ptr<unsigned char>(i);
                 ptr[3*j] = ptri[3*j];
                 ptr[3*j+1] = ptri[3*j+1];
                 ptr[3*j+2] = ptri[3*j+2];
             }
             else
             {
                 ptri = img_aux.ptr<unsigned char>(i);
                 ptr[3*j] = ptri[3*(j-img_aux.cols)];
                 ptr[3*j+1] = ptri[3*(j-img_aux.cols)+1];
                 ptr[3*j+2] = ptri[3*(j-img_aux.cols)+2];
             }
         }
    }

    for( int i = 0; i < (int) inliers.size(); i+= 2)
    {
         x1 = (int)(inliers[i].x+.5);
         y1 = (int)(inliers[i].y+.5);

         x2 = (int)(inliers[i+1].x*ufactor + img1.cols +.5);
         y2 = (int)(inliers[i+1].y*vfactor + .5);

         cv::line(img_com,cv::Point(x1,y1),cv::Point(x2,y2),CV_RGB(255,0,0),2);
    }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function creates a new image that displays the inliers for the
 * set of correspondeces between the two images
 * @param img1 First image
 * @param img2 Second image
 * @param img_com Composite image
 * @param inliers Vector of inliers
 * @param index Variable to choose a different color
 */
void Composite_Image_with_Line(cv::Mat &img1, cv::Mat &img2, cv::Mat &img_com, const std::vector<cv::Point2f> &inliers, int index)
{
    int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    unsigned char *ptr, *ptri;

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
    cv::Mat img_aux = cv::Mat(cv::Size(img1.cols,img1.rows),CV_8UC3);
    cv::resize(img2,img_aux,cv::Size(img1.cols,img1.rows),0,0,CV_INTER_LINEAR);

    for( int i = 0; i < img_com.rows; i++ )
    {
        ptr = img_com.ptr<unsigned char>(i);
        for( int j = 0; j < img_com.cols; j++ )
         {
             if( j < img1.cols )
             {
                 ptri = img1.ptr<unsigned char>(i);
                 ptr[3*j] = ptri[3*j];
                 ptr[3*j+1] = ptri[3*j+1];
                 ptr[3*j+2] = ptri[3*j+2];
             }
             else
             {
                 ptri = img_aux.ptr<unsigned char>(i);
                 ptr[3*j] = ptri[3*(j-img_aux.cols)];
                 ptr[3*j+1] = ptri[3*(j-img_aux.cols)+1];
                 ptr[3*j+2] = ptri[3*(j-img_aux.cols)+2];
             }
         }
    }

    for( int i = 0; i < (int) inliers.size(); i+= 2)
    {
         x1 = (int)(inliers[i].x+.5);
         y1 = (int)(inliers[i].y+.5);

         x2 = (int)(inliers[i+1].x*ufactor + img1.cols +.5);
         y2 = (int)(inliers[i+1].y*vfactor + .5);

         if( index == 0 )
         {
             cv::line(img_com,cv:: Point(x1,y1),cv::Point(x2,y2),CV_RGB(255,255,0),2);
         }
         else if( index == 1 )
         {
             cv::line(img_com,cv:: Point(x1,y1),cv::Point(x2,y2),CV_RGB(255,0,0),2);
         }
         else if( index == 2 )
         {
             cv::line(img_com,cv:: Point(x1,y1),cv::Point(x2,y2),CV_RGB(0,255,0),2);
         }
    }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief Function for reading the ground truth homography from a txt file
 * @param calib_file Name of the txt file that contains the ground truth data
 * @param HG Matrix to store the ground truth homography
 */
void Read_Homography(const char *hFile, cv::Mat &H1toN)
{
   float h11 = 0.0, h12 = 0.0, h13 = 0.0;
   float h21 = 0.0, h22 = 0.0, h23 = 0.0;
   float h31 = 0.0, h32 = 0.0, h33 = 0.0;
   int  tmp_buf_size = 256;
   char tmp_buf[tmp_buf_size];
   std::string tmp_string;

   // Allocate memory for the OpenCV matrices
   H1toN = cv::Mat::zeros(3,3,CV_32FC1);

   setlocale(LC_ALL,"C");

   std::string filename(hFile);

   std::ifstream infile;
   infile.exceptions ( std::ifstream::eofbit | std::ifstream::failbit | std::ifstream::badbit );
   infile.open(filename.c_str(), std::ifstream::in);

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
