
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

// Some image matching options
const bool COMPUTE_HOMOGRAPHY = false;	// 0->Use ground truth homography, 1->Estimate homography with RANSAC
const float MAX_H_ERROR = 5.0;	// Maximum error in pixels to accept an inlier
const float DRATIO = .60;		// NNDR Matching value

//*************************************************************************************
//*************************************************************************************

/** Main Function 																	 */
int main( int argc, char *argv[] )
{
	// Variables
    toptions options;
	Mat img1, img1_32, img2, img2_32, img1_rgb, img2_rgb, img_com, img_r;
    char img_name1[NMAX_CHAR], img_name2[NMAX_CHAR], hfile[NMAX_CHAR];
    char rfile[NMAX_CHAR];
    float ratio = 0.0, rfactor = .90;
    std::vector<cv::KeyPoint> kpts1, kpts2;
    cv::Mat desc1, desc2, H;
    int nkpts1 = 0, nkpts2 = 0, nmatches = 0, ninliers = 0, noutliers = 0;

	// Variables for measuring computation times
    float t1 = 0.0, t2 = 0.0, tkaze = 0.0, tmatch = 0.0, thomo = 0.0;
	
	// Parse the input command line options
	if( Parse_Input_Options(options,img_name1,img_name2,hfile,rfile,argc,argv) )
	{
		return -1;
	}
	
	// Read the image, force to be grey scale
	img1 = imread(img_name1,0);
	
	if( img1.data == NULL )
	{
		cout << "Error loading image: " << img_name1 << endl;
		return -1;
	}
	
	// Read the image, force to be grey scale
	img2 = imread(img_name2,0);
	
	if( img2.data == NULL )
	{
		cout << "Error loading image: " << img_name2 << endl;
		return -1;
	}	
	
	// Convert the images to float
	img1.convertTo(img1_32,CV_32F,1.0/255.0,0);
	img2.convertTo(img2_32,CV_32F,1.0/255.0,0);
	
	// Color images for results visualization
	img1_rgb = cv::Mat(Size(img1.cols,img1.rows),CV_8UC3);	
	img2_rgb = cv::Mat(Size(img2.cols,img1.rows),CV_8UC3);	
	img_com = cv::Mat(Size(img1.cols*2,img1.rows),CV_8UC3);	
	img_r = cv::Mat(Size(img_com.cols*rfactor,img_com.rows*rfactor),CV_8UC3);	
	
    // Read the homography file
    Read_Homography(hfile,H);
	
    // Create the first KAZE object
    options.img_width = img1.cols;
    options.img_height = img1.rows;
    KAZE evolution1(options);

    t1 = cv::getTickCount();
	
	// Create the nonlinear scale space 
	// and perform feature detection and description for image 1
	evolution1.Create_Nonlinear_Scale_Space(img1_32);
    evolution1.Feature_Detection(kpts1);
    evolution1.Feature_Description(kpts1,desc1);

    // Create the second KAZE object
    options.img_width = img2.cols;
    options.img_height = img2.rows;
    KAZE evolution2(options);
    evolution2.Set_Detector_Threshold(options.dthreshold2);

    evolution2.Create_Nonlinear_Scale_Space(img2_32);
    evolution2.Feature_Detection(kpts2);
    evolution2.Feature_Description(kpts2,desc2);
	 
    t2 = cv::getTickCount();
    tkaze = 1000.0*(t2-t1) / cv::getTickFrequency();

    nkpts1 = kpts1.size();
    nkpts2 = kpts2.size();

    // Matching Descriptors!!
    std::vector<cv::Point2f> matches, inliers;
    t1 = cv::getTickCount();

    findmatches_nndr(kpts1,desc1,kpts2,desc2,matches,DRATIO);

    t2 = cv::getTickCount();
    tmatch = 1000.0*(t2-t1) / cv::getTickFrequency();

    // Compute Inliers!!
    t1 = cv::getTickCount();
	
	if( COMPUTE_HOMOGRAPHY == false )
	{
        Compute_Inliers_Homography(matches,inliers,MAX_H_ERROR,H);
	}
	else
	{
        Compute_Inliers_RANSAC(matches,inliers,MAX_H_ERROR,false);
	}
	
    t2 = cv::getTickCount();
    thomo = 1000.0*(t2-t1) / cv::getTickFrequency();

	// Compute the inliers statistics
    nmatches = matches.size()/2;
	ninliers = inliers.size()/2;
	noutliers = nmatches - ninliers;
	ratio = 100.0*((float) ninliers / (float) nmatches);
	
    // Prepare the visualization
	cvtColor(img1,img1_rgb,CV_GRAY2BGR);
	cvtColor(img2,img2_rgb,CV_GRAY2BGR);
	
	// Draw the list of detected points
    DrawKeyPoints(img1_rgb,kpts1);
    DrawKeyPoints(img2_rgb,kpts2);

	// Create the new image with a line showing the correspondences
    Composite_Image_with_Line(img1_rgb,img2_rgb,img_com,inliers);
	cv::resize(img_com,img_r,cv::Size(img_r.cols,img_r.rows),0,0,CV_INTER_LINEAR);
   
	// Show matching statistics
    if( options.show_results == true )
	{
        cout << "Number of Keypoints Image 1: " << nkpts1 << endl;
        cout << "Number of Keypoints Image 2: " << nkpts2 << endl;
		cout << "KAZE Features Extraction Time (ms): " << tkaze << endl;
		cout << "Matching Descriptors Time (ms): " << tmatch << endl;
		cout << "Homography Computation Time (ms): " << thomo << endl;
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
void Save_Matching_Image(cv::Mat img)
{
   char cad[NMAX_CHAR];
   sprintf(cad,"./output/images/image_matching.jpg");
   imwrite(cad,img);
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
int Parse_Input_Options(toptions &options, char *img_name1, char *img_name2, char *hom, 
						char *kfile, int argc, char *argv[] )
{
	// If there is only one argument return
	if( argc == 1 )
	{
		Show_Input_Options_Help();			
		return -1;
	}
	// Set the options from the command line
	else if( argc >= 2 )
	{
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
		options.verbosity = DEFAULT_VERBOSITY;
        options.show_results = DEFAULT_SHOW_RESULTS;

		strcpy(img_name1,argv[1]);
		strcpy(img_name2,argv[2]);
		strcpy(hom,argv[3]);
		strcpy(kfile,"./results.txt");

		for( int i = 1; i < argc; i++ )
		{
			if( !strcmp(argv[i],"--soffset") )
			{
				i = i+1;
				if( i >= argc )
				{
					cout << "Error introducing input options!!" << endl;
					return -1;
				}
				else
				{
					options.soffset = atof(argv[i]);
				}
			}
			else if( !strcmp(argv[i],"--omax") )
			{
				i = i+1;
				if( i >= argc )
				{
					cout << "Error introducing input options!!" << endl;
					return -1;
				}
				else
				{
					options.omax = atof(argv[i]);
				}
			}
			else if( !strcmp(argv[i],"--dthreshold") )
			{
				i = i+1;
				if( i >= argc )
				{
					cout << "Error introducing input options!!" << endl;
					return -1;
				}
				else
				{
					options.dthreshold = atof(argv[i]);
				}
			}
			else if( !strcmp(argv[i],"--dthreshold2") )
			{
				i = i+1;
				if( i >= argc )
				{
					cout << "Error introducing input options!!" << endl;
					return -1;
				}
				else
				{
					options.dthreshold2 = atof(argv[i]);
				}
			}
			else if( !strcmp(argv[i],"--sderivatives") )
			{
				i = i+1;
				if( i >= argc )
				{
					cout << "Error introducing input options!!" << endl;
					return -1;
				}
				else
				{
					options.sderivatives = atof(argv[i]);
				}
			}	
			else if( !strcmp(argv[i],"--nsublevels") )
			{
				i = i+1;
				if( i >= argc )
				{
					cout << "Error introducing input options!!" << endl;
					return -1;
				}
				else
				{
					options.nsublevels = atoi(argv[i]);
				}
			}
			else if( !strcmp(argv[i],"--diffusivity") )
			{
				i = i+1;
				if( i >= argc )
				{
					cout << "Error introducing input options!!" << endl;
					return -1;
				}
				else
				{
					options.diffusivity = atoi(argv[i]);
                    if( options.diffusivity > 2 || options.diffusivity < 0 )
					{
						options.diffusivity = DEFAULT_DIFFUSIVITY_TYPE;
					}
				}
			}
			else if( !strcmp(argv[i],"--descriptor") )
			{
				i = i+1;
				if( i >= argc )
				{
					cout << "Error introducing input options!!" << endl;
					return -1;
				}
				else
				{
					options.descriptor = atoi(argv[i]);
					
                    if( options.descriptor > 2 || options.descriptor < 0 )
					{
						options.descriptor = DEFAULT_DESCRIPTOR_MODE;
					}
				}
			}
			else if( !strcmp(argv[i],"--save_scale_space") )
			{
				i = i+1;
				if( i >= argc )
				{
					cout << "Error introducing input options!!" << endl;
					return -1;
				}
				else
				{
					options.save_scale_space = (bool)atoi(argv[i]);
				}
			}
            else if( !strcmp(argv[i],"--show_results") )
            {
                i = i+1;
                if( i >= argc )
                {
                    cout << "Error introducing input options!!" << endl;
                    return -1;
                }
                else
                {
                    options.show_results = (bool)atoi(argv[i]);
                }
            }
			else if( !strcmp(argv[i],"--kfile") )
			{
				i = i+1;
				if( i >= argc )
				{
					cout << "Error introducing input options!!" << endl;
					return -1;
				}
				else
				{
					strcpy(kfile,argv[i]);
				}
			}
			else if( !strcmp(argv[i],"--upright") )
			{
				i = i+1;
				if( i >= argc )
				{
					cout << "Error introducing input options!!" << endl;
					return -1;
				}
				else
				{
					options.upright = (bool)atoi(argv[i]);
				}
			}
            else if( !strcmp(argv[i],"--extended") )
            {
                i = i+1;
                if( i >= argc )
                {
                    cout << "Error introducing input options!!" << endl;
                    return -1;
                }
                else
                {
                    options.extended = (bool)atoi(argv[i]);
                }
            }
			else if( !strcmp(argv[i],"--verbose") )
			{
				options.verbosity = true;
			}
			else if( !strcmp(argv[i],"--help") )
            {
				// Show the help!!
				Show_Input_Options_Help();
				return -1;
			}
		}			
	}
	else
	{
		cout << "Error introducing input options!!" << endl;
		
		// Show the help!!
		Show_Input_Options_Help();
		return -1;
	}
	
	return 0;
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function shows the possible command line configuration options
 */
void Show_Input_Options_Help(void)
{
	fflush(stdout);
	
	cout << "KAZE Features" << endl;
	cout << "************************************************" << endl;
	cout << "For running the program you need to type in the command line the following arguments: " << endl;
	cout << "./kaze_match img1.jpg img2.pgm homography.txt options" << endl;
	cout << "The options are not mandatory. In case you do not specify additional options, default arguments will be used" << endl << endl;
	cout << "Here is a description of the additional options: " << endl;
	cout << "--verbose " << "\t\t if verbosity is required" << endl;
	cout << "--help" << "\t\t for showing the command line options" << endl;
    cout << "--soffset" << "\t\t the base scale offset (sigma units)" << endl;
    cout << "--omax" << "\t\t maximum octave evolution of the image 2^sigma (coarsest scale)" << endl;
    cout << "--nsublevels" << "\t\t number of sublevels per octave" << endl;
    cout << "--dthreshold" << "\t\t Feature detector threshold response for accepting points (0.001 can be a good value)" << endl;
    cout << "--sderivatives" << "\t\t Standard deviation for the Gaussian derivatives in the nonlinear diffusion filtering" << endl;
    cout << "--descriptor" << "\t\t Descriptor Type 0 -> SURF, 1 -> M-SURF, 2 -> G-SURF" << endl;
    cout << "--upright" << "\t\t 0 -> Rotation Invariant, 1 -> No Rotation Invariant" << endl;
    cout << "--extended" << "\t\t 0 -> Normal Descriptor (64), 1 -> Extended Descriptor (128)" << endl;
    cout << "--show_results" << "\t\t 1 in case we want to show detection results. 0 otherwise" << endl;
	cout << endl;
}
