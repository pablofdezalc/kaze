
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

// Namespaces
using namespace std;
using namespace cv;

//*************************************************************************************
//*************************************************************************************

/** Main Function 																	 */
int main( int argc, char *argv[] )
{
	// Variables
	toptions options;
	Mat img, img_32, img_rgb;
    char img_name[NMAX_CHAR], kfile[NMAX_CHAR];
	
	// Parse the input command line options
	if( Parse_Input_Options(options,img_name,kfile,argc,argv) )
	{
		return -1;
	}
	
	// Read the image, force to be grey scale
	img = imread(img_name,0);
	
	if( img.data == NULL )
	{
		cout << "Error loading image: " << img_name << endl;
		return -1;
	}
	
	// Convert the image to float
	img.convertTo(img_32,CV_32F,1.0/255.0,0);
	img_rgb = cv::Mat(Size(img.cols,img.rows),CV_8UC3);	
	
	options.img_width = img.cols;
	options.img_height = img.rows;

	// Create the KAZE object
	KAZE evolution(options);
	
	// Create the nonlinear scale space
	evolution.Create_Nonlinear_Scale_Space(img_32);

    std::vector<cv::KeyPoint> kpts;
    cv::Mat desc;

    evolution.Feature_Detection(kpts);
    evolution.Feature_Description(kpts,desc);
	
	// Save the nonlinear scale space images
	if( options.save_scale_space == true )
	{
        cout << "Saving" << endl;
		evolution.Save_Nonlinear_Scale_Space();
		evolution.Save_Flow_Responses();
	}
	
    if( options.show_results == true )
	{
        cout << "Time Scale Space: " << evolution.Get_Time_NLScale() << endl;
        cout << "Time Detector: " << evolution.Get_Time_Detector() << endl;
        cout << "Time Descriptor: " << evolution.Get_Time_Descriptor() << endl;
        cout << "Number of Keypoints: " << kpts.size() << endl;

		// Create the OpenCV window
		namedWindow("Image",CV_WINDOW_FREERATIO);		
		
		// Copy the input image to the color one
		cvtColor(img,img_rgb,CV_GRAY2BGR);
		
		// Draw the list of detected points
        DrawKeyPoints(img_rgb,kpts);

		imshow("Image",img_rgb);
		waitKey(0);
		
        imwrite("./kaze01.png",img_rgb);

		// Destroy the windows
		destroyAllWindows();

        if( options.save_scale_space == true )
        {
            // Copy the input image to the color one
            cvtColor(img,img_rgb,CV_GRAY2BGR);

            // Draw the list of detected points
            DrawKeyPoints(img_rgb,kpts);

            // Save the rgb image
            Save_Image_with_Features(img_rgb);
        }
    }

	// Save the list of keypoints
    if( options.save_keypoints == true )
	{
        SaveKeyPoints(kfile,kpts,desc,false);
    }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief  This function saves the input image with the detected features
 * @param img Image to be saved
 */
void Save_Image_with_Features(cv::Mat img)
{
   char cad[NMAX_CHAR];
   sprintf(cad,"./output/images/image_features.jpg");
   imwrite(cad,img);
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function parses the command line arguments for setting KAZE parameters
 * @param options Structure that contains KAZE settings
 * @param img_name Name of the input image
 * @param kfile Name of the file where the keypoints where be stored
 */
int Parse_Input_Options(toptions &options, char *img_name, char *kfile, int argc, char *argv[] )
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
        options.diffusivity = DEFAULT_DIFFUSIVITY_TYPE;
        options.descriptor = DEFAULT_DESCRIPTOR_MODE;
		options.upright = DEFAULT_UPRIGHT;
        options.extended = DEFAULT_EXTENDED;
		options.sderivatives = DEFAULT_SIGMA_SMOOTHING_DERIVATIVES;
		options.save_scale_space = DEFAULT_SAVE_SCALE_SPACE;
        options.show_results = DEFAULT_SHOW_RESULTS;
        options.save_keypoints = DEFAULT_SAVE_KEYPOINTS;
		strcpy(kfile,"./output/files/keypoints.txt");
		options.verbosity = DEFAULT_VERBOSITY;

        if( !strcmp(argv[1],"--help") )
        {
            Show_Input_Options_Help();
            return -1;
        }

        strcpy(img_name,argv[1]);
		
		for( int i = 2; i < argc; i++ )
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
			else if( !strcmp(argv[i],"--output") )
			{
				options.save_keypoints = true;
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
			else if( !strcmp(argv[i],"--help") )
			{
				// Show the help!!
				Show_Input_Options_Help();
				return -1;
			}
		}			
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
	cout << "***********************************************************" << endl;
	cout << "For running the program you need to type in the command line the following arguments: " << endl;
    cout << "./kaze_features img.jpg options" << endl;
	cout << "The options are not mandatory. In case you do not specify additional options, default arguments will be used" << endl << endl;
	cout << "Here is a description of the additional options: " << endl;
	cout << "--verbose " << "\t\t if verbosity is required" << endl;
	cout << "--help" << "\t\t for showing the command line options" << endl;
    cout << "--soffset" << "\t\t the base scale offset (sigma units)" << endl;
    cout << "--omax" << "\t\t maximum octave evolution of the image 2^sigma (coarsest scale)" << endl;
    cout << "--nsublevels" << "\t\t number of sublevels per octave" << endl;
    cout << "--dthreshold" << "\t\t Feature detector threshold response for accepting points (0.001 can be a good value)" << endl;
    cout << "--descriptor" << "\t\t Descriptor Type 0 -> SURF, 1 -> M-SURF, 2 -> G-SURF" << endl;
    cout << "--upright" << "\t\t 0 -> Rotation Invariant, 1 -> No Rotation Invariant" << endl;
    cout << "--extended" << "\t\t 0 -> Normal Descriptor (64), 1 -> Extended Descriptor (128)" << endl;
    cout << "--output keypoints.txt" << "\t\t For saving the detected keypoints into a .txt file" << endl;
    cout << "--save_scale_space" << "\t\t 1 in case we want to save the nonlinear scale space images. 0 otherwise" << endl;
    cout << "--show_results" << "\t\t 1 in case we want to show detection results. 0 otherwise" << endl;
	cout << endl;
}
