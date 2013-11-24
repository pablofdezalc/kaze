
## README - KAZE Features

Version: 1.6.0
Date: 23-11-2013

You can get the latest version of the code from github:
`https://github.com/pablofdezalc/kaze`

## CHANGELOG
Version: 1.6.0
Changes:
- Code style has been changed substantially to match portability with other libraries
- Matching is now performed using OpenCV BruteForce matching
- KAZE Features now uses by default Fast Explicit Diffusion (FED) for discretizing the
  nonlinear diffusion equation. See use_fed command line option.
  With FED, KAZE is much faster than using AOS

For more information about FED, please check:

1. **Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces**. Pablo F. Alcantarilla, J. Nuevo and Adrien Bartoli. _In British Machine Vision Conference (BMVC), Bristol, UK, September 2013._

2. **From box filtering to fast explicit diffusion**. S. Grewenig, J. Weickert, and A. Bruhn. _In Proceedings of the DAGM Symposium on Pattern Recognition, pages 533–542, 2010._

Version: 1.5.2
Changes:
- KAZE Features now support OpenMP parallelization. This improves speed
  with respect to previous version in 30%
- Boost library dependency has been removed
- Bug corrected: In the previous version, the Clipping_Descriptor function
  with 3 input parameters was missing. Thanks to Mario Maresca

Version: 1.5.1
Changes:
- Integrated modifications from 1.5 into the old Ipoint interface
  The old interface library is named kaze_features_1_5_1_old
- Corrected a bug in checking descriptors limits

Version: 1.5
Changes:

- Important: Remove of Ipoint class interface. Now the code uses cv::KeyPoint for
  computing features and cv::Mat to store the descriptors
- Remove of Check_Descriptors_Limits check. The check if a descriptor is out
  of the image is done now in Determinant_Hessian_Parallel
- Speeded-up computation of Scharr-kernel derivatives
- Use OpenCV functions for measuring time computations
- Bug corrected with the number of sublevels Do_Subpixel_Refinement

Version: 1.4
Changes:

- Bug corrected in Compute_K_Percentile. Thanks to Willem Sanberg
  Declaring the static array hist[nbins] has compilation problems with
  Microsoft Visual Studio Express 2010
  Now, the array memory is allocated in runtime
- G-SURF descriptor pattern added as a possible option for the descriptor
  G-SURF descriptor is a novel family of descriptors that measure blurring
  and detail enhancing information at a certain scale level. G-SURF has been
  introduced recently in Pablo F. Alcantarilla, Luis M. Bergasa and
  Andrew J. Davison, Gauge-SURF Descriptors, Image and Vision Computing 31(1), 2013
- In version 1.3 the Check_Maximum_Neighbourhood was more strict in the
  sense that the detector response should be higher than the neighbors response
  plus a constant value. This produced in general less detected keypoints for
  some images. I modified the code back to the Check_Maximum_Neighbourhood from
  version 1.2 or below
- Bug corrected with the sample step and pattern size in SURF extended descriptor

Version: 1.3
Changes:

- Modifications of the CMakeLists.txt and code structure
- Several code improvements that make the algorithm faster. Thanks to Jesús Nuevo
- Small bug corrected with the sample step and pattern size in SURF descriptor

Version: 1.2
Changes:

- Small improvement in speed due to the use of pointer-based access instead of .at method in OpenCV Mat. Thanks to Martín Peris
- Bug in the default initialization of the descriptor has been corrected. Thanks to José Javier Yebes
- Bug in Subpixel Refinement has been corrected

Version: 1.1 
Changes:

- The code has been cleaned up a bit and some functions that were not used were removed
- Added the option for extended descriptors. Now you can choose between 64 or 128 descriptors
- Results visualization can be turned on/off from command line with the show_results option
- kaze_compare program added that compares KAZE against SIFT, SURF (OpenCV)
- Now kaze_match and kaze_compare can work with images that have different resolution between them

## What is this file?

This file explains how to make use of source code for computing KAZE features
and two practical image matching applications.

## Library Dependencies

The code is mainly based on the **OpenCV** library using the C++ interface.

In order to compile the code, the following libraries to be installed on your system:
- **OpenCV** version 2.4.0 or higher
- **Cmake** version 2.6 or higher

If you want to use **OpenMP** parallelization you will need to install OpenMP in your system
In Linux you can do this by installing the **gomp** library

- Since version 1.5.2 KAZE features does not use **Boost**

You will also need **doxygen** in case you need to generate the documentation

Tested compilers
- GCC 4.2-4.7

Tested systems:
- Ubuntu 11.10, 12.04, 12.10
- Kubuntu 10.04
- Mac OS 10.6.8

## Getting Started

Compiling:
1. `$ mkdir build`
2. `$ cd build>`
3. `$ cmake ..`
4. `$ make`

Additionally you can also install the library in `/usr/local/kaze/lib` by typing:
`$ sudo make install`

If the compilation is successful you should see three executables in the folder bin:
- `kaze_features`
- `kaze_match`
- `kaze_compare`

Additionally, the library `libKAZE[.a, .lib]` will be created in the `lib` folder.

If there is any error in the compilation, perhaps some libraries are missing.
Please check the Library dependencies section.

Examples:
To see how the code works, examine the two examples provided.

## Documentation
In the working folder type:
`doxygen`

The documentation will be generated in the `doc` folder.

## Computing KAZE Features

For running the program you need to type in the command line the following arguments:
`./kaze_features img.jpg [options]`

The options are not mandatory. In case you do not specify additional options, default arguments will be
used. Here is a description of the additional options:

- `--verbose` if verbosity is required
- `--help` for showing the command line options
- `--soffset` the base scale offset (sigma units)
- `--omax` the coarsest nonlinear scale space level (sigma units)
- `--nsublevels` number of sublevels per octave
- `--dthreshold` Feature detector threshold response for accepting points
- `--descriptor` Descriptor Type 0 -> SURF, 1 -> M-SURF, 2 -> G-SURF
- `--use_fed` 0 -> AOS, 1 -> FED
- `--upright` 0 -> Rotation Invariant, 1 -> No Rotation Invariant
- `--extended 0` -> Normal Descriptor (64), 1 -> Extended Descriptor (128)
- `--show_results` 1 in case we want to show detection results. 0 otherwise

Important Things:
- Check config.h in case you would like to change the value of some default settings
- The k constrast factor is computed as the 70% percentile of the gradient histogram of a
smoothed version of the original image. Normally, this empirical value gives good results, but
depending on the input image the diffusion will not be good enough. Therefore I highly
recommend you to visualize the output images from save_scale_space and test with other k
factors if the results are not satisfactory

## Image Matching Example with KAZE Features

The code contains one program to perform image matching between two images.
If the ground truth transformation is not provided, the program estimates a fundamental matrix using
RANSAC between the set of correspondences between the two images.

For running the program you need to type in the command line the following arguments:
`./kaze_match img1.jpg img2.pgm homography.txt [options]`

The datasets folder contains the **Iguazu** dataset described in the paper and additional datasets from Mykolajczyk et al. evaluation.
The **Iguazu** dataset was generated by adding Gaussian noise of increasing standard deviation.

For example, with the default configuration parameters used in the current code version you should get
the following results:

```
./kaze_match ../../datasets/iguazu/img1.pgm
              ../../datasets/iguazu/img4.pgm
              ../../datasets/iguazu/H1to4p
```

```
Number of Keypoints Image 1: 1902
Number of Keypoints Image 2: 1951
KAZE Features Extraction Time (ms): 1593.48
Matching Descriptors Time (ms): 37.6806
Number of Matches: 869
Number of Inliers: 842
Number of Outliers: 27
Inliers Ratio: 96.893
```

## Image Matching Comparison between KAZE, SIFT and SURF (OpenCV)

The code contains one program to perform image matching between two images, showing a comparison between KAZE features, SIFT
and SURF. All these implementations are based on the OpenCV library. 

The program assumes that the ground truth transformation is provided

For running the program you need to type in the command line the following arguments:
`./kaze_compare img1.jpg img2.pgm homography.txt [options]`

For example, running kaze_compare with the first and third images from the boat dataset you should get the following results:

```
./kaze_compare ../../datasets/boat/img1.pgm
               ../../datasets/boat/img3.pgm
               ../../datasets/boat/H1to3p
```
```
SIFT Results
**************************************
Number of Keypoints Image 1: 2000
Number of Keypoints Image 2: 2000
Number of Matches: 584
Number of Inliers: 575
Number of Outliers: 9
Inliers Ratio: 98.4589
SIFT Features Extraction Time (ms): 964.85

SURF Results
**************************************
Number of Keypoints Image 1: 4021
Number of Keypoints Image 2: 3162
Number of Matches: 301
Number of Inliers: 264
Number of Outliers: 37
Inliers Ratio: 87.7076
SURF Features Extraction Time (ms): 373.713

KAZE Results
**************************************
Number of Keypoints Image 1: 5170
Number of Keypoints Image 2: 4375
Number of Matches: 1309
Number of Inliers: 1246
Number of Outliers: 63
Inliers Ratio: 95.1872
KAZE Features Extraction Time (ms): 2254.6
```

One of the interesting reasons why you should use KAZE features is because is open source and you can use that freely even in commercial applications, which is not the case
of SIFT and SURF. The code is released under the BSD license. In general, KAZE results are superior to the other OpenCV methods (in terms of number of inliers and ratio), while being more slower to compute.
Future work will try to speed-up the process as much as possible while keeping good performance

## Citation
If you use this code as part of your work, please cite the following paper:

1. **KAZE Features**. Pablo F. Alcantarilla, Adrien Bartoli and Andrew J. Davison. _In European Conference on Computer Vision (ECCV), Fiorenze, Italy. October 2012_.

## Contact Info

**Important**: If you work in a research institution, university, company or you are a freelance and you are using KAZE or A-KAZE in your work, please send me an email!!
I would like to know the people that are using KAZE around the world!!"

In case you have any question, find any bug in the code or want to share some improvements,
please contact me:

Pablo F. Alcantarilla
email: pablofdezalc@gmail.com
