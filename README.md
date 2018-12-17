# Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Author: Guang Yang

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## System and Software Requirements <br>
* Ubuntu 18.04/ Mac OS Mojave
* Python 3.6.6
* Numpy 1.15.4
* OpenCV 3.4.4
* Matplotlib 3.0.2

## Folders Explanation
**src** contains source code and processed images folders for the project. Check *Advanced_Lane_Lines_Project.ipynb* for details.

**camera_cal** contains chessboard pictures for calibration

**test_images** contains testing images labeled as test*.jpg

**output_images** contains the output from *pipeline()* function. Each image is labeled with detected area, as well as lane lines curvature and vehicle position

**output_video** contains output videos
