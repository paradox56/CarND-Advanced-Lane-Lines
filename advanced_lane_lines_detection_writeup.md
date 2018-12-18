# **Advanced Lane Finding Project**

Author: Guang Yang

The goals of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Camera Calibration

Each lens has its unique lens distortion based on its parameters. The most common distortions include radial distortion and tangential distortion. Since we will be using camera to track lane lines, as well as determine curvature of the road, it becomes critical for us to remove these optical distortions. Based on the documentation from [OpenCV](https://docs.opencv.org/3.4.0/d4/d94/tutorial_camera_calibration.html), the radial distortion can be modeled with the following function:<br/>
$$ x_{rad distort} = x(1+k_1r^2+k_2r^4+k_3r^6) $$
$$ y_{rad distort} = y(1+k_1r^2+k_2r^4+k_3r^6) $$

and the tangential distortion is modeled as the following: <br />
$$x_{tan distort} = x+[2p_1xy+p_2(r^2+2x^2)]$$
$$y_{tan distort} = y+[p_1(r^2+2y^2)+2p_2xy]$$

Therefore, we can define the distortion coefficients $d$ as: <br />
$$d = [k_1, k_2, p_1, p_2, k_3]$$

The camera matrix is defined as:
$$\mathbf{C} = \left[\begin{array}
{rrr}
f_x & 2 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{array}\right],
$$
where $f_x, f_y$ is the focal length of the camera lens and $c_x, c_y$ is the optical center. To optain camera matrix $\mathbf{C}$ and distortion coefficient $d$, we can utilize cv2.calibrateCamera( ) function from OpenCV. We can now transform the original chessboard image to the undistorted image:

<img src="https://raw.githubusercontent.com/paradox56/CarND-Advanced-Lane-Lines/master/src/calibrated_images/original_2.jpg" width="480" height="270" /><img src="https://raw.githubusercontent.com/paradox56/CarND-Advanced-Lane-Lines/master/src/calibrated_images/undistorted_2.jpg" width="480" height="270" />

The code is shown as the following:

```python
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
%matplotlib qt

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.
d = 0
# Make a list of calibration images
images = glob.glob('../camera_cal/calibration*.jpg')

def camera_calibration_matrix(objpoints, imgpoints, img):
    #This function takes chessboard images with known dimensions and obtain camera calibration matrix
    localImg = np.copy(img)
    grayImg = cv2.cvtColor(localImg,cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tves = cv2.calibrateCamera(objpoints, imgpoints, grayImg.shape[::-1],None,None)    
    return mtx,dist


# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(grayImg, (9,6),None)


    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        mtx,dist = camera_calibration_matrix(objpoints, imgpoints, img)

        # Undistort images
        undistortedImg = cv2.undistort(img,mtx,dist, None, mtx)
        # Draw and display the corners
        originalImg_with_chessboard = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        undistortedImg_with_chessboard = cv2.drawChessboardCorners(undistortedImg, (9,6), corners, ret)

        outputImgPath = "./calibrated_images/"
        cv2.imwrite(outputImgPath+"original_%d.jpg"%d,originalImg_with_chessboard)
        cv2.imwrite(outputImgPath+"undistorted_%d.jpg"%d,undistortedImg_with_chessboard)
        d+=1

np.savetxt("camera_calibration_matrix.csv",mtx, delimiter=",")
np.savetxt("camera_distortion_coefficients.csv",dist, delimiter=",")
```

<script src="https://gist.github.com/paradox56/e7c2583029f0664cae66df6eb120a3ed.js"></script>

Through this process, we can also obtain both camera calibration matrix $mtx$ and distortion coefficient $dist$, which will be used later.

## Image Distortion Correction
Once we obtain the camera matrix and distortion coefficients, we will use it for distortion correction for lane line images. To demonstrate this process, I will describe how I apply the distortion correction to one of the test images like this one:

<img src="https://raw.githubusercontent.com/paradox56/CarND-Advanced-Lane-Lines/master/test_images/test2.jpg" width="480" height="270" /><img src="https://raw.githubusercontent.com/paradox56/CarND-Advanced-Lane-Lines/master/src/calibrated_images/undistorted_image_example.jpg" width="480" height="270" />


## Gradient and Color Threshold
### Sobel Opeartor for gradient measurements
The Sobel Opeartor are used to perform convolution on the original image to determine how gradient changes. Intuitvely, we want to measure the change of gradient with respect to both x axis and y axis, as well as direction of gradient.

In this project, I use $3 \times 3$ Scharr filters (set ksize = -1) kernels, namely $\mathbf{S_x}$ and $\mathbf{S_y}$ for the two axises. <br />
Gradient along x axis:
$$\mathbf{S_x} = \left[\begin{array}
{rrr}
-3 & 0& 3\\
-10 & 0 & 10\\
3 & 0 & 3
\end{array}\right],
$$
Gradient along y axis:
$$\mathbf{S_y} = \left[\begin{array}
{rrr}
-3 & -10 & -3 \\
0 & 0 & 0 \\
3 & 10 & 3
\end{array}\right].
$$

The magnitude $\mathbf{S}$ and direction $\theta$ of the gradient can be easily obtained through trigonometry:<br />

$$\mathbf{S} = \sqrt{\mathbf{S_x}^2+\mathbf{S_y}^2}$$
$$\theta = atan(\frac{\mathbf{S_y}}{\mathbf{S_x}})$$

The Sobel operator code is the following:

```python
def abs_sobel_thresh(img, orient='x', sobel_kernel=-1, thresh=(60, 150)):
    # Calculate directional gradient
    # Apply threshold
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if orient =='x':
        sobelDir = cv2.Sobel(grayImg, cv2.CV_64F, 1,0, ksize=sobel_kernel) #x oritentaion
    elif orient =='y':
        sobelDir = cv2.Sobel(grayImg, cv2.CV_64F, 0,1, ksize=sobel_kernel) #y orientation

    absSobelDir = np.absolute(sobelDir)
    scaled_sobel = np.uint8(255*absSobelDir/np.max(sobelDir))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return grad_binary

def mag_thresh(image, sobel_kernel=-1, mag_thresh=(50, 150)):
    # Calculate gradient magnitude
    # Apply threshold
    thresh_min = mag_thresh[0]
    thresh_max = mag_thresh[1]
    grayImg = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(grayImg, cv2.CV_64F, 1,0,ksize=sobel_kernel) #x oritentaion
    sobely = cv2.Sobel(grayImg, cv2.CV_64F, 0,1,ksize=sobel_kernel) #y orientation
    absSobelxy = np.sqrt(sobelx**2+sobely**2)
    scaled_sobel = np.uint8(255*absSobelxy/np.max(absSobelxy))
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return mag_binary

def dir_threshold(image, sobel_kernel=-1, thresh=(0.5, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    grayImg = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(grayImg, cv2.CV_64F, 1,0,ksize=sobel_kernel) #x oritentaion
    sobely = cv2.Sobel(grayImg, cv2.CV_64F, 0,1,ksize=sobel_kernel) #y orientation
    gradientDirectionImg = np.arctan2(np.absolute(sobely), np.absolute(sobelx));
    dir_binary = np.zeros_like(gradientDirectionImg)
    dir_binary[(gradientDirectionImg >= thresh_min) & (gradientDirectionImg <= thresh_max)] = 1
return dir_binary
```
<script src="https://gist.github.com/paradox56/ce1292714bbd334349c1c907b0b95af3.js"></script>

Here is a comparison of original image and the image after applying Sobel thresholding:

<img src="https://raw.githubusercontent.com/paradox56/CarND-Advanced-Lane-Lines/master/test_images/test2.jpg" width="480" height="270" /><img src="https://raw.githubusercontent.com/paradox56/CarND-Advanced-Lane-Lines/master/src/combined_sobel_images/combined_sobel_images4.jpg" width="480" height="270" />


### Color Thresholding
Another thresoholding techqniue is used in color space. Intuitively, we want to extract the color of interest from the image that resembles the lane line (In this case, yellow and white colors) The standard RGB color space is a three dimensional vector space with Red, Green and Blue for each axis. In theory, we can directly perform color thresholding in RGB color space. However, the surrounding light can change dramatically in real-life situation, which can lead to poor performance and various of other issues. Alternatively, we can represent an image in Hue, Saturation and Value (HSV) color space or Hue, Lightness and Saturation (HLS) color space. Why do we want to perform color thresholding in those color spaces? Well, the use of Hue and Saturation are critical because they are indepentdent of brightness.

For the project, I decide to use HLS color space for color thresholding. To convert the image from RGB to HLS, I use the OpenCV function cv2.cvtColor(im, cv2.COLOR_RGB2HLS). After some testings, the saturation channel (S channel) performs the best in terms of extracting lane line, but I will combine it with Hue thresholding to get more degree of freedom. The following code demonstrate how a binary image is generated through S and H channel thresholding.

The code is the following:
```python
def color_threshold(imgage, S_thresh=(0, 255),H_thresh=(0, 255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    H = hls[:,:,0]
    S = hls[:,:,2]
    binary_output = np.zeros_like(S)
    binary_output[(S > S_thresh[0]) & (S <= S_thresh[1])& (H > H_thresh[0])& (H <= H_thresh[1])] = 1
return binary_output
```
<script src="https://gist.github.com/paradox56/3c28869d98882873fdeef6b3c2bf5f93.js"></script>

Here is an example of applying color thresholding:

<img src="https://raw.githubusercontent.com/paradox56/CarND-Advanced-Lane-Lines/master/test_images/test2.jpg" width="480" height="270" /><img src="https://raw.githubusercontent.com/paradox56/CarND-Advanced-Lane-Lines/master/src/saturation_thresh_images/saturation_thresh_images4.jpg" width="480" height="270" />

By combining both gradient and color threshold together, we can achieve more degree of freedom to filter out unwanted background:
```python

def threshold(image,ksize, abs_sobel_thresh_param_x, abs_sobel_thresh_param_y ,mag_thresh_param, dir_thresh_param,saturation_thresh_param, hue_thresh_param):
    # Convert original RGB image to Gray Image
    grayImg = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Gradient Threshold
    gradx = abs_sobel_thresh(grayImg, orient='x', sobel_kernel=ksize, thresh=abs_sobel_thresh_param_x)
    grady = abs_sobel_thresh(grayImg, orient='y', sobel_kernel=ksize, thresh=abs_sobel_thresh_param_y)
    mag_binary = mag_thresh(grayImg, sobel_kernel=ksize, mag_thresh=mag_thresh_param)
    dir_binary = dir_threshold(grayImg, sobel_kernel=ksize, thresh=dir_thresh_param)
    combined_gradient_binary = np.zeros_like(dir_binary)
    combined_gradient_binary[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    # Color Threshold
    color_binary = color_threshold(image, S_thresh=saturation_thresh_param,H_thresh=hue_thresh_param)

    # Combine Gradient Threshold and Color Threshold
    image_after_threshold_binary = np.zeros_like(color_binary)
    image_after_threshold_binary[(combined_gradient_binary == 1)|(color_binary == 1)]=1
return image_after_threshold_binary
```
<script src="https://gist.github.com/paradox56/43f97bee60f5aa0527b6a919b44a567c.js"></script>

To further improve the result, I performed a masking on thresholded image by using the following code:
```python
def mask_image(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
return masked_image
```
<script src="https://gist.github.com/paradox56/45e1afe9707155923da32fbe4ca4878e.js"></script>


Here is a example of processed image using combined thresholding technique and image masking:

<img src="https://raw.githubusercontent.com/paradox56/CarND-Advanced-Lane-Lines/master/test_images/test2.jpg" width="480" height="270" /><img src="https://raw.githubusercontent.com/paradox56/CarND-Advanced-Lane-Lines/master/src/thresholded_images/thresholded_images4.jpg" width="480" height="270" />

### Perspective Transform
The code for my perspective transform includes a function called `perspective_transform(image)`, as shown below:
```python
def perspective_transform(image):
    if len(image.shape) > 2:
        height, width, chanels = image.shape
    else:
        height, width = image.shape

    src = np.array([[(width*0.1,height*0.9),
                    (width/2.3,height/1.6),
                    (width/1.7,height/1.6),
                    (width*0.9,height*0.9)]],dtype=np.float32)

    dst = np.array([[(0.2*width,height),
                    (0.2*width,0),
                    (0.8*width,0),
                    (0.8*width,height)]],dtype=np.float32)
    transform_matrix = cv2.getPerspectiveTransform(src,dst)
    top_down_image = cv2.warpPerspective(image, transform_matrix, (width,height), flags = cv2.INTER_LINEAR)
return top_down_image, transform_matrix
```
<script src="https://gist.github.com/paradox56/0b91bb59f730a091a1bc001d45e8e3c0.js"></script>
To make the code more general to cameras with different resolution, here is the source and destination chart with respect to image size (In this project, width = 1280, height = 720):

| Source        | Destination   |
|:-------------:|:-------------:|
| width $\times$ 0.1, height $\times$ 0.9      | 0.2 $\times$ width, height       |
| width/2.3,height/1.6      | 0.2 $\times$width, 0     |
| width/1.7,height/1.6     | 0.8 $\times$width, 0     |
| width $\times$ 0.9,height $\times$ 0.9      | 0.8 $\times$width, height        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

<img src="https://raw.githubusercontent.com/paradox56/CarND-Advanced-Lane-Lines/master/test_images/test1.jpg" width="480" height="270" /><img src="https://raw.githubusercontent.com/paradox56/CarND-Advanced-Lane-Lines/master/src/top_down_images/top_down_images6.jpg" width="480" height="270" />

### Lane Line Detection
The next step is to fit lane line using a polynomial as the following:
$$y = ax^2+bx+c$$
The fitting process is nothing more than finding the correct coefficients for the polynomial function. Before fitting the lane line, we need to process raw images with functions that we have defined above and output thresholded, masked, transformed binary images. We name these "warped_images". Here is an example of a warped image:

<img src="https://raw.githubusercontent.com/paradox56/CarND-Advanced-Lane-Lines/master/src/images_for_lane_lines_fitting/warped_images1.jpg"/>

One approach to detect lane lines is to count the number of pixels along the x-axis and then get a histogram from it.



#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
