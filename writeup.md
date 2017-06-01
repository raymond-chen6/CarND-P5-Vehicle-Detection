**Vehicle Detection Project** # #    sd      

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car1.png
[image2]: ./output_images/car_not_car2.png
[image3]: ./output_images/HOG1.png
[image4]: ./output_images/HOG_orient4.png
[image5]: ./output_images/HOG_pix_per_cell32.png
[image6]: ./output_images/sliding_windows1.png
[image7]: ./output_images/find_cars_boxes.png
[image8]: ./output_images/vehicle_detect.png
[image9]: ./output_images/vehicle_detect_heatmap.png
[image10]: ./output_images/vehicle_detect2.png
[image11]: ./output_images/vehicle_detect3.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 5th code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images in the 7th code cell of the IPython notebook.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `Gray` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found that too small orient or too large pix_per_cell would lead to less HOG features that aren't easy to be distinguished. 

This illustrated as the following examples : 
Here's an hog feature visualization example with small orient =4 (pix_per_cell=8, cell_per_block=2) and another example with large pix_per_cell=32 (orient=8, cell_per_block=2) :

![alt text][image4]
![alt text][image5]

Finally, I chose orient =8, pix_per_cell =8, and cell_per_block=2 as the final HOG parameters based on the HOG feature image complexity.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using LUV color space with several combined features in the 11th code cell of the IPython notebook:
1. spatial features with spatial_size (32,32)
2. histgram features with hist_bins = 32
3. gray hog features with orient = 8, pix_per_cell = 8, cell_per_block = 8


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

My first implementation of sliding window searching is in the 12th code cell of the IPython notebook. In the beginning, I calculate the span of the region to be searched. Then, compute the sliding step in pixel along x and y direction according to search window size (xy_window) and overlap level (xy_overlap). Finally, slide window from top left corner to right bottom of the entire span.

Here's an sliding window example with y = 400~640, xy_window = (128, 128), and xy_overlap = (0.85, 0.85)

![alt text][image6]


After that, I tried to implement HOG sub-sampling window searching and make prediction in find_cars_boxes function in the 18th code cell of the IPython notebook.

As in the 19th code cell of the IPython notebook, I test window search for different scale (0.8, 1.0, 1.2 ...) and draw the detected car boxes in different color. 

Here's one example of find_cars_boxes :

![alt text][image7]


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The full pipeline for a single image is implemented in vehicle_detect function in the 21th code cell of the IPython notebook. It uses various window scale in find_cars_boxes function to detect cars and using heatmap with threshold to get reliable car detection. The heat threshold is optimized to be 5 that minimized false positive and maxmized true positive.

Here's some examples of my full pipeline working :

![alt text][image8]
![alt text][image9]


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In order to reduce false positive car detection, I create heatmap using those car boxes detected from different window search scale of find_cars_boxes. Afterwards, boxes are obtained after applying heat threshold :

Here's one car detection boxes and heatmap example :

![alt text][image10]
![alt text][image11]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My implementation exhaustedly searches almost half plane of the image several times and its process time is a bit long. If some tracking algorithm is implemented, it can be improved by only searching the nearby region of previous detected car. 

Besides, my pipeline fails when cars are far away and there's only a few pixels can be recognized. I can use smaller scale window size to improve it but it would introduce many false positives because features in such a small window are too few to differentiate from cars and notcars. Nevertheless, some region of interest (only focus on road region) and vehicle tracking (utilize vehicle position from previous frame to improve the confidence of vehicle detection in next frame) might help to make it more robust.


