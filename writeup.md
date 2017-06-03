# **Vehicle Detection Project**

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
[image6]: ./output_images/find_cars_boxes.png
[image7]: ./output_images/vehicle_detect_multi.png
[image8]: ./output_images/vehicle_detect_heat.png
[image9]: ./output_images/vehicle_detect_result.png


* The utility function used in this project is in Python file "./utility.py".

* The utility function test in IPython notebook "./utility_test.ipynb".

* The model training is in IPython notebook "./train_model.ipynb".

* The vehicle detection pipeline and vehicle detected video generation is in IPython notebook "./vehicle_detection_final_v2.ipynb".

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in ./utility.py from line 8 to 18. I use skimage.feature.hog() in get_hog_features() function to extract HOG features. 

I started by reading in all the `vehicle` and `non-vehicle` images in the 3rd code cell of the IPython notebook.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

I then explored different color spaces and different `skimage.feature.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG features visualization with parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found that too small orient or too large pix_per_cell would lead to less HOG features that aren't easy to be distinguished. 

Here's hog feature visualization examples with large pix_per_cell=32 (orient=9, cell_per_block=2) and another example with small orient = 4 (pix_per_cell=8, cell_per_block=2) :

![alt text][image4]
![alt text][image5]

Finally, I chose orient =9, pix_per_cell =8, and cell_per_block=2 as the final HOG parameters based on the HOG feature image complexity.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using YCrCb color space with several combined features in the 5~9th code cell of the IPython notebook "./train_model.ipynb" :

1. spatial features with spatial_size (32,32)
2. histgram features with hist_bins = 32
3. YCrCb hog features with orient = 9, pix_per_cell = 8, cell_per_block = 2


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

For efficient feature extraction in sliding window, HOG sub-sampling and prediction-making is implemented in find_cars_boxes() in ./utility.py. 

As in the 3rd code cell of the IPython notebook "./vehicle_detection_final_v2.ipynb", I test window search for different scale (1.0, 1.5, 2.0, 2.5 ...) and draw the detected car boxes in different color. 

Here's one example with multiple scale window searching using find_cars_boxes:

![alt text][image6]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The full pipeline is implemented as find_cars() in vehicle_detector class in './utility.py'. It applies different scale window searching, summing heatmap over subsequence frames recorded in 'heatmap_acc' deque and applying heatmap threashold to get reliable car detection. 

Here's an example of vehicle detected bounding boxes with different scale window searching, its heatmap after applying threshold and final detected result :

![alt text][image7]
![alt text][image8]
![alt text][image9]

The deque records 3 subsequence heatmap and heat threshold is optimized to be 8 that minimized false positive and maxmized true positive.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_v2_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In python file './utility.py', a 'heatmap_acc' deque is maintained in vehicle_detector class to record subsequence frames. By summing heatmap from different window search scale of find_cars_boxes() in subsequence frames and applying higher heatmap threshold than threshold tested in single frame, false positives are suppressed.


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In my implementation, bounding boxes are generally larger than the size vehicles detected. It may cause nearby vehicles to be detected as one bounding box. This could be further improved by fine tuning the scale of searching window.

Besides, there are splitted bounding boxes and false positives in my result video ocassionally. Some tracking and bounding boxes merging strategy may improve it. For examples, utilize vehicle position from previous frame to improve the confidence of vehicle detection in next frame or recording the bounding boxes size over subsequence frames and using them to merge bounding boxes in the following frames.

