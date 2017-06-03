# CarND-P5 Vehicle Detection

## Description

**This my 5th project result of Udacity self-driving car nanodegree (CarND) term 1. It's required to train a vehicle/not-vehicle classifier, build a robust vehicle detection pipeline and generate video with bounding boxes of detected vehicles for a given video. I chose linear SVM as classifier and trained classifier using dataset from a combination of the GTI vehicle image database, the KITTI vision benchmark suite, and examples extracted from the project video itself.**

* Udacity self-driving car nanodegree (CarND) :

  https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013

* Vehicle dataset :

  https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip

* Non-vehicle dataset :

  https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip


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


Here's a sample result after my software pipeline processed in a series of video :

![alt text][image9]

* The utility function used in this project is in Python file "./utility.py".

* The utility function is tested in IPython notebook "./utility_test.ipynb".

* The classifier model is trained in IPython notebook "./train_model.ipynb".

* The vehicle detection pipeline and vehicle detected video generation is in IPython notebook "./vehicle_detection_final_v2.ipynb".


## Histogram of Oriented Gradients (HOG)

#### HOG feature extraction

The code for this step is contained in ./utility.py from line 8 to 18. I use skimage.feature.hog() in get_hog_features() function to extract HOG features. 

I started by reading in all the `vehicle` and `non-vehicle` images in the 3rd code cell of the IPython notebook "./vehicle_detection_final_v2.ipynb".  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

I then explored different color spaces and different `skimage.feature.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.feature.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG features visualization with parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image3]

#### HOG parameter choice

I tried various combinations of parameters and found that too small orient or too large pix_per_cell would lead to less HOG features that aren't easy to be distinguished. 

Here's hog feature visualization examples with large pix_per_cell=32 (orient=9, cell_per_block=2) and another example with small orient = 4 (pix_per_cell=8, cell_per_block=2) :

![alt text][image4]
![alt text][image5]

Finally, I chose orient =  9, pix_per_cell = 8, and cell_per_block = 2 as the final HOG parameters based on the HOG feature image complexity.

#### Classifier training

I trained a linear SVM using YCrCb color space with several combined features in the 5~9th code cell of the IPython notebook "./train_model.ipynb" :

1. spatial features with spatial_size (32,32)
2. histgram features with hist_bins = 32
3. YCrCb hog features with orient = 9, pix_per_cell = 8, cell_per_block = 2


### Sliding Window Search

#### Sliding window implementation

For efficient feature extraction in sliding window, HOG sub-sampling and prediction-making is implemented in find_cars_boxes() in ./utility.py. 

As in the 3rd code cell of the IPython notebook "./vehicle_detection_final_v2.ipynb", I tested window search for different scale (1.0, 1.5, 2.0, 2.5 ...) and draw the detected car boxes in different colors. 

Here's one example with multiple scale window search using find_cars_boxes:

![alt text][image6]


#### Examples of full pipeline and optimization

The full pipeline is implemented as find_cars() in vehicle_detector class in './utility.py'. It applies different scale window search, summing heatmap over subsequence frames recorded in 'heatmap_acc' deque and applying heatmap threashold to get reliable vehicle detection. 

Here's an example of vehicle detected bounding boxes with different scale window search, its heatmap after applying threshold and final detected result :

![alt text][image7]
![alt text][image8]
![alt text][image9]

The deque records 3 heatmaps for the subsequent 3 frames and heat threshold is optimized to be 8 that tried to minimize false positive and maxmize true positive.

---

### Video Implementation

#### Vehicle detection video after full pipeline process

Here's a [link to my video result](./project_video_v2_out.mp4)

#### Suppress false positive car detection

In python file './utility.py', a 'heatmap_acc' deque is maintained in vehicle_detector class to record subsequent frames. By summing heatmap from different scale window search in subsequent frames and applying higher heatmap threshold than threshold tested in single frame, false positives are suppressed.


---

### Discussion

In my implementation, bounding boxes are generally larger than the size vehicles detected. It may cause nearby vehicles to be detected as one bounding box. This could be further improved by fine tuning the scale of searching window.

Besides, there are splitted bounding boxes and false positives in my result video ocassionally. Some tracking and bounding boxes merging strategy may improve it. For examples, utilize vehicle position from previous frame to improve the confidence of vehicle detection in next frame or recording the bounding boxes size over subsequence frames and using them to merge bounding boxes in the following frames.





