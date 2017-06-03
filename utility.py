import cv2
import matplotlib.image as mpimg
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt

import pickle
from scipy.ndimage.measurements import label
from collections import deque

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features


# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel() 
    color2 = cv2.resize(img[:,:,1], size).ravel() 
    color3 = cv2.resize(img[:,:,2], size).ravel()   
    return np.hstack((color1, color2, color3))

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

 # Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='YCrCb', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        img = mpimg.imread(file)
#        img_flip = cv2.flip(img,1)
        images = []
        images.append(img)
#        images.append(img_flip)
        for image in images:
            
            file_features = []    
            
            # apply color conversion if other than 'RGB'
            if color_space != 'RGB':
                if color_space == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif color_space == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif color_space == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif color_space == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                elif color_space == 'YCrCb':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            else: feature_image = np.copy(image)      

            #=== 1. bin_spatial feature ===
            if spatial_feat == True:
                spatial_features = bin_spatial(feature_image, size=spatial_size)
                file_features.append(spatial_features)

            #=== 2. color_hist feature ===
            if hist_feat == True:
                # Apply color_hist()
                hist_features = color_hist(feature_image, nbins=hist_bins)
                file_features.append(hist_features)

            #=== 3. hog_feat feature ===
            if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
                if hog_channel == 'ALL':
                    hog_features = []
                    for channel in range(feature_image.shape[2]):
                        hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                            orient, pix_per_cell, cell_per_block, 
                                            vis=False, feature_vec=True))
                    hog_features = np.ravel(hog_features)        
                else:
                    hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)                    
                    
                # Append the new feature vector to the features list
                file_features.append(hog_features)

            #=== 4. append this image file feature to features ===
            features.append(np.concatenate(file_features))
        
    # Return list of feature vectors
    return features

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars_boxes(img, ystart, ystop, xstart, xstop, scale, svc, X_scaler, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins):
    bboxes = []
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,xstart:xstop,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2'+color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]    

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    if hog_channel =='ALL':
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    else:
        hog1 = get_hog_features(hog_channel, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            if hog_channel =='ALL':
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale) + xstart
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
                bboxes.append( ( (int(xbox_left), int(ytop_draw+ystart)),(int(xbox_left+win_draw), int(ytop_draw+win_draw+ystart)) ) )
                
    return bboxes



def draw_boxes(img,bboxes,color=(0,0,255),thick=6):
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy


def show_image (img):
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

def show_2by3_images(img1, img2, img3, img4, img5, img6, txt1='Y', txt2='Cr', txt3='Cb', txt4='HOG_Y', txt5='HOG_Cr', txt6='HOG_Cb'):
    fig = plt.figure(figsize=(12,8))
    plt.subplot(231)
    plt.imshow(img1, cmap = 'gray')
    plt.title(txt1)
    plt.subplot(232)
    plt.imshow(img2, cmap = 'gray')
    plt.title(txt2)
    plt.subplot(233)
    plt.imshow(img3, cmap = 'gray')
    plt.title(txt3)
    plt.subplot(234)
    plt.imshow(img4, cmap = 'gray')
    plt.title(txt4)
    plt.subplot(235)
    plt.imshow(img5, cmap = 'gray')
    plt.title(txt5)
    plt.subplot(236)
    plt.imshow(img6, cmap = 'gray')
    plt.title(txt6)
    plt.show()

def add_heat(heatmap, bbox_list):
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        return heatmap
   
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    bboxes = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((int(np.min(nonzerox)), int(np.min(nonzeroy))), (int(np.max(nonzerox)), int(np.max(nonzeroy))))       
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        bboxes.append( (bbox[0], bbox[1]) )

    return bboxes


class vehicle_detector:
    def __init__(self, model_file):
        with open(model_file, 'rb') as pfile:
            pickle_data = pickle.load(pfile)
            for key in pickle_data:
                exec("self." + key + "= pickle_data['" + str(key) + "']")
        self.ystart_ystop_xstart_xstop_scale_color = [(380, 650 , 350, None, 1.0, (0,0,255) ),
                                                    (380, 650 , 350, None, 1.5, (0,255,0) ),
                                                    (380, 650 , 350, None, 2.0, (255,0,0) ),
                                                    (380, 650 , 350, None, 2.2, (255,255,0) ),
                                                    (380, 650 , 350, None, 1.2, (255,0,255) ),
                                                    (380, 650 , 350, None, 1.7, (0,255,255) )
                                                    ]
        self.heatmap_acc = deque(maxlen=3)
        self.threshold = 8
        
    def find_cars (self, img):
        cars_boxes_single =[]    
        cars_boxes =[]    
        for (ystart, ystop, xstart, xstop, scale, color) in self.ystart_ystop_xstart_xstop_scale_color:
            cars_boxes_single = find_cars_boxes(img, ystart, ystop, xstart, xstop, scale, self.svc, self.X_scaler, self.color_space, self.orient, self.pix_per_cell, self.cell_per_block, self.hog_channel, self.spatial_size, self.hist_bins)      
            cars_boxes += cars_boxes_single

#------            
#        heat = np.zeros_like(img[:,:,0]).astype(np.float)
#        heat = add_heat(heat,cars_boxes)
#        heat = apply_threshold(heat,6)
#        
#        heatmap = np.clip(heat, 0, 255)
#        labels = label(heatmap)
#        cars_boxes = draw_labeled_bboxes(np.copy(img), labels)
#        result_img = draw_boxes(img,cars_boxes)
#        show_image(cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

#-----
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        heat = add_heat(heat, cars_boxes)
        self.heatmap_acc.append(heat)
        heat = np.sum(np.array(self.heatmap_acc), axis=0)
        heat = apply_threshold(heat, self.threshold)

        heatmap = np.clip(heat, 0, 255)
        labels = label(heatmap)
        cars_boxes = draw_labeled_bboxes(np.copy(img), labels)
        result_img = draw_boxes(img,cars_boxes)
                
        return result_img
    

