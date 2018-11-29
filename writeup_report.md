# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* video.mp4 is a recording of autnomous driving with 9mph.
* video_20mph.mp4 is a recording of autonomous driving with 20mph.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I refered [NVIDIA's end-to-end deep learning model for self-diving cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/
). The model architecture is implemented in `line:77 in model.py`.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. ( `line:84 in model.py`)

The model was trained and validated on different data sets to ensure that the model was not overfitting (`line: 100 in model.py`). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (`line:113 in model.py`).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I made proper driving (run on center) for a few times also made recovering examples from left and right end of the lane. I made `line: 13-17 in model.py` to accept arbitrary number of `driving_log.csv` for model tuning.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because this problem looks finding appropriate steering angle from the images.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

The final step was to run the simulator to see how well the car was driving around track one. I found that my first model actually works not so bad and achieve to run a vehicle without falling out of the lane.

However as mentioned in the lecture, I tried Nvidia's model. With this, the vehicle is also able to drive autonomously around the track without leaving the road.

Also I tried speed up the vehicle from 9mph to 20mph. The trained model managed to run safely in this speed.

#### 2. Final Model Architecture

Here is a visualization of the final model architecture.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 66 x 200 x 3 YUV image.   							| 
| Convolution 5x5    	| 2x2 stride, valid padding, outputs 31 x 98 x 24 	|
| RELU					|												|
| Convolution 5x5    	| 2x2 stride, valid padding, outputs 14 x 47 x 36 	|
| RELU					|												|
| Convolution 5x5    	| 2x2 stride, valid padding, outputs 5 x 22 x 48 	|
| RELU					|												|
| Convolution 3x3    	| 1x1 stride, valid padding, outputs 3 x 20 x 64 	|
| RELU					|												|
| Convolution 3x3    	| 1x1 stride, valid padding, outputs 1 x 18 x 64 	|
| RELU					|												|
| DROPOUT               | keep prob = 0.5                               |
| Fully connected		| 1164 input feature, outputs 100 features |
| RELU					|												| 
| Fully connected		| 100 input feature, outputs 50 features |
| RELU					|												| 
| Fully connected		| 50 input feature, outputs 10 features |
| RELU					|												| 
| Fully connected		| 10 input feature, outputs 1 feature |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover center position.

Then I ran a training and test that model with simulator. It does not work well, most of time the vehicle falls into water just after the first curve and before the bridge. So I decided to collect more proper running laps and recovering procedure recordings.

At last number of collected data points is about 10K and each point has 3 camera images. For training and validation, I chose to use `train_test_split in scikitlearn`, you can find that at `line:100 in model.py`. This performs shuffling the datapoints and then split them into 75% of training data and 25% of validation data.

The batch generator which feeds data to model is implemented in `line:56 in model.py`. It picks up randomly from the training dataset and feed batches to the model. In that procedure, the process(inplmented in `preprocessor.py`) is performed, which takes care of

- Crop image to remove skies
- Resize cropped image to 200x66, which is expected input shape of nvidia's model.
- Change color schema from BGR to YUV.
- Normalize value range from 0-255 to -0.5-0.5, to speed up training.

Also I randomly apply horizontal flip as a augmentation. You can find that in `line:23 in model.py`.

For epoch of training, I started from 2, but it seemed it is enough by running a simulator, so I did not try tuning this parameter.

### Conclusion

To run the vehicle in the simulator, I also modified `drive.py, line:66` because Nividia's model expecte the input in shape of 200x66x3.

Finally, as you can find in `video.mp4` and `video_20mph.mp4`, the model is well trained for the course 1 and it well drive the vehicle.