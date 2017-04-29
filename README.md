**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model1.PNG "Model"
[image2]: ./images/model.png "Model Visualization"
[image3]: ./images/center1.jpg "Center Driving Image"
[image4]: ./images/center2.jpg "Center Driving Image"
[image5]: ./images/recovery1.png "Recovery Image"
[image6]: ./images/recovery2.png "Recovery Image"
[image7]: ./images/recovery2.png "Recovery Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md and Model.ipynb file summarizing the results
* run1, run2, and run3 video files capturing successful runs

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and along with the Model.ipynb contains comments to explain how the code works.,

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the NVIDIA model, and consists of a convolution neural network with 2 5x5 conv2D layers and 2 3x3 with depths between 32 and 64. It also included average pooling layers following the conv2d layers (model.py lines 93-129).

The model includes ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 94). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 98, 102). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 122). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an nadam optimizer, so the learning rate was not tuned manually (model.py line 122).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of data provided by Udacity and my own data consisting of center lane driving, recovering from the left and right sides of the road, driving on curves, and driving on the bridge. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to minimize the mse of the steering vector based on the image inputs.

My first step was to use a convolution neural network model similar to the NVIDIA model, due to its success in autonomous driving applications.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to include dropout layers, which successfully decreased the overfitting.

Then I decided to add average pooling layers, to simplify the training process. Then I decided to resize the image to make the training process easier and less memory demanding.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, like on the bridge or the curve after the bridge. To improve the driving behavior in these cases, I added extra driving data focusing on these instances. I also added data on how to recover from bad positions, like close to the edge of the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes

![alt text][image1]

Here is a visualization of the architecture 

![alt text][image2]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image3]
![alt text][image4]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to turn when close to the edge of the road. These images show what a recovery looks like starting from the right edge to the middle :

![alt text][image5]
![alt text][image6]
![alt text][image7]

To augment the data sat, I also flipped images thinking that this would help remove the bias of the car to turn left. 

After the collection process, I had 47250 number of data points. I then preprocessed this data by making them YUV from RGB, cropping the data, and then resizing it to 64x64.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the decrease in both training and validation error, and then it starts to increase or stays the same. 

I used an nadam optimizer so that manually training the learning rate wasn't necessary.
