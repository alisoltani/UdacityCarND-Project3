
# coding: utf-8

# In[ ]:

import csv
import cv2
import ntpath
import numpy as np

def process_files(IsWindows=False, originalpath='data/', flip=False):
	lines = []
	with open(originalpath + 'driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		next(reader)
		for line in reader:
			lines.append(line)
        
	car_camera_images = []
	steering_angles = []

	steering_correction = 0.15

	for line in lines:
		source_path = line[0]
		if IsWindows:
			drive, path_and_file = ntpath.splitdrive(source_path)
			path, filename = ntpath.split(path_and_file)
		else:
			filename = source_path.split('/')[-1]
			
		left_filename = filename.replace('center', 'left')
		right_filename = filename.replace('center', 'right')
		
		center_path = originalpath + 'IMG/' + filename
		left_path = originalpath + 'IMG/' + left_filename
		right_path = originalpath + 'IMG/' + right_filename
		
		image_center = cv2.imread(center_path)
		image_left = cv2.imread(left_path)
		image_right = cv2.imread(right_path)

		if flip:
			image_center = cv2.flip(image_center, 0)
			image_left = cv2.flip(image_left,0)
			image_right = cv2.flip(image_right,0)
		
		steering_angle = float(line[3])
		steering_left = steering_angle + steering_correction
		steering_right = steering_angle - steering_correction
		
		if flip:
			steering_angle = -steering_angle
			steering_left = -steering_left
			steering_right = -steering_right
		
		car_camera_images.extend([image_center, image_left, image_right])
		steering_angles.extend([steering_angle, steering_left, steering_right])
		
	return car_camera_images, steering_angles
car_camera_images = []
steering_angles = []

car_camera_images, steering_angles = process_files(IsWindows=False, originalpath='data/', flip=False)
car_camera_images_windows, steering_angles_windows = process_files(IsWindows=True, originalpath='MyOwnData/', flip=True)

car_camera_images = car_camera_images + car_camera_images_windows
steering_angles = steering_angles + steering_angles_windows

car_camera_images_windows, steering_angles_windows = process_files(IsWindows=True, originalpath='MyOwnData2/', flip=False)
car_camera_images = car_camera_images + car_camera_images_windows
steering_angles = steering_angles + steering_angles_windows

car_camera_images_windows, steering_angles_windows = process_files(IsWindows=True, originalpath='MyOwnData3', flip=False)
car_camera_images = car_camera_images + car_camera_images_windows
steering_angles = steering_angles + steering_angles_windows

X_train = np.array(car_camera_images)
y_train = np.array(steering_angles)
#X_train_to_flip = np.array(car_camera_images_windows)
#y_train_to_flip = np.array(steering_angles_windows)

#X_train_flipped = np.fliplr(X_train_to_flip)
#y_train_flipped = -y_train_to_flip

#X_train = np.concatenate((X_train, X_train_flipped), axis=0)
#y_train = np.concatenate((y_train, y_train_flipped), axis=0)

print('The shape of the image data is', X_train.shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, Cropping2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D

## Inspired from the NVIDIA model
model = Sequential()
model.add(Lambda(lambda x: (x/127.5) - 1, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((55,5),(0,0))))
model.add(Conv2D(24, 5, 5, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(36, 5, 5, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(48, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1014, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)

model.save('model.h5')

