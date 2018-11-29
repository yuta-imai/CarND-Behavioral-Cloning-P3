import csv
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Activation, Flatten,  Dense, Lambda, Dropout, MaxPooling2D
from keras.layers.convolutional import Conv2D, Cropping2D

import preprocessor

recorded_examples = [
    '../recordings/proper/driving_log.csv',
    '../recordings/lap3/driving_log.csv',
    '../recordings/recover/driving_log.csv',
    '../recordings/recover2/driving_log.csv',
    '../recordings/smooth/driving_log.csv'
]

def augment(images,steerings):
    
    for img,steering in zip(images, steerings):
        if random.randint(0,100) < 5:
            flipped_img = cv2.flip(img, 1)
            flipped_steering = steering * -1
            images.append(flipped_img)
            steerings.append(flipped_steering)
    
    return images, steerings

def random_augment(image,steering):
    
    if random.randint(0,100) < 50:
        image = cv2.flip(image, 1)
        steering = steering * -1

    return image, steering
    

def load_images(path):
    lines = []
    
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    steerings = []
    for line in lines:

        steering_center = float(line[3])

        correction = 0.2 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        img_center =preprocessor.preprocess(cv2.imread(line[0]))
        img_left = preprocessor.preprocess(cv2.imread(line[1]))
        img_right = preprocessor.preprocess(cv2.imread(line[2]))

        steerings.extend([steering_center,steering_left,steering_right])    
        images.extend([img_center,img_left,img_right])

    images, steerings = augment(images, steerings)
    return images, steerings

def load_image_and_steering_for_train(csv_line):
    
    correction = 0.2 
    
    steering = float(csv_line[3])
    i = random.randint(0,2)
    
    if i == 0:
        img = preprocessor.preprocess(cv2.imread(csv_line[0]))
    elif i == 1:
        img = preprocessor.preprocess(cv2.imread(csv_line[1]))
        steering = steering + correction
    else:
        img = preprocessor.preprocess(cv2.imread(csv_line[2]))
        steering = steering - correction
        
    img, steering = random_augment(img, steering)
        
    return img, steering
        
def load_image_and_steering_for_valid(csv_line):
    steering = float(csv_line[3])
    img =preprocessor.preprocess(cv2.imread(csv_line[0]))
    return img, steering
        
    
def generator(csv_lines, batch_size, is_training):
    
    images = np.zeros((batch_size, 66, 200, 3))
    steerings = np.zeros(batch_size)
    
    while True:
        i = 0
        for index in np.random.permutation(len(csv_lines)):
            if is_training:
                image, steering = load_image_and_steering_for_train(csv_lines[index])
            else:
                image, steering = load_image_and_steering_for_valid(csv_lines[index])
            
            images[i] = image
            steerings[i] = steering
            
            i += 1
            if i == batch_size:
                break
        yield images, steerings
            
def model_lenet():
    model = Sequential()
    model.add(Conv2D(6,5,5, input_shape=preprocessor.input_shape()))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    model.add(Conv2D(6,5,5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

def model_nvidia():
    model = Sequential()
    model.add(Conv2D(3,(5,5), strides=(2,2), activation='relu', input_shape=preprocessor.input_shape()))
    model.add(Conv2D(24,(5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(36,(5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(48,(3,3),  activation='relu'))
    model.add(Conv2D(64,(3,3),  activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model
    

csv_lines = []
for path in recorded_examples:
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            csv_lines.append(line)

train_lines, valid_lines = train_test_split(csv_lines)      
print("Num of train csv lines: " + str(len(train_lines)))
print("Num of valid csv lines: " + str(len(valid_lines)))

batch_size = 32
steps_per_epoch =1000
validation_steps=int(len(valid_lines)/batch_size)

print("Batch size: " + str(batch_size))
print("Steps per epoch: " + str(steps_per_epoch))
print("Validation steps: " + str(validation_steps))

model = model_nvidia()
model.compile(loss='mse', optimizer='adam')
model.fit_generator(
    generator(train_lines, batch_size, True),
    steps_per_epoch= steps_per_epoch ,
    epochs=2,
    validation_data=generator(valid_lines,32,False),
    validation_steps=validation_steps,
    verbose=1)
model.save('model.h5')