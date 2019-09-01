import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier= Sequential()

classifier.add(Convolution2D(32, 3, 3, input_shape= (64, 64, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size= (2, 2)))

classifier.add(Convolution2D(32, 3, 3, activation='relu'))

classifier.add(MaxPooling2D(pool_size= (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units=128, activation= 'relu' , kernel_initializer= 'uniform' ))

classifier.add(Dense(units=1, activation = 'sigmoid', kernel_initializer= 'uniform'))

classifier.compile(optimizer='adam' ,loss='binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_augment = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_augment = ImageDataGenerator(rescale=1./255)

training_set= train_augment.flow_from_directory(
                                                'dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')    

test_set = test_augment.flow_from_directory(
                                            'dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=64,
                                            class_mode='binary')

classifier.fit_generator(
                    training_set,
                    steps_per_epoch=(8000/64),
                    epochs=1,
                    validation_data=test_set,
                    nb_val_samples=(2000/64
                                    ))

from skimage.io import imread
from skimage.transform import resize
img = imread('dataset/cat_test.jpg') #make sure that path_to_file contains the path to the image you want to predict on. 
img = resize(img,(64,64))
img = np.expand_dims(img,axis=0)
img = img/(255.0)
prediction = classifier.predict_classes(img)
         
if(prediction):
   print ("DOG")
else:
   print ("CAT")