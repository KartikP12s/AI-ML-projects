import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('/Users/kartikpatel/Downloads/dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_datagen=ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('/Users/kartikpatel/Downloads/dataset/test_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
cnn= Sequential()
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(MaxPool2D(pool_size=2, strides=2))
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(MaxPool2D(pool_size=2, strides=2))
cnn.add(Flatten())

cnn.add(Dense(units=128,activation='relu'))
cnn.add(Dense(units=1,activation='sigmoid'))

cnn.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])
cnn.fit(x=training_set,validation_data=test_set, batch_size= 32, epochs=10)
test_image= image.load_img('/Users/kartikpatel/Downloads/dataset/test_set/cats/cat.4001.jpg',color_mode='rgb',target_size=(64,64))
test_image= image.img_to_array(test_image)
test_image= np.expand_dims(test_image,axis=0)
result=cnn.predict(test_image)
print(result)
training_set.class_indices
if result[0][0]==0:
    prediction='cat'
else:
    prediction='dog'
print(prediction)