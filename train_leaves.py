from keras.preprocessing.image import ImageDataGenerator  #import ImageDataGenerator function for the preprocessing of the data
from keras.models import Sequential, load_model  #the model we want to train will be sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense  #layers and the activation function 
from keras import backend as K  #import the keras library for computation
from keras.callbacks import TensorBoard
from keras.callbacks import History
import numpy as np
import numpy
import matplotlib.pyplot as plt

#image specifications
img_width, img_height = 150, 150  #dimension of the input image
train_data_dir = 'data/train' #directory path for training data
valid_data_dir = 'data/valid' #directory path for validation data
n_sample_train = 1008 #number of train samples (train neural network)
n_sample_valid = 406 #number of validation samples (target neural network)
epochs = 10

#input format check, not necessary but as a precausion
if K.image_data_format() == 'channels_last':
	input_shape = (img_width,img_height , 3)  #3 means RGB
else:
	input_shape = (3, img_width, img_height)  #3 means RGB (channels_first)

#design the network
model = Sequential()  #sequential model

model.add(Conv2D(32, (3, 3), input_shape=input_shape)) #convolutional layer 1 with 32 filters and kernal 3*3 
model.add(Activation('relu'))  #relu activation
model.add(MaxPooling2D(pool_size=(2, 2))) #pooling 1 filter size 2*2

model.add(Conv2D(32, (3, 3)))  #conv 2 filter 32, kernel 3*3
model.add(Activation('relu')) #relu activation
model.add(MaxPooling2D(pool_size=(2, 2))) #pooling 2 with filter 2*2

model.add(Conv2D(64, (3, 3)))  #conv 3 filter 64 kernal 3*3
model.add(Activation('relu'))  #relu activation
model.add(MaxPooling2D(pool_size=(2, 2))) #pooling 3 with filter 2*2

model.add(Flatten())  #image flatten layer

model.add(Dense(64)) #Dense layer 1
model.add(Activation('relu'))  #relu activation

model.add(Dropout(0.2))  #the layer shows dropout

model.add(Dense(1))  #Dense layer 2 (output)
model.add(Activation('sigmoid')) #sigmoid activation

#model compilation settings
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

batch_size = 32


history = History()

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

#image augmentation for testing using rescale
test_datagen = ImageDataGenerator(rescale=1. / 255)

#training part (training network)
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size =(img_width,img_height), batch_size=batch_size, class_mode = 'binary')

#validation part (target network)
validation_generator = test_datagen.flow_from_directory(valid_data_dir, target_size =(img_width,img_height), batch_size=batch_size, class_mode='binary')

#feed data in to model
model.fit_generator(train_generator, steps_per_epoch= n_sample_train // batch_size, epochs = epochs, validation_data=validation_generator, validation_steps=n_sample_valid // batch_size, callbacks=[history])


#save our model
model.save('model.h5')



print history.history

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
