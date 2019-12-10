from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from keras.layers.core import Lambda
from keras import backend as K
from keras.optimizers import SGD
from keras import regularizers
from keras.models import load_model
weight_decay = 0.0005
nb_epoch=100
batch_size=32
model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same',
input_shape=(224,224,3),kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same')  )
model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))
model.summary()
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range = 90,
                                   shear_range=0.5,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True
                                   )
test_datagen = ImageDataGenerator(rescale=1./255,
                                  )
t1="gaoerji"
first="C:/Users/Administrator/Desktop/耿霞/vgg16/train/cracked_traindata"
end1="C:/Users/Administrator/Desktop/耿霞/vgg16/train/uncracked_traindata"

print("#############################################")
# train_dir = r'C:\Users\\Administrator\\Desktop\\耿霞\\vgg16\\cracked_traindata'
# validation_dir = r'C:\Users\\Administrator\\Desktop\\耿霞\\vgg16\\uncracked_traindata'
hear=".h5"
change=t1
path1="C:/Users/Administrator/Desktop/耿霞/vgg16/train"
fp=first+"vgg16_"+change+hear
print(path1)

#print(path2)
print(fp)
train_generator = train_datagen.flow_from_directory(path1,
                                                    target_size=(224,224),
                                                    batch_size=5,
                                                    class_mode='binary'
                                                    )
try:
    model.load_weights(fp)
    print("done")
except:
    print("not found error!")
model.save_weights(fp)
print("save the model")