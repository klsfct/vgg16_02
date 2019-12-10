from keras.preprocessing.image import ImageDataGenerator
import os,shutil

original_dataset_dir1='D:/ImageSet/wenlidalou'
original_dataset_dir2='D:/ImageSet/guozhonglou'
base_dir='D:/building'
os.mkdir(base_dir)

train_dir=os.path.join(base_dir,'train')
os.mkdir(train_dir)
validation_dir=os.path.join(base_dir,'validation')
os.mkdir(validation_dir)
test_dir=os.path.join(base_dir,'test')
os.mkdir(test_dir)
#训练图像
train_wenlidalou_dir=os.path.join(train_dir,'wenlidalou')
os.mkdir(train_wenlidalou_dir)

train_guozhonglou_dir=os.path.join(train_dir,'guozhonglou')
os.mkdir(train_guozhonglou_dir)
#验证图像
validation_wenlidalou_dir=os.path.join(validation_dir,'wenlidalou')
os.mkdir(validation_wenlidalou_dir)

validation_guozhonglou_dir=os.path.join(validation_dir,'guozhonglou')
os.mkdir(validation_guozhonglou_dir)
#测试测试图像
test_wenlidalou_dir=os.path.join(test_dir,'wenlidalou')
os.mkdir(test_wenlidalou_dir)

test_guozhonglou_dir=os.path.join(test_dir,'guozhonglou')
os.mkdir(test_guozhonglou_dir)

#将前100张图片复制到train_wenlidalou_dir
fnames = ['wenlidalou{}.jpg'.format(i) for i in range(100)]
for fname in fnames:
    src = os.path.join(original_dataset_dir1, fname)
    dst = os.path.join(train_wenlidalou_dir, fname)
    shutil.copyfile(src, dst)

fnames=['wenlidalou{}.jpg'.format(i) for i in range(100,115)]
for fname in fnames:
    src=os.path.join(original_dataset_dir1,fname)
    dst=os.path.join(validation_wenlidalou_dir,fname)
    shutil.copyfile(src,dst)

fnames=['wenlidalou{}.jpg'.format(i) for i in range(115,130)]
for fname in fnames:
    src=os.path.join(original_dataset_dir1,fname)
    dst=os.path.join(test_wenlidalou_dir,fname)
    shutil.copyfile(src,dst)

fnames = ['guozhonglou{}.jpg'.format(i) for i in range(100)]
for fname in fnames:
    src = os.path.join(original_dataset_dir2, fname)
    dst = os.path.join(train_guozhonglou_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['guozhonglou{}.jpg'.format(i) for i in range(100, 115)]
for fname in fnames:
    src = os.path.join(original_dataset_dir2, fname)
    dst = os.path.join(validation_guozhonglou_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['guozhonglou{}.jpg'.format(i) for i in range(115, 130)]
for fname in fnames:
    src = os.path.join(original_dataset_dir2, fname)
    dst = os.path.join(test_guozhonglou_dir, fname)
    shutil.copyfile(src, dst)



#对于图片进行数据增强


from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc']
              )
# 数据的预处理

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 225)
test_datagen = ImageDataGenerator(rescale=1. / 225)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')

validation_generator=train_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

#history=model.fit_generator(
   # train_generator,
    #steps_per_epoch=100,
    #epochs=30,
    #validation_data=validation_generator,
    #validation_steps=50
#)

#model.save('building_train1.h5')

import matplotlib.pyplot as plt

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')

plt.title('Training and validation loss')
plt.legend()

plt.show()

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

import os
from keras.preprocessing import image

fnames = [os.path.join(train_wenlidalou_dir, fname) for fname in os.listdir(train_wenlidalou_dir)]
img_path = fnames[3]
img = image.load_img(img_path, target_size=(150, 150))

x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot=plt.imshow(image.array_to_img(batch[0]))
    i+=1
    if i%4==0:
        break

#到此数据增强已经完成
model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))

model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

train_datagen=ImageDataGenerator(
    rescale=1./225,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
    train_dir,#这个地方是训练目录
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

validation_generator=test_datagen.flow_from_directory(
    validation_dir,#要写具体的目录
    target_size=(150,150),
    batch_size=32,
    class_mode='binary',

)

history=model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50
)

#保存模型
model.save('buliding_train2.h5')
