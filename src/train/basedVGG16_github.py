from __future__ import print_function
import os
import time
time_start = time.time()
from keras.layers import Dense, Flatten,Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import EarlyStopping 
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras import optimizers
from keras.layers import Input
from keras.utils.data_utils import get_file
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.models import Sequential

from datetime import datetime
from LogHistory import LogHistory

nb_classes = 100
nb_epoch = 10
img_rows, img_cols = 224, 224  # 输入图像的维度
top_model_weights_path = 'save/bottleneck_100_model_1layer_2048_1024.h5'
vgg16_no_top_weights_path = "save/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
mid_model_weights_path = "save/mid_model_100.h5"
final_model_weights_path = "save/final_model_100.h5"
train_data_dir = "imgs_100c/train"
validation_data_dir = "imgs_100c/test"
nb_train_samples = 14876
nb_validation_samples = 3771

'''数据提升'''
train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1./255)

'''
构建网络模型
'''
# build the VGG16 network
input_shape = (img_rows, img_cols, 3)
inputs= Input(shape=input_shape)

'''VGG16'''
#-----------------------------------------------------------
# Block 1
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Block 3
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# Block 4
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

# Block 5
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
x1 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
#-----------------------------------------------------------
x = Flatten(name='flatten')(x1)
x = Dense(2048, activation='relu',name='bottleneck_100_1')(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation='relu',name='bottleneck_100_2')(x)
x = Dropout(0.2)(x)
predictions_100 = Dense(nb_classes, activation='softmax')(x)

'''VGG16 over'''
#vgg16_model = Model(inputs=inputs, outputs=x1,name='vgg16')
#vgg16_model.load_weights(vgg16_no_top_weights_path)
#print('vgg16 Model loaded.')
#print("vgg16 model shape:",vgg16_model.output_shape)
#
#top_inputs= Input(shape=vgg16_model.output_shape[1:])
#top_model = Model(inputs=top_inputs, outputs=predictions_10,name='top_model')
#top_model.load_weights(top_model_weights_path)
#print('top Model loaded.')
#print("top model shape:",top_model.output_shape)

model = Model(inputs=inputs,outputs=predictions_100,name='petdog')
# model.load_weights(top_model_weights_path,by_name=True)
# model.load_weights(vgg16_no_top_weights_path,by_name=True)
# print("vgg16 weights loaded")
#
# print("top_model weights loaded")

print("model model shape:",model.output_shape)

#top_model = Sequential()
#top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
#top_model.add(Dense(1024, activation='relu'))
#top_model.add(Dropout(0.2))
#top_model.add(Dense(nb_classes, activation='softmax'))
#top_model.load_weights(top_model_weights_path)

#print("top_model shape:",top_model.output_shape)
##
#
#y = vgg16_model
#model = Model(inputs=inputs, outputs=y,name='petdog')
#
#model.add(top_model)
#print('top_model loaded.')
#print("all_model shape:",model.output_shape)

#载入模型
if os.path.exists(mid_model_weights_path) == True:
   model.load_weights(mid_model_weights_path,by_name=True)
   print("final_model_weights loaded")


'''锁定前面的层不训练'''
for layer in model.layers[:16]: #16
   layer.trainable = False
for layer in model.layers[16:]: #20
   layer.trainable = True

print('depth:',model.layers_by_depth)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.5),
              metrics=['accuracy'])

'''回调函数callback'''
# 保存中间训练结果
model_save = ModelCheckpoint(mid_model_weights_path, monitor='val_loss', 
                                verbose=1, save_best_only=False, save_weights_only=False,
                             mode='auto', period=1)

write_log = LogHistory('log.txt')

'''训练模型'''
batch_size = 50

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size)

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size)

print(train_generator.class_indices)
print(validation_generator.class_indices)
history = model.fit_generator(train_generator, 
                    steps_per_epoch=(nb_train_samples//batch_size)+1, 
                    epochs=nb_epoch,verbose=1, 
                    validation_data=validation_generator,
                    validation_steps=(nb_validation_samples//batch_size)+1,
                    callbacks=[model_save,write_log])  # early_stopping,tensorboard

with open('log.txt','a') as f:
    str_time = str(datetime.now().isoformat(' '))
    f.write(str_time+'\n')
    f.write(str(history.history))
model.save(final_model_weights_path)  # save final model'save/final_model_100.h5'

#------------------------------------
time_end = time.time()
epis = time_end - time_start
print('used time:',int(epis/60),'mins',int(epis%60),'secs')








