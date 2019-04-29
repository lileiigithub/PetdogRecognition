from __future__ import print_function
import os
import numpy as np
import random
#import keras
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
import input_data

train_batch_size = 40
test_batch_size = 41
nb_classes = 100
nb_epoch = 16

img_rows, img_cols = 224, 224  # 输入图像的维度

#top_model_weights_path = 'save/bottleneck_10_model.h5'
top_model_weights_path = 'save/bottleneck_100_model_2layer_1024.h5'
train_data_dir = "imgs_100c/train"
validation_data_dir = "imgs_100c/test"
nb_train_samples = 14876
nb_validation_samples = 3771

'''
读入及预处理数据
'''
petdog_set = input_data.load_data()
train_set = petdog_set['train_set']  # 训练数据
test_set = petdog_set['test_set']   # 测试数据
random.shuffle(train_set)  # 将训练数据打乱
random.shuffle(test_set) 
'''训练集与测试集生成器'''
def calc_loop(data_set,batch_size):
    if len(data_set)%batch_size != 0:
        loopamount = int(len(data_set)/batch_size)+1
    else:
        loopamount = int(len(data_set)/batch_size)
    return loopamount
#定义一个生成器实时生成需要处理的训练集batch ， batch以一定的概率从训练集获取，其概率为均匀分布
def generate_train_batch():
    data_set = train_set
    batch_size = train_batch_size
    while (True):
        for i in range(calc_loop(data_set,batch_size)):
            batch_set = np.random.choice(data_set, batch_size, replace=False) #不放回的均匀分布，p=[1/len(data_set)]*len(data_set)
            images_list,labels_list = input_data.seperate_img_lebel(batch_set)
            imgs_array = input_data.imgs_to_array(images_list)
            labels_array = input_data.labels_to_array(labels_list)
            labels_array = np_utils.to_categorical(labels_array, nb_classes)
            yield (imgs_array,labels_array)

images_list,labels_list = input_data.seperate_img_lebel(train_set)
x_train = input_data.imgs_to_array(images_list)
labels_array = input_data.labels_to_array(labels_list)
y_train = np_utils.to_categorical(labels_array, nb_classes)

#定义一个生成器实时生成需要处理的测试集batch ，batch为 测试集从头到尾的遍历
def generate_test_batch():
    data_set = test_set
    batch_size = test_batch_size
    images_test,labels_test = input_data.seperate_img_lebel(data_set)
    while (True):
        for i in range(calc_loop(data_set,batch_size)):
            imgs_array_test = input_data.imgs_to_array(images_test[i*batch_size:(i+1)*batch_size])
            labels_array_test = input_data.labels_to_array(labels_test[i*batch_size:(i+1)*batch_size])
            labels_array_test = np_utils.to_categorical(labels_array_test, nb_classes)
            yield (imgs_array_test,labels_array_test)

print(len(train_set), 'train_set')
print(len(test_set), 'test_set')

'''数据提升'''
train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

'''
构建网络模型
'''
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
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
#-----------------------------------------------------------
'''VGG16 over'''
# build the VGG16 network
model = applications.VGG16(weights='imagenet', include_top=False)
print('Model loaded.')

top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(1024, activation='relu'))
top_model.add(Dropout(0.2))
top_model.add(Dense(1, activation='softmax'))
top_model.load_weights(top_model_weights_path)

#top_inputs= Input(shape=model.output_shape[1:])
#top_inputs= Input(shape=(7,7,512))
#x = Flatten(name='flatten')(top_inputs)
#x = Dense(1024, activation='relu',name='bottleneck_10')(x)
#x = Dropout(0.2)(x)
#predictions_10 = Dense(nb_classes, activation='softmax')(x)
#top_model = Model(inputs=top_inputs, outputs=predictions_10,name='petdog')
#top_model.load_weights(top_model_weights_path)

model.add(top_model)
#model = Model(inputs=inputs, outputs=predictions_10,name='petdog')

#载入模型
if os.path.exists('save/mid_model.h5') == True:
#    model = load_model('save/mid_model.h5')
#    WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    #vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
#    weights_path = get_file('save/mid_model.h5', 
#                        WEIGHTS_PATH_NO_TOP,
#                        cache_subdir='models')
#    model.load_weights('save/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',by_name=True)
#    model.load_weights('save/bottleneck_10_model.h5',by_name=True)
#    model.load_weights('save/mid_model.h5',by_name=True)
    pass

'''锁定前面的层不训练'''
for layer in model.layers[:20]: #16
   layer.trainable = False
for layer in model.layers[20:]:
   layer.trainable = True

print('depth:',model.layers_by_depth)
model.summary()
#sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.5),
              metrics=['accuracy'])

'''回调函数callback'''
# 保存中间训练结果
model_save = ModelCheckpoint('save/mid_model.h5', monitor='val_loss', 
                                verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
#tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, 
#                            embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

write_log = LogHistory('log.txt')

'''训练模型'''
#history = model.fit_generator(generate_train_batch(), 
#                    steps_per_epoch=calc_loop(train_set,train_batch_size), 
#                    epochs=nb_epoch,verbose=1, 
#                    validation_data=generate_test_batch(),
#                    validation_steps=calc_loop(test_set,test_batch_size),
#                    callbacks=[model_save,tensorboard,write_log])  # early_stopping,

#datagen.fit(x_train)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=32)

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=32)

history = model.fit_generator(train_generator, 
                    steps_per_epoch=nb_train_samples//32+1, 
                    epochs=nb_epoch,verbose=1, 
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples//32+1,
                    callbacks=[model_save,write_log])  # early_stopping,tensorboard

# 按batch计算在某些输入数据上模型的误差
score = model.evaluate_generator(generate_test_batch(), 
                                 steps=calc_loop(test_set,test_batch_size))

with open('log.txt','a') as f:
    str_time = str(datetime.now().isoformat(' '))
    f.write(str_time+'\n')
    f.write(str(history.history))

model.save('save/final_model_10.h5')  # save final model
print('score:',score)











