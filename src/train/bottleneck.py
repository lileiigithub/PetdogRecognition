from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dropout, Flatten, Dense
from keras import optimizers
from keras.utils import np_utils
import numpy as np
import os
import time

'''
先得出图片的 bottleneck
在训练出全连接层
参考：https://gist.github.com/lileiigithub/74e88b115e866759000c37d153ad52fd
'''

train_data_dir = "imgs_100c/train"
validation_data_dir = "imgs_100c/test"
nb_train_samples = 14876
nb_validation_samples = 3771
nb_classes = 100

img_width, img_height = 224, 224  # 输入图像的维度
batch_size = 50
epochs = 3

top_model_weights_path = 'save/bottleneck_100_model_1layer_2048_1024.h5'

'''记录图片数据在VGG16下的bottleneck特征'''
def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    model.summary()
    generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)
    
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples // batch_size+1)
    np.save('bottleneck_features/bottleneck_features_train.npy',bottleneck_features_train)
    
    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator, 
                                                             nb_validation_samples // batch_size+1)
    np.save('bottleneck_features/bottleneck_features_validation.npy',bottleneck_features_validation)
    
'''one-hot start'''
# 获取每张图片的路径并存入列表
def get_labels_array(_data_dir):
    path_file_list = []
    for root,dirs,files in os.walk(_data_dir):
        for name in files:
            path_file_list.append(os.path.join(root,name))
    return path_file_list
    
def get_class_file(_path_file_list):
    class_file_list = []
    for item in _path_file_list:
        item = item.strip()
        class_file_list.append([item.split('\\')[-2],item.split('\\')[-1].split('.jpg')[0]])
    return class_file_list
    
def get_ordered_label(_class_file_list):
    label_ordered_list = []
    for item in _class_file_list:
        label_ordered_list.append(int(item[0]))
    return label_ordered_list

def ordered_list_to_one_hot(_label_ordered_list):
    labels_array = np_utils.to_categorical(_label_ordered_list, nb_classes)
    return labels_array

def from_dir_to_one_hot(_dir):
    path_file_list = get_labels_array(_dir)
    class_file_list = get_class_file(path_file_list)
    label_ordered_list = get_ordered_label(class_file_list)
    labels_array = ordered_list_to_one_hot(label_ordered_list)
    return labels_array
'''one-hot over'''

'''
训练全连接网络，
acc最后收敛到0.76,增加层数未起作用，使用SGD收敛快速
'''
def train_top_model():
    train_data = np.load(('I:/Python/tensorflow_petdog/bottleneck_features/bottleneck_features_train.npy'))
    train_labels = from_dir_to_one_hot(train_data_dir)
    
    validation_data = np.load(('bottleneck_features/bottleneck_features_validation.npy'))
    validation_labels  = from_dir_to_one_hot(validation_data_dir)
    
    
    inputs= Input(shape=train_data.shape[1:])
    x = Flatten(name='flatten')(inputs)
    x = Dense(2048, activation='relu',name='bottleneck_100_1')(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation='relu',name='bottleneck_100_2')(x)
    x = Dropout(0.2)(x)
    
    predictions_100 = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions_100,name='petdog')
    if os.path.exists(top_model_weights_path) == True:
       model.load_weights(top_model_weights_path)
       print("loaded top_model_weights_path")
    model.summary()
    model.compile(optimizer=optimizers.SGD(lr=0.01, momentum=0.4),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    
    model.save_weights(top_model_weights_path)


time_start = time.time()
#------------------------------------

#save_bottlebeck_features()
#train_top_model()
train_data = np.load(('I:/Python/tensorflow_petdog/bottleneck_features/bottleneck_features_train.npy'))
train_labels = from_dir_to_one_hot(train_data_dir)
validation_data = np.load(('bottleneck_features/bottleneck_features_validation.npy'))
validation_labels  = from_dir_to_one_hot(validation_data_dir)
#------------------------------------
time_end = time.time()
epis = time_end - time_start
print('used time:',int(epis/60),'mins',int(epis%60),'secs')




