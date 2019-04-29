from __future__ import print_function
import os
import numpy as np
#np.random.seed(1337)
import input_data
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import EarlyStopping 
from keras.callbacks import ModelCheckpoint

batch_size = 150
nb_classes = 100
nb_epoch = 200

img_rows, img_cols = 256, 256  # 输入图像的维度
pool_size = (2,2)  # 池化层操作的范围
#  读入数据
petdog_set = input_data.load_data()
train_set = petdog_set['train_set']  # 训练数据
test_set = petdog_set['test_set']   # 测试数据
random.shuffle(train_set)  # 将训练数据打乱

if len(train_set)%batch_size != 0:
    loopamount = int(len(train_set)/batch_size)+1
else:
    loopamount = int(len(train_set)/batch_size)

'''定义一个生成器实时生成需要处理的训练集batch'''
def generate_batch_data(data_set,batch_size):
    images_test,labels_test = input_data.seperate_img_lebel(data_set)

    while (True):
        for i in range(loopamount):
            imgs_array_test = input_data.imgs_to_array(images_test[i*batch_size:(i+1)*batch_size])
            labels_array_test = input_data.labels_to_array(labels_test[i*batch_size:(i+1)*batch_size])
            labels_array_test = np_utils.to_categorical(labels_array_test, nb_classes)
            yield (imgs_array_test,labels_array_test)

#X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
#X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# 将X_train, X_test的数据格式转为float32
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
# 打印出相关信息
print(len(train_set), 'train_set')
print(len(test_set), 'test_set')

test_set = petdog_set['test_set']
images_test,labels_test = input_data.seperate_img_lebel(test_set)
labels_array_test = input_data.labels_to_array(labels_test)
X_test = input_data.imgs_to_array(images_test)
# 相当于将向量用one-hot重新编码
Y_test = np_utils.to_categorical(labels_array_test, nb_classes)  

# 建立序贯模型
model = Sequential()

model.add(Conv2D(filters=96, kernel_size=(11,11),strides=(4, 4),padding='valid',input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3),strides=2))
model.add(Dropout(0.25))

model.add(Conv2D(filters=256, kernel_size=(5,5),strides=(2, 2),padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3),strides=2))
model.add(Dropout(0.25))

model.add(Conv2D(filters=384, kernel_size=(3,3),strides=(1, 1),padding='valid'))
model.add(Activation('relu'))
#model.add(Dropout(0.25))

model.add(Conv2D(filters=384, kernel_size=(3,3),strides=(1, 1),padding='same'))
model.add(Activation('relu'))
#model.add(Dropout(0.25))

model.add(Conv2D(filters=256, kernel_size=(3,3),strides=(1, 1),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3),strides=2))
#model.add(Dropout(0.25))

# Flatten层，把多维输入进行一维化，常用在卷积层到全连接层的过渡
model.add(Flatten())
# 包含1024个神经元的全连接层
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.25))

# 包含nb_classes个神经元的输出层，激活函数为Softmax
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# 输出模型的参数信息
model.summary()
# 配置模型的学习过程
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


'''回调函数callback'''
#early_stopping = EarlyStopping(monitor='val_loss', patience=5)  # 终止训练条件
# 保存中间训练结果
model_save = ModelCheckpoint('save/mid_model.h5', monitor='val_loss', 
                                verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=3)
'''训练模型'''
if os.path.exists('save/mid_model.h5') == True:
    model = load_model('save/mid_model.h5')

model.fit_generator(generate_batch_data(train_set,batch_size), steps_per_epoch=loopamount, 
                    epochs=nb_epoch,verbose=1, validation_data=(X_test, Y_test),
                    callbacks=[model_save])  # early_stopping,

# 按batch计算在某些输入数据上模型的误差
score = model.evaluate(X_test, Y_test, verbose=0)
model.save('save/model_final.h5')  # save final model
# 输出训练好的模型在测试集上的表现
print('Test score:', score[0])
print('Test accuracy:', score[1])




