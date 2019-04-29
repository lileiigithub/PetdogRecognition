from __future__ import print_function
import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import random
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import EarlyStopping 
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras import optimizers
import input_data
from keras.layers import Input
train_batch_size = 32
test_batch_size = 32
nb_classes = 5
nb_epoch = 1

img_rows, img_cols = 224, 224  # 输入图像的维度

'''
读入及预处理数据
'''
petdog_set = input_data.load_data()
train_set = petdog_set['train_set']  # 训练数据
test_set = petdog_set['test_set']   # 测试数据
random.shuffle(train_set)  # 将训练数据打乱

#定义一个生成器实时生成需要处理的训练集batch
def calc_loop(data_set,batch_size):
    if len(data_set)%batch_size != 0:
        loopamount = int(len(data_set)/batch_size)+1
    else:
        loopamount = int(len(data_set)/batch_size)
    return loopamount

def generate_batch_data(data_set,batch_size):
    while (True):
        for i in range(calc_loop(data_set,batch_size)):
            batch_set = np.random.choice(data_set, batch_size, replace=False, p=[1/len(data_set)]*len(data_set))
            images_list,labels_list = input_data.seperate_img_lebel(batch_set)
            imgs_array = input_data.imgs_to_array(images_list)
            labels_array = input_data.labels_to_array(labels_list)
            labels_array = np_utils.to_categorical(labels_array, nb_classes)
            yield (imgs_array,labels_array)

print(len(train_set), 'train_set')
print(len(test_set), 'test_set')


'''
构建网络模型
'''
input_shape = (img_rows, img_cols, 3)

inputs= Input(shape=input_shape)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3),input_tensor=inputs)
x = base_model.output
x = Flatten(name='flatten')(x)
x = Dense(1024, activation='relu',name='fc1')(x)
predictions = Dense(nb_classes, activation='softmax')(x)
model = Model(inputs=inputs, outputs=predictions,name='petdog')

'''锁定前面的层不训练'''
for layer in base_model.layers:
    layer.trainable = False

model.summary()

#model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

'''回调函数callback'''
#early_stopping = EarlyStopping(monitor='val_loss', patience=5)  # 终止训练条件
# 保存中间训练结果
model_save = ModelCheckpoint('save/mid_model.h5', monitor='val_loss', 
                                verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
#载入模型
if os.path.exists('save/model_final.h5') == True:
    model = load_model('save/model_final.h5')

images_list,labels_list = input_data.seperate_img_lebel(test_set)
X_test= input_data.imgs_to_array(images_list)
Y_test=input_data.labels_to_array(labels_list)
Y_test=np_utils.to_categorical(Y_test, nb_classes)

'''训练模型'''
#model.fit_generator(generate_batch_data(train_set,train_batch_size), 
#                    steps_per_epoch=calc_loop(train_set,train_batch_size), 
#                    epochs=nb_epoch,verbose=1, 
#                    validation_data=generate_batch_data(test_set,test_batch_size),
#                    validation_steps=calc_loop(test_set,test_batch_size),
#                    callbacks=[model_save])  # early_stopping,

history = model.fit_generator(generate_batch_data(train_set,train_batch_size), 
                    steps_per_epoch=calc_loop(train_set,train_batch_size), 
                    epochs=nb_epoch,verbose=1, 
                    validation_data=(X_test, Y_test),
                    callbacks=[model_save])  # early_stopping
                              
# 按batch计算在某些输入数据上模型的误差
#score = model.evaluate_generator(generate_batch_data(test_set,test_batch_size), 
#                                 steps=calc_loop(test_set,test_batch_size))

model.save('save/model_final.h5')  # save final model
# 输出训练好的模型在测试集上的表现
#print('myacc:',history.history['acc'])
#images_list,labels_list = input_data.seperate_img_lebel(test_set)
#X_test= input_data.imgs_to_array(images_list)
#Y_test=input_data.labels_to_array(labels_list)
#Y_test=np_utils.to_categorical(Y_test, nb_classes)
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

#print(score)

                                        






