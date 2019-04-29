from __future__ import print_function
import time
time_start = time.time()
import os
import numpy as np
from PIL import Image
from keras.models import load_model

test_batch_size = 50
nb_classes = 100
IMG_SIZE = 224
IMG_CHANNEL = 3
vgg16_final_model_weights_path = "save/final_model_100.h5"
vgg19_final_model_weights_path = "save/final_model_100_vgg19.h5"
predict_imgs_path = "imgs_predict/"
predict_txt_path = "predict_vgg16_vgg19.txt"
predict_probability_txt_path = "predict_probability_vgg16_vgg19.txt"
def make_imgs_list(_path):
    imgs_list = []
    imgs_list = os.listdir(_path)
    return imgs_list

'''
输入：将图片列表 
返回：图像点的 array 类型 (None.224,224,3)
'''
def imgs_to_array(_img_list):
    images_pixel_list = []
    for item in _img_list:
        PATH = predict_imgs_path + item 
        im = Image.open(PATH)
        images_pixel_list.append(np.array(im)/255)  # 图片像素点归一化到 0~1
    return np.array(images_pixel_list,dtype='float32').reshape((len(_img_list),IMG_SIZE,IMG_SIZE,IMG_CHANNEL))

# the array shape must be (,100)
def find_array_max(_array):
    if _array.ndim is not 2:
        print("array ndim is error")
        return 0
    
    array_max_list = []
    for i in range(_array.shape[0]): # 10593
        max_index = np.argmax(_array[i])
        max_value = _array[i].max()
        array_max_list.append((max_index,max_value))
    return array_max_list
    
def make_img_label_score_list(_imgs_list,_array_max):
    if len(_imgs_list) != len(_array_max):
        print("the list format is error")
        return 0
    
    img_label_score_list = []
    for i in range(len(_imgs_list)):
        img_label_score_list.append((_imgs_list[i],_array_max[i][0],_array_max[i][1]))
    
    return img_label_score_list
    
def write_result_to_txt(_path,_path_probability,_img_label_score_list):
    with open(_path,"w") as file:
        for item in _img_label_score_list:
            file.write(item[0].split(".jpg")[0]+' '+str(item[1])+'\n')
    
    with open(_path_probability,"w") as file1:
        for item in _img_label_score_list:
            file1.write(item[0]+' '+str(item[1])+' '+str(round(item[2],4))+'\n')

def predict_imgs(_imgs_list,_predict_batch,_model):
    model = load_model(_model)
    for i in range(len(_imgs_list)//_predict_batch+1):
        #print("the index: {0}".format(i))
        x_predict=imgs_to_array(_imgs_list[i*_predict_batch:(i+1)*_predict_batch])
        y = model.predict(x_predict,verbose=1)
        if i == 0:
            y_predict = y
            print("while y_predict = y ")
        else: 
            y_predict = np.append(y_predict,y,axis = 0)
    return y_predict

#def predict_a_imags(_image_path):
#    im = Image.open(_image_path)
#    images_pixel=(np.array(im)/255)
#    img_array = np.array(images_pixel,dtype='float32').reshape((1,224,224,3))
#    
#    model = load_model(final_model_weights_path)
#    y = model.predict(img_array,verbose=1)
#    return y

def correct_the_keras_map(_img_label_score_list):
    keras_map = {'0': 0, '1': 1, '10': 2, '11': 3, '12': 4, '13': 5, '14': 6, 
                 '15': 7, '16': 8, '17': 9, '18': 10, '19': 11, '2': 12, 
                 '20': 13, '21': 14, '22': 15, '23': 16, '24': 17, '25': 18, 
                 '26': 19, '27': 20, '28': 21, '29': 22, '3': 23, '30': 24, 
                 '31': 25, '32': 26, '33': 27, '34': 28, '35': 29, '36': 30, 
                 '37': 31, '38': 32, '39': 33, '4': 34, '40': 35, '41': 36, 
                 '42': 37, '43': 38, '44': 39, '45': 40, '46': 41, '47': 42, 
                 '48': 43, '49': 44, '5': 45, '50': 46, '51': 47, '52': 48, 
                 '53': 49, '54': 50, '55': 51, '56': 52, '57': 53, '58': 54, 
                 '59': 55, '6': 56, '60': 57, '61': 58, '62': 59, '63': 60, 
                 '64': 61, '65': 62, '66': 63, '67': 64, '68': 65, '69': 66, 
                 '7': 67, '70': 68, '71': 69, '72': 70, '73': 71, '74': 72, 
                 '75': 73, '76': 74, '77': 75, '78': 76, '79': 77, '8': 78, 
                 '80': 79, '81': 80, '82': 81, '83': 82, '84': 83, '85': 84, 
                 '86': 85, '87': 86, '88': 87, '89': 88, '9': 89, '90': 90, 
                 '91': 91, '92': 92, '93': 93, '94': 94, '95': 95, '96': 96, 
                 '97': 97, '98': 98, '99': 99}
    keras_map_list = []
    new_img_label_score_list = []
    for key,value in keras_map.items():
        keras_map_list.append((key,str(value)))
    
    for item in _img_label_score_list:
        for i in keras_map_list:
            if str(i[1]) == str(item[1]):
                new_label = i[0]
                new_img_label_score_list.append((item[0],new_label,item[2]))

    return new_img_label_score_list

predict_batch = 40
imgs_list = make_imgs_list(predict_imgs_path)
vgg16_predict = predict_imgs(imgs_list,predict_batch,vgg16_final_model_weights_path)
vgg19_predict = predict_imgs(imgs_list,predict_batch,vgg19_final_model_weights_path)
y_predict = (vgg16_predict+vgg19_predict)/2
array_max_list = find_array_max(y_predict)

img_label_score_list = make_img_label_score_list(imgs_list,array_max_list)
new_img_label_score_list = correct_the_keras_map(img_label_score_list)
write_result_to_txt(predict_txt_path,predict_probability_txt_path,new_img_label_score_list)

#y = predict_a_imags('imgs_predict_100c/26/73104746,4009991996.jpg')

#------------------------------------
time_end = time.time()
epis = time_end - time_start
print('used time:',int(epis/60),'mins',int(epis%60),'secs')






