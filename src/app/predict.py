# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from PIL import Image
from keras.models import load_model

'''
杈撳叆锛氬皢鍥剧墖鍒楄〃 
杩斿洖锛氬浘鍍忕偣鐨� array 绫诲瀷 (None.224,224,3)
'''
class PredictImg(object):
    
    def __init__(self,_img_array):
        #self.predict_img_path = "resize.jpg"
        self.IMG_SIZE = 224
        self.IMG_CHANNEL = 3
        self.final_model_weights_path = "model/final_model_100.h5"
        self.img_array = _img_array
        
    def find_array_max(self):
        
        _array = self.predict_a_imags()
        if _array.ndim is not 2:
            print("array ndim is error")
            return 0
        
        for i in range(_array.shape[0]): # 10593
            max_index1 = np.argmax(_array[i])
            max_value1 = _array[i].max()
            
            _array[i][max_index1] = 0
            max_index2 = np.argmax(_array[i])
            max_value2 = _array[i].max()
            
            _array[i][max_index2] = 0
            max_index3 = np.argmax(_array[i])
            max_value3 = _array[i].max()
            
        return (self.__map_index(self.__correct_index(max_index1)),max_value1,
                self.__map_index(self.__correct_index(max_index2)),max_value2,
                self.__map_index(self.__correct_index(max_index3)),max_value3)
    
    # 训练时 使用了keras的数据增强，真实标签与实际标签没有一一对应
    def __correct_index(self,_index):
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
        for key,value in keras_map.items():
            keras_map_list.append((int(key),(value)))
        #print(keras_map_list)
        for i in keras_map_list:
            if (i[1]) == _index:
                train_label = i[0]
        print("train label:",train_label)
        return train_label
    
    def __map_index(self,_index):
        label_map = []
        with open("map.txt") as file:
            for line in file.readlines():
                line = line.strip()
                bmap = int(line.split(',')[0])  # before bmap
                amap = int(line.split(',')[1])
                label_map.append((bmap,amap))
        for i in label_map:
            if (i[1]) == _index:
                real_label = i[0]
        return real_label
    
    def show_label_name(self,_index):
        name = ''
        label_name = []
        with open("label_name_1.txt") as file:  # ,encoding='utf-8'
            for line in file.readlines():
                line = line.strip()
                name = (line.split(' ')[0])  
                number = int(line.split(' ')[1])
                label_name.append((name,number))
        for i in label_name:
            if (i[1]) == _index:
                name = i[0] 
        return name
    
    def predict_a_imags(self):
        print("come in ")
        #im = Image.open(self.predict_img_path)
        im = self.img_array
        images_pixel=(np.array(im)/255)
        img_array = np.array(images_pixel,dtype='float32').reshape((1,self.IMG_SIZE,self.IMG_SIZE,self.IMG_CHANNEL))
        
        model = load_model(self.final_model_weights_path)
        y = model.predict(img_array,verbose=1)
        print("y:",y)
        return y
        
# p = PredictImg(1)
# name = p.show_label_name(18)
# print(name)



