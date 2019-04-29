from PIL import Image
import numpy as np
import time
import os

def make_tag_dic(_path):
    dic_list = []
    file = open(_path)
    for item in file.readlines():
        key = item.split(' ')[0]
        value = item.split(' ')[1]
        dic_list.append({key:value})
    return dic_list

def load_data():
    dic_list = make_tag_dic(PATH_txt)
    print(len(dic_list))
    train,test = make_tran_test(dic_list,SCALE)
    
    train_imgs_list = []
    train_labels_list = []
    test_imgs_list = []
    test_labels_list = []
    # 将训练集，测试集字典 里的图像与标签分离
    for item in train:
        for i in item:
            train_imgs_list.append(list(i.keys())[0])
            train_labels_list.append(int(list(i.values())[0]))
    for item in test:
        for i in item:
            test_imgs_list.append(list(i.keys())[0])
            test_labels_list.append(int(list(i.values())[0]))

    # 图像 转为 array
    images_train_list = []
    images_test_list = []
    for item in train_imgs_list:
        PATH = 'I:/python/tensorflow_petdog/images1/'+ item + '.jpg'
        images_train_list.extend(img_line_list(PATH))
    images_train = np.array(images_train_list).reshape(727,80*80)
    
    for item in test_imgs_list:
        PATH = 'I:/python/tensorflow_petdog/images1/'+ item + '.jpg'
        images_test_list.extend(img_line_list(PATH))
    images_test = np.array(images_test_list).reshape(184,80*80)
    
    labels_train = np.array(train_labels_list)
    labels_test = np.array(test_labels_list)
    
    return {'images_train':images_train,'labels_train':labels_train,
            'images_test':images_test,'labels_test':labels_test}

def img_line_list(_infile):
    im = Image.open(_infile)
    return list(np.array(im,dtype='float32').reshape(80*80))

'''将{图像，标签}字典以SCALE的比例分为测试集和训练集'''
def make_tran_test(_dic_list,_SCALE):
    count_list = [0]*11
    for item in _dic_list:
        count_list[int(list(item.values())[0])] = count_list[int(list(item.values())[0])]+1
    
    list_list = []
    accum_old = 0
    accum_new = 0
    for i in range(11):
        accum_new = accum_old + count_list[i]
        temp_list = _dic_list[accum_old:accum_new]
        accum_old = accum_new
        list_list.append(temp_list)
    
    for item in list_list:
        train.append(list(item)[0:int(len(item)*_SCALE)])
        test.append(list(item)[int(len(item)*_SCALE):])
        
    return train,test


time_start = time.time()
#------------------------------------
PATH_source = 'I:/python/tensorflow_petdog/images/'
PATH_txt =  'I:/python/tensorflow_petdog/dataset1.txt'
DES_PATH = 'I:/python/tensorflow_petdog/images1'
SCALE = 0.8
train = []  #训练的字典
test = []  #测试的字典

#dic_list = make_tag_dic(PATH_txt)
#print(len(dic_list))
#train,test = make_tran_test(dic_list,SCALE)
#a_dic = load_data()


#------------------------------------
time_end = time.time()
epis = time_end - time_start
print('used time:',int(epis/60),'mins',int(epis%60),'secs')






