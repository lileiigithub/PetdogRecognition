from PIL import Image
import numpy as np
import time


'''将txt里的数据 转为{img,label}的字典格式'''
def make_img_label_dic(_path):
    dic_list = []
    file = open(_path)
    total_imgs = 0
    for item in file.readlines():
        item = item.strip()
        total_imgs = total_imgs + 1
        key = item.split(' ')[0]
        value = item.split(' ')[1]
        dic_list.append({key:value})
    print('readed items from txt:',total_imgs)
    return dic_list

def seperate_img_lebel(_dic_list):
    imgs_list = []
    labels_list = []
    for item in _dic_list:
        imgs_list.append(list(item.keys())[0])
        labels_list.append(int(list(item.values())[0]))
    return imgs_list,labels_list

'''
输入：将图片列表 
返回：图像点的 array 类型 (None.224,224,3)
'''
def imgs_to_array(_img_list):
    images_pixel_list = []
    for item in _img_list:
        PATH = PATH_imgs + item + '.jpg'
        im = Image.open(PATH)
        images_pixel_list.append(np.array(im)/255)  # 图片像素点归一化到 0~1
    return np.array(images_pixel_list,dtype='float32').reshape((len(_img_list),IMG_SIZE,IMG_SIZE,IMG_CHANNEL))

'''将标签列表转为array类型'''
def labels_to_array(_label_list):
    return np.array(_label_list)

'''
将从txt中读取到的{图像，标签}列表按照标签分类
'''
def classify_with_labels(_dic_list):
    classified_list = []
    head_index = 1
    end_index = 0
    temp_value = list(_dic_list[0].values())[0]
    now_index = 0
    for item in _dic_list:
        now_index = now_index + 1
        now_value = list(item.values())[0]  # examps: '0'
        if now_value == temp_value:
            pass
        else:
            end_index = now_index
            temp_value = now_value
            classified_list.append(_dic_list[head_index-1:end_index-1])
            head_index = now_index
    classified_list.append(_dic_list[head_index-1:])
    
    total_imags = 0
    for item in classified_list:
        total_imags = total_imags + len(item)
    print('classified total imags:',total_imags)
    return classified_list

'''将{图像，标签}字典以SCALE的比例分为测试集和训练集,
只要dataset.txt不变，scale不变，每次生成相同测试集'''
def make_train_test(_classified_list,_SCALE):
    #clusters_num = len(_classified_list)
    train_dic_list = []
    test_dic_list = []
    for item in _classified_list:
        end_index = int(len(item)*_SCALE)
        train_dic_list.extend(list(item)[0:end_index])
        test_dic_list.extend(list(item)[end_index:])
    
    return train_dic_list,test_dic_list

'''用于外部调用数据的函数 ，返回一个字典'''
def load_data():
    # 将数据集分为 训练集 和 测试集
    dic_list = make_img_label_dic(PATH_txt)
    classified_list = classify_with_labels(dic_list)
    train_set,test_set = make_train_test(classified_list,SCALE)
    # ##
    # 总共的类 数量
    #CLUSTER_NUM = len(classified_list)
    return {'train_set':train_set,'test_set':test_set}

'''从图片名字转换到地址'''
def get_img_path_from_path(_name):
    path = PATH_imgs + _name + '.jpg'
    return path

time_start = time.time()
#------------------------------------

CLUSTER_NUM = 2  # 总标签数
PATH_imgs = 'data/'
PATH_txt =  'dataset_'+str(CLUSTER_NUM)+'c.txt'

SCALE = 0.8  #训练集占总图像集的比例
IMG_SIZE = 224 # 读取的图片的大小
IMG_CHANNEL = 3
#train = []  #训练的字典
#test = []  #测试的字典
#dic_list = make_img_label_dic(PATH_txt)
#classified_list = classify_with_labels(dic_list)
#train,test = make_train_test(classified_list,SCALE)
#img_list,la = seperate_img_lebel(test)
#array_ = imgs_to_array(img_list[:100])
#result_dic = load_data()
#------------------------------------
time_end = time.time()
epis = time_end - time_start
print('used time:',int(epis/60),'mins',int(epis%60),'secs')





