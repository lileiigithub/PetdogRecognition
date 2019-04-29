import os
import shutil
import input_data

def make_label_list(_path):
    label_list = []
    with open(_path) as file:
        for line in file.readlines():
            line = line.strip()
            name = line.split(' ')[0]
            label = line.split(' ')[-1]
            label_list.append([name,label])
        return label_list
            
    
def make_label_dir(_dir_path,_dir_num):
    
    for i in range(_dir_num):
        path = _dir_path + str(i) 
        if os.path.exists(path):
            continue
        os.mkdir(path)

def copy_img_by_label(_src_path_img,_des_father_dir,_label):
    src_path = images_path+_src_path_img+".jpg"
    des_path = _des_father_dir+str(_label)
    shutil.copy(src_path,  des_path)


def copy_imgs_from_list(_list,_des_father_dir):
    for item in _list:
        copy_img_by_label(item[0],_des_father_dir,item[1])


def dict_to_list(_set):
    result_list = []
    for item_dic in _set:
        result_list.append([list(item_dic.keys())[0],list(item_dic.values())[0]])
    return result_list

def make_predict_list(_path):
    predict_list = []
    with open(_path) as file:
        for line in file.readlines():
            line = line.strip()
            predict_list.append((line.split(' ')[0],line.split(' ')[1]))
    return predict_list
NB_CLASSES = 100

images_path = "imgs_predict/"
path = 'predict.txt'
dir_predict = "imgs_predict_100c/"

make_label_dir(dir_predict,NB_CLASSES)

predict_list = make_predict_list(path)  

copy_imgs_from_list(predict_list,dir_predict) # 拷贝图片到指定分类的文件夹








