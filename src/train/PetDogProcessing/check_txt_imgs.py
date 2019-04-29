from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import time
from collections import OrderedDict
'''生成标签的映射列表，去掉空余的标签'''
def make_label_map():
    label_list = []
    with open("E:/images_dog/val_NoLink.txt") as file:
        for line in file.readlines():
            label_list.append(int(line.split(' ')[1]))
    label_list = list(set(label_list))
    sorted(label_list)
    
    with open("E:/images_dog/val_mpa.txt",'a') as file:
        count = 0
        for item in label_list:
            file.write(str(item)+','+str(count)+'\n')
            count = count+1
    print(label_list)

'''找出一张图片对应多个标签的重复对应的项'''
def find_repetition_item():
    result_list = []
    img_name_list = []
    with open("E:/repetition_labels.txt") as file:
        for line in file.readlines():
            line = line.strip()
            img_name_list.append(line)
    print(img_name_list)
    
    for item in img_name_list:
        with open("E:/train.txt") as file:
            item_new = item
            for line in file.readlines():
                if line.split(' ')[0] == item:
                    item_new = item_new +' ' + line.split(' ')[1]
        result_list.append(item_new)
    print(result_list)
    
    with open("E:/repetition.txt",'a') as file:
        for item in result_list:
            file.write(item+'\n')


'''查找一个txt数据集里是否有一张图对应多个标签'''
def find_txt_one_to_more():
    img_name_list = []
    with open("E:/images_dog/val_NoLink.txt") as file:
        for line in file.readlines():
            line = line.strip().split(' ')[0]
            img_name_list.append(line)
    print(len(img_name_list))
    print(len(list(set(img_name_list))))
    
''' 生产一个txt文件,里面的标签的是映射之后的 '''
def make_maped_txt():
    map_list = []
    train_list = []
    result_list = []
    path_unmapped = 'E:/petdog训练集2/val_NoLink.txt'
    path_map = 'E:/petdog训练集2/val_map.txt'
    path_mapped = 'E:/petdog训练集2/val_mapped.txt'
    with open(path_unmapped) as file:
        for line in file.readlines():
            line = line.strip()
            train_list.append([line.split(' ')[0],line.split(' ')[1]])

    
    with open(path_map) as file:
        for line in file.readlines():
            line = line.strip()
            map_list.append([line.split(',')[0],line.split(',')[1]])
    
    for item in train_list:
        for i in map_list:
            if item[1] == i[0]:
                item[1] = i[1]
                result_list.append(item)
    print(result_list)
    with open(path_mapped,'a') as file:
        for item in result_list:
            file.write(item[0]+' '+item[1]+'\n')

'''去掉脏标签,生成新的txt'''
def make_txt_removed_dirty_labels():
    dirty_labels = []
    train_order_dic = OrderedDict() # 有序字典
    with open("E:/train_repetition.txt") as file:
        for line in file.readlines():
            line = line.strip()
            dirty_labels.append(line.split(' ')[0])
            
    with open("E:/train_maped.txt") as file:
        for line in file.readlines():
            line = line.strip()
            train_order_dic[line.split(' ')[0]] = line.split(' ')[1]
        
    print(len(dirty_labels))
    print(len(train_order_dic))
    
    for dirty in dirty_labels:
        del train_order_dic[dirty]
    print(len(train_order_dic))
    
    with open("E:/train_maped_remove_dirty_labels.txt",'a') as file:
        for key,value in train_order_dic.items():
            file.write(key+' '+ value +'\n')

                

def main():
    make_maped_txt()

if __name__ == "__main__":
    main()
