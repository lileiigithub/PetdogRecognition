from PIL import Image
import numpy as np
import time
import os

PATH_txt1 =  'E:/images_dog/val_NoLink.txt'
PATH_txt2 =  'E:/dataset.txt'
CLUSTER_indexs = 134  # 总标签数

def make_tag_dic(_path):
    dic_list = []
    with open(_path) as file:
        for item in file.readlines():
            key = item.split(' ')[0]
            value = item.split(' ')[1]
            dic_list.append({key:value})
        return dic_list
def count_clusters(dic_list):
    count_list = [0]*CLUSTER_indexs
    for item in dic_list:
            count_list[int(list(item.values())[0])] = count_list[int(list(item.values())[0])]+1
    print(len(count_list))
    return count_list

imag1_list = count_clusters(make_tag_dic(PATH_txt1))
imag2_list = count_clusters(make_tag_dic(PATH_txt2))
imag_compare_list = []

for i in range(CLUSTER_indexs):
    imag_compare_list.append([imag1_list[i],imag2_list[i]])

print(imag_compare_list)
print(len(imag_compare_list))

count_zero = 0
count_non_zero = 0
for item in imag_compare_list:
    if item[0] == 0:
        count_zero = count_zero + 1
        print(item)
    elif item[1] == 0:
        count_zero = count_zero + 1
        print(item)
    else:
        count_non_zero = count_non_zero + 1

print('zero',count_zero)
print('non_zero',count_non_zero)
