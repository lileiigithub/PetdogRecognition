from PIL import Image
import numpy as np
import time
import os
import shutil

def find_pic_path(PATH):
    path_list = []
    for root, dirs, files in os.walk(PATH):
        for name in files:
            #print(os.path.join(root, name))
            path_list.append(os.path.join(root, name).replace('\\','/'))
    return path_list


def read_tag(_PATH):
    picname_list = []
    file = open(_PATH)
    for item in file.readlines():
        picname_list.append(item.split(' ')[0])
    return picname_list;


time_start = time.time()
#------------------------------------
PATH_source = 'I:/python/tensorflow_petdog/images/'
PATH_txt =  'I:/python/tensorflow_petdog/dataset1.txt'
DES_PATH = 'I:/python/tensorflow_petdog/images1'
picname_list = read_tag(PATH_txt)
img_list = find_pic_path(PATH_source)
for name in picname_list:
    src_path = 'I:/python/tensorflow_petdog/images/'+ name +'.jpg'
    shutil.copy(src_path,DES_PATH)

#path_list = find_pic_path(PATH_source)
#print(len(path_list))
#print('example:',path_list[0])
#for one_path in path_list:

#------------------------------------
time_end = time.time()
epis = time_end - time_start
print('used time:',int(epis/60),'mins',int(epis%60),'secs')




