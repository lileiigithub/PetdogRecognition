from PIL import Image
import numpy as np
import time
import os

def find_pic_path(PATH):
    path_list = []
    for root, dirs, files in os.walk(PATH):
        for name in files:
            #print(os.path.join(root, name))
            path_list.append(os.path.join(root, name).replace('\\','/'))
    return path_list

def check_size(_infile):
    im = Image.open(_infile)
    if im.size[0] < 224:
        print(_infile.split('/')[-1])
        print(im.size)
    elif im.size[1] < 224:
        print(_infile.split('/')[-1])
        print(im.size)
    else:
        pass


time_start = time.time()
#------------------------------------
PATH_source = 'E:/petdog训练集1/train_3c_224/'

path_list = find_pic_path(PATH_source)
print(len(path_list))
print(path_list[0])
for one_path in path_list:
    check_size(one_path)
    
#------------------------------------
time_end = time.time()
epis = time_end - time_start
print('used time:',int(epis/60),'mins',int(epis%60),'secs')