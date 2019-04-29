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


def clip_thunmnail(_infile):   
    outfile = PATH_result + _infile.split('/')[-1]   
    size = (224, 224)   # the thumbnail size       
    global count_size_little
    try:
        im = Image.open(_infile)
        x_line = im.size[0]
        y_line = im.size[1]

        if x_line > y_line:
            box = (int((x_line-y_line)/2),0,int((x_line-y_line)/2)+y_line,y_line)
        elif x_line < y_line:
            box = (0,int((y_line-x_line)/2),x_line,int((y_line-x_line)/2)+x_line)
        else:
            box = (0,0,x_line,y_line) 
        region = im.crop(box)  # clip a region
        region.thumbnail(size)  # make a thumbnail
        # 判断图片 x,y 是否小于 256
        if region.size[0] < 224 or region.size[1] < 224:
            count_size_little = count_size_little + 1
            region = region.resize((224, 224),Image.ANTIALIAS)
            print(str(count_size_little)+'(<224:)',_infile.split('/')[-1])
        
        region.save(outfile, "JPEG")
    except IOError:
        print("cannot create thumbnail for", _infile)


time_start = time.time()
#------------------------------------
PATH_source = 'E:/petdog测试集/image/'
PATH_result =  'E:/petdog测试集/test_3c_224/'
count_size_little = 0
path_list = find_pic_path(PATH_source)
print(len(path_list))
print('example:',path_list[0])
for one_path in path_list:
    clip_thunmnail(one_path)
#------------------------------------
time_end = time.time()
epis = time_end - time_start
print('used time:',int(epis/60),'mins',int(epis%60),'secs')








