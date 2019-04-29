from PIL import Image
import time
import os


def find_pic_path(PATH):
    path_list = []
    for root, dirs, files in os.walk(PATH):
        for name in files:
            #print(os.path.join(root, name))
            path_list.append(os.path.join(root, name).replace('\\','/'))
    return path_list

def rgb2gray(_infile): 
    outfile = PATH_result + _infile.split('/')[-1]
    im = Image.open(_infile)
    im = im.convert('L')
    im.save(outfile)
    
time_start = time.time()
#---------------------------------------------------

PATH_source = 'E:/petdog训练集2/val_3c/'
PATH_result =  'E:/petdog训练集2/val_1c/'
path_list = find_pic_path(PATH_source)
print(len(path_list))
print(path_list[0])
for one_path in path_list:
    rgb2gray(one_path)

#---------------------------------------------------
time_end = time.time()
epis = time_end - time_start
print('used time:',int(epis/60),'mins',int(epis%60),'secs')

