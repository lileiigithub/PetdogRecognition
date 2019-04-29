import os
root_dir = "I:/Python/tensorflow_petdog/imgs_100c-half/train"

def get_dir_list_path(_root_path):
    dir_list = []
    dir_list_path = []
    dir_list = os.listdir(_root_path)
    for item in dir_list:
        item = root_dir+'/'+item
        dir_list_path.append(item)
    return dir_list_path

#input: I:/Python/tensorflow_petdog/imgs_100c - half/train/0
def get_img_list_path(_a_dir_path):
    dir_img_list_path = []
    dir_img_list = os.listdir(_a_dir_path)
    for item in dir_img_list:
        item = _a_dir_path+'/'+item
        dir_img_list_path.append(item)

    return dir_img_list_path

#input: [I:/Python/tensorflow_petdog/imgs_100c - half/train/0/1003645637,3600678665.jpg,...]
def delete_some_imgs(_img_path_list): 
    global count
    imgs_num = len(_img_path_list)
    needless_imgs_path = []
    if imgs_num > limited_num:
        count = count+1
        needless_imgs_path = _img_path_list[limited_num:]
        for item in needless_imgs_path:
            os.remove(item)


count = 0
limited_num = 60
dir_list_path = get_dir_list_path(root_dir)
for dir_item in dir_list_path:
    dir_img_list_path = get_img_list_path(dir_item)
    delete_some_imgs(dir_img_list_path)






