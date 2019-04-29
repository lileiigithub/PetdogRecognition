import time
time_start = time.time()

def get_map_list(_path):
    map_list = []
    with open(_path) as file:
        for line in file.readlines():
            line = line.strip()
            map_list.append((line.split(',')[0],line.split(',')[1]))
    return map_list

def get_predict_list(_path):
    predict_list = []
    with open(_path) as file:
        for line in file.readlines():
            line = line.strip()
            predict_list.append((line.split(' ')[0].split('.jpg')[0],line.split(' ')[1]))
    return predict_list

def modify_label(_map_list,_predict_list):
    modified_list = []
    for item in _predict_list:
        img = item[0]
        label = item[1]
        for item1 in _map_list:
            before_map = item1[0]
            after_map = item1[1]
            if label == after_map:
                modified_list.append((img,before_map))
                break
    return modified_list
            
def write_modified_txt(_modified_list,_path):
    with open(_path,'w') as file:
        for item in _modified_list:
            file.write(item[1]+'\t'+ item[0]+'\n')
            
map_path = "map.txt"
predict_path = "predict_vgg16_vgg19.txt"
modified_path = "predict_modified.txt"
#_____________________________________

map_list = get_map_list(map_path)
predict_list = get_predict_list(predict_path)
modified_list = modify_label(map_list,predict_list)
write_modified_txt(modified_list,modified_path)
#------------------------------------
time_end = time.time()
epis = time_end - time_start
print('used time:',int(epis/60),'mins',int(epis%60),'secs')








