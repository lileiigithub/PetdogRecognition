predict_vgg16_path = "predict_modified.txt"
predict_vgg16_vgg19_path = r"C:\Users\LiLei\Desktop\predict_modified.txt"
vgg16_list = []
vgg19_list = []

def read_line(_path):
    _list = []
    with open(_path) as file:
        for line in file.readlines():
            line = line.strip()
            _list.append(line)#(line.split(' ')[0],line.split(' ')[1],line.split(' ')[2])
    return _list

def find_different(_a_list,_b_list):
    if len(_a_list) != len(_b_list):
        print("format is error.")
        return -1
    count = 0
    for i in range(len(_a_list)):
        if _a_list[i][1]!=_b_list[i][1] : #and float(_a_list[i][2])<float(_b_list[i][2])
            count = count+1
            print(count,_a_list[i],_b_list[i])


vgg16_list = read_line(predict_vgg16_path)
vgg19_list = read_line(predict_vgg16_vgg19_path)
find_different(vgg16_list,vgg19_list)


























