path = 'E:/images_dog/val.txt'
path1 = 'E:/images_dog/val_NoLink.txt'
file = open(path, 'r')
str_new = ""
list_line = file.readlines()
for line in list_line:
    str_new = str_new + line.split('http')[0] +'\n'
file.close()

file1 = open(path1, 'a')
file1.writelines(str_new)
file1.close()


