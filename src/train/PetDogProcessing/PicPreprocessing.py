from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import time
#img = np.array(Image.open('E:/test/1.jpg'))
#print(img.shape, img.dtype)
time_start = time.time()

N_CLUSTERS = 2

def loadData(filePath):
    f = open(filePath,'rb')
    data = []
    img = Image.open(f)
    m,n = img.size
    for i in range(m):
        for j in range(n):
            x,y,z = img.getpixel((i,j))
            data.append([x,y,z])   #  data.append([i,j,x,y,z])
    f.close()
    return np.mat(data),m,n


imgData,row,col = loadData('E:/test/5.jpg')
label = KMeans(n_clusters=N_CLUSTERS).fit_predict(imgData)

for one_cluster in range(N_CLUSTERS):
    img_new = np.full((imgData.shape), fill_value = 255, dtype = np.uint8)
    img_new1 = np.full((imgData.shape), fill_value = 255, dtype = np.uint8)
    for item in range(label.shape[0]):
            if label[item] == one_cluster:
                img_new[item] = imgData[item]
                img_new1[item] = img_new[item]
    pic_new = Image.fromarray(img_new1.reshape((row,col,3)))
    path = 'E:/test/_result' + str(one_cluster) +'.jpg'
    pic_new.save(path)


time_end = time.time()
print(int((time_end - time_start)),'seconds')






