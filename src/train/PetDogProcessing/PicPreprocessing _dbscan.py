from PIL import Image
import numpy as np
from sklearn.cluster import DBSCAN
import time
time_start = time.time()


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

EPS = 100
MINPTS = int(row*col/5)
db = DBSCAN(eps=EPS, min_samples=MINPTS).fit(imgData)
#core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
# calculate the number of clusters
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

for one_cluster in range(n_clusters_):
    img_new = np.full((imgData.shape), fill_value = 255, dtype = np.uint8)
    img_new1 = np.full((imgData.shape), fill_value = 255, dtype = np.uint8)
    for item in range(labels.shape[0]):
            if labels[item] == one_cluster:
                img_new[item] = imgData[item]
                img_new1[item] = img_new[item]
    pic_new = Image.fromarray(img_new1.reshape((row,col,3)))
    path = 'E:/test/_result' + str(one_cluster) +'.jpg'
    pic_new.save(path)


time_end = time.time()
print(int((time_end - time_start)),'seconds')






