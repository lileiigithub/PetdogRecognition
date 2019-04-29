# -*- coding: utf-8 -*-
from PIL import Image

class ProcessImg(object):
    size = (224, 224)
    
    def __init__(self,_path):
        self.img_path = _path
        
        
    def resize(self):

        try:
            im = Image.open(self.img_path)
            x = im.size[0]
            y = im.size[1]
            if x > y:
                box = (int((x-y)/2),0,int((x-y)/2)+y,y)
            elif x < y:
                box = (0,int((y-x)/2),x,int((y-x)/2)+x)
            else:
                box = (0,0,x,y) 
            region = im.crop(box)  # clip a region
            region.thumbnail(ProcessImg.size)  # make a thumbnail
            # 判断图片 x,y 是否小于 224
            if region.size[0] < x or region.size[1] < y:
                region = region.resize(ProcessImg.size,Image.ANTIALIAS)
            region.save("resize.jpg", "JPEG")
            return region
        except IOError:
            print("cannot create thumbnail for", self.img_path)









