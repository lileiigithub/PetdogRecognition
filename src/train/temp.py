#import numpy as np
#import tensorflow as tf
#import time
#
#input = tf.Variable(tf.random_normal([100,256,256,1]))
#
#filter1 = tf.Variable(tf.random_normal([8,8,1,64]))  
#conv1 = tf.nn.conv2d(input, filter1, strides=[1, 2, 2, 1], padding='VALID') 
#pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
#
#filter2 = tf.Variable(tf.random_normal([4, 4, 64, 128]))  
#conv2 = tf.nn.conv2d(pool1, filter2, strides=[1, 2, 2, 1], padding='VALID') 
#pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
#
#filter3 = tf.Variable(tf.random_normal([4, 4, 128, 256]))  
#conv3 = tf.nn.conv2d(pool2, filter3, strides=[1, 2, 2, 1], padding='VALID')
#
#filter4 = tf.Variable(tf.random_normal([4, 4, 256, 256]))  
#conv4 = tf.nn.conv2d(conv3, filter4, strides=[1, 1, 1, 1], padding='VALID')
#
#filter5 = tf.Variable(tf.random_normal([4, 4, 256, 128]))  
#conv5 = tf.nn.conv2d(conv4, filter5, strides=[1, 1, 1, 1], padding='VALID') 
#pool5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
#
##tf.nn.max_pool(x, ksize=[1, 2, 2, 1],  
##                        strides=[1, 2, 2, 1], padding='SAME')
##x = tf.placeholder(tf.float32, shape=[None, 6400])
##x = tf.Variable(tf.random_normal([100,6400])) 
##x_image = tf.reshape(x, [100,80,80,1]) 
#
#init = tf.global_variables_initializer()  
#with tf.Session() as sess:
#    sess.run(init)
#    print('conv1',sess.run(conv1).shape)
#    print('pool1',sess.run(pool1).shape)
#    print('conv2',sess.run(conv2).shape)
#    print('pool2',sess.run(pool2).shape)
#    print('conv3',sess.run(conv3).shape)
#    print('conv4',sess.run(conv4).shape)
#    print('conv5',sess.run(conv5).shape)
#    print('pool5',sess.run(pool5).shape)
#    #print('pool',sess.run(pool2).shape)
#    #print(sess.run(x_image).shape)

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

img = load_img('dog.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
datagen.fit(x)
# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely














