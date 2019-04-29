# -*- coding: utf-8 -*-  
""" 
Created on Sat Apr  1 14:57:40 2017 
 
tensor = [batch,         in_height,    in_width,    in_channels]
filter = [filter_height, filter_width, in_channels, out_channels]
         [卷积核的高度，  卷积核的宽度， 图像通道数，  卷积核个数]
"""  
import time
time_start = time.time()
import numpy as np
import input_data  
mnist = input_data.load_data()  
#
import tensorflow as tf  
sess = tf.InteractiveSession()  
saver = tf.train.Saver(write_version=tf.train.SaverDef.V1) # 声明tf.train.Saver类用于保存模型
#
#

x_image = tf.placeholder(tf.float32, shape=[None, 80, 80, 1])   # shape=[None, 784]
y_ = tf.placeholder(tf.int64, shape=[None])   # shape=[None, 10]
#
#W = tf.Variable(tf.zeros([784,10]))  
#b = tf.Variable(tf.zeros([10]))  
#  
#sess.run(tf.initialize_all_variables())  
##一次性初始化所有变量  
#  
#y = tf.nn.softmax(tf.matmul(x,W) + b)  
#  
#cross_entropy = -tf.reduce_sum(y_*tf.log(y))  
#  
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)  
#  
#for i in range(1000):  
#  batch = mnist.train.next_batch(50)  
#  train_step.run(feed_dict={x: batch[0], y_: batch[1]})  
# 
#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  
#  
#print (accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))  


#CNN  
#我们会建立大量权重和偏置项，为了方便，定义初始函数  

def weight_variable(shape):  
  #tf.truncated_normal初始函数将根据所得到的均值和标准差，生成一个随机分布  
  initial = tf.truncated_normal(shape, stddev=0.1) 
  return tf.Variable(initial)  
  
def bias_variable(shape):  
  initial = tf.constant(0.1, shape=shape)  
  return tf.Variable(initial)  
""" 
由于我们使用的是ReLU神经元，因此比较好的做法是用一个较小的正数来初始化偏置项， 
以避免神经元节点输出恒为0的问题（dead neurons） 
"""  

#卷积池化操作    

""" 
1. x是输入的样本，在这里就是图像。x的shape=[batch, height, width, channels]。  
- batch是输入样本的数量  
- height, width是每张图像的高和宽  
- channels是输入的通道，比如初始输入的图像是灰度图，那么channels=1，如果是rgb，那么channels=3。对于第二层卷积层，channels=32。  
2. W表示卷积核的参数，shape的含义是[height,width,in_channels,out_channels]。  
3. strides参数表示的是卷积核在输入x的各个维度下移动的步长。
  了解CNN的都知道，在宽和高方向stride的大小决定了卷积后图像的size。
  这里为什么有4个维度呢？因为strides对应的是输入x的维度，所以strides第一个参数表示在batch方向移动的步长，第四个参数表示在channels上移动的步长，这两个参数都设置为1就好。
  重点就是第二个，第三个参数的意义，也就是在height于width方向上的步长，这里也都设置为1。  
4. padding参数用来控制图片的边距，’SAME’表示卷积后的图片与原图片大小相同，’VALID’的话卷积以后图像的高为Heightout=Height原图−Height卷积核+1StrideHeight， 宽也同理。 
"""  
def conv2d(x, W, strade):  
  return tf.nn.conv2d(x, W, strides=[1, strade, strade, 1], padding='SAME')  

"""                        
这里用2∗2的max_pool。参数ksize定义pool窗口的大小，每个维度的意义与之前的strides相同 
第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape 
第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1 
第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1] 
第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME' 
返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式 
这个函数的功能是将整个图片分割成2x2的块， 
对每个块提取出最大值输出。可以理解为对整个图片做宽度减小一半，高度减小一半的降采样 
"""  
def max_pool_2x2(x):  
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],  
                        strides=[1, 2, 2, 1], padding='SAME')  

#x_image = tf.reshape(x, [None,80,80,1])  # [100,80,80,1]
""" 
为了用这一层，我们把x变成一个4d向量，其第2、第3维对应图片的宽、高， 
最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3) 
第一维-1代表将x沿着最后一维进行变形 
""" 

#卷积一  
W_conv1 = weight_variable([8, 8, 1, 32])  
b_conv1 = bias_variable([32])  
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 2) + b_conv1)  
h_pool1 = max_pool_2x2(h_conv1)  

#卷积二  
W_conv2 = weight_variable([4, 4, 32, 64])  
b_conv2 = bias_variable([64])  
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)  
h_pool2 = max_pool_2x2(h_conv2)  

#卷积三 
W_conv3 = weight_variable([3, 3, 64, 64])  
b_conv3 = bias_variable([64])  
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, 1) + b_conv3)
#密集连接层  
""" 
神经元的全连接层，用于处理整个图片。
我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置， 
然后对其使用ReLU 
"""
W_fc1 = weight_variable([1600, 512])   # [7 * 7 * 64, 1024]
b_fc1 = bias_variable([512])  

h_pool2_flat = tf.reshape(h_conv3, [-1, 1600])  #[1,]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  

#droput  
""" 
为了减少过拟合，我们在输出层之前加入dropout。 
我们用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。 
这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。  
TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的输出外， 
还会自动处理神经元输出值的scale。所以用dropout的时候可以不用考虑scale 

Dropout是指在模型训练时随机让网络某些隐含层节点的权重不工作， 
不工作的那些节点可以暂时认为不是网络结构的一部分， 
但是它的权重得保留下来（只是暂时不更新而已）， 
因为下次样本输入时它可能又得工作了 
"""
keep_prob = tf.placeholder("float")  
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  

#输出层  
W_fc2 = weight_variable([512, 11])  
b_fc2 = bias_variable([11])  

y_conv=tf.matmul(h_fc1_drop, W_fc2) + b_fc2  # tf.nn.softmax

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                      (labels=y_,logits=y_conv)) # 

#cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv)) 

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss) 

correct_prediction = tf.equal(tf.argmax(y_conv,1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
sess.run(tf.global_variables_initializer())  

SUM = mnist['images_train'].shape[0]
SCALE = 100
index_list = [i for i in range(0,SUM,SCALE)]

for i in range(len(index_list)-1):
        #batch = mnist.train.next_batch(50)
        #indices = np.random.choice(mnist['images_train'].shape[0], 100)
    if i <= len(index_list)-2:
        images_batch = mnist['images_train'][index_list[i]:index_list[i+1]]
        labels_batch = mnist['labels_train'][index_list[i]:index_list[i+1]]
    else:
        images_batch = mnist['images_train'][index_list[i]:]
        labels_batch = mnist['labels_train'][index_list[i]:]

    images_batch = images_batch.reshape((100,80,80,1))
    
    train_accuracy = accuracy.eval(feed_dict={x_image:images_batch, y_: labels_batch, keep_prob: 1.0})  
#    print ("step %d, training accuracy %g"%(i, train_accuracy))  
    train_step.run(feed_dict={x_image: images_batch, y_: labels_batch, keep_prob: 0.5})  
    if i%100 == 0:
        #train_accuracy = accuracy.eval(feed_dict={x_image:images_batch, y_: labels_batch, keep_prob: 1.0})  
        print ("step %d, training accuracy %g"%(i, train_accuracy))  
        #train_step.run(feed_dict={x_image: images_batch, y_: labels_batch, keep_prob: 0.5})  

    if i%1000 == 0:
        time_end = time.time()
        epis = time_end - time_start
        print('used time:',int(epis/60),'mins',int(epis%60),'secs')
        saver_path = saver.save(sess, "save/model_"+str(i)+".ckpt")  # 将模型保存

images_test = mnist['images_test']
images_test = images_test.reshape((images_test.shape[0],80,80,1))

print ("test accuracy %g"%accuracy.eval(feed_dict={  
    x_image: images_test, y_: mnist['labels_test'], keep_prob: 1.0}))  

saver_path = saver.save(sess, "save/model_final.ckpt")  # 保存最终模型
time_end = time.time()
epis = time_end - time_start
print('used time:',int(epis/60),'mins',int(epis%60),'secs')


    