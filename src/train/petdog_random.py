# -*- coding: utf-8 -*-  
""" 
Created on 2017/7/7

1.从input_data里读取数据，总图片被分成80%训练集和20%测试集；
2.循环CYCLE_NUM次，每次随机从训练集里抽取 CHOICE_IMAGES 张图片，用于训练模型；
3.训练完后用模型测试测试集，并给出准确率。
4.模型输入为 x_image 即图片， 输出为 y_ 及标签

5.做卷积的重要参数：
tensor = [batch,         in_height,    in_width,    in_channels]
filter = [filter_height, filter_width, in_channels, out_channels]
         [卷积核的高度，  卷积核的宽度， 图像通道数，  卷积核个数]
"""  
import numpy as np
import input_data  
import tensorflow as tf 
import time
time_start = time.time()

'''
定义一些常量
CLUSTER_NUM:输入图片种类
CHOICE_IMAGES：每次循环训练随机选取的图片
CYCLE_NUM: 循环运行的次数
'''

CHOICE_IMAGES = 100
CYCLE_NUM = 10001 # 90001
IMGE_SIZE = 256
########
CLUSTER_NUM,petdog_set = input_data.load_data()  # 读取数据
#有100 个种类， 但标签的范围是 0-133
print('CLUSTER_NUM:',CLUSTER_NUM)
CLUSTER_NUM = 100
sess = tf.InteractiveSession()  
#输入为 x_image 即图片， 输出为 y_ 及标签
x_image = tf.placeholder(tf.float32, shape=[None, IMGE_SIZE, IMGE_SIZE, 1])   #
y_ = tf.placeholder(tf.int64, shape=[None])   #

#CNN  
#我们会建立大量权重和偏置项，为了方便，定义初始函数  

def weight_variable(shape):  
  #tf.truncated_normal初始函数将根据所得到的均值和标准差，生成一个随机分布  
  initial = tf.truncated_normal(shape, stddev=0.1) 
  return tf.Variable(initial)  

def bias_variable(shape):  
  initial = tf.constant(0.1, shape=shape)  
  return tf.Variable(initial)  

# 计算程序运行时间
def used_time():
    time_end = time.time()
    epis = time_end - time_start
    used_time_str = str(int(epis/60))+' mins '+str(int(epis%60))+' secs'
    return 'used time: '+ used_time_str
# 将 中间结果 保存到数据
def write_log(_cycle_time,_accuracy,_precision=4,hint ='train'):
    with open('log.txt','a') as file:
        file.write(str(_cycle_time)+':'+time.ctime()+'\n')
        file.write(hint+"_accuracy:"+str(round(_accuracy,_precision))+"\n")
        file.write(used_time() +'\n')

#卷积和池化操作    
""" 
1. x是输入的样本，在这里就是图像。x的shape=[batch, height, width, channels]。  
- batch是输入样本的数量  
- height, width是每张图像的高和宽  
- channels是输入的通道，比如初始输入的图像是灰度图，那么channels=1，如果是rgb，那么channels=3。对于第二层卷积层，channels=32。  
2. W表示卷积核的参数，shape的含义是[height,width,in_channels,out_channels]。
3. strides参数表示的是卷积核在输入x的各个维度下移动的步长。
4. padding参数用来控制图片的边距，’SAME’表示卷积后的图片与原图片大小相同，’VALID’的话卷积以后图像的高为Heightout=Height原图−Height卷积核+1StrideHeight， 宽也同理。 
"""  
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
def conv2d(x, W, strade):  
  return tf.nn.conv2d(x, W, strides=[1, strade, strade, 1], padding='SAME') 
def max_pool_2x2(x):  
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],  
                        strides=[1, 2, 2, 1], padding='SAME')  

#卷积一  [256,256,1]-->[128,128,64]-->[64,64,64]
W_conv1 = weight_variable([8, 8, 1, 64])  
b_conv1 = bias_variable([64])  
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 2) + b_conv1)  
h_pool1 = max_pool_2x2(h_conv1)  

#卷积二  -->[16,16,128]
W_conv2 = weight_variable([4, 4, 64, 128])  
b_conv2 = bias_variable([128])  
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)  
h_pool2 = max_pool_2x2(h_conv2)  

#卷积三 -->[8,8,256]
W_conv3 = weight_variable([4, 4, 128, 256])  
b_conv3 = bias_variable([256])  
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, 2) + b_conv3)

#卷积四 -->[8,8,256]
W_conv4 = weight_variable([4, 4, 256, 256])  
b_conv4 = bias_variable([256])  
h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, 1) + b_conv4)

#卷积五  -->[4,4,128]
W_conv5 = weight_variable([4, 4, 256, 128])  
b_conv5 = bias_variable([128])  
h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5, 1) + b_conv5)
h_pool5 = max_pool_2x2(h_conv5)
#密集连接层  
""" 
神经元的全连接层，用于处理整个图片。
我们把输出的张量reshape成一些向量，乘上权重矩阵，加上偏置， 
然后对其使用ReLU 
relu为激活函数
"""
#  [4,4,128] -->[1,4096] 
# 4*4*128 = 2048
W_fc1 = weight_variable([2048, 4096])   
b_fc1 = bias_variable([4096])  
h_pool5_flat = tf.reshape(h_pool5, [-1, 2048])  #
h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)  
# [1,4096]-->[1,4096]
W_fc2 = weight_variable([4096, 4096])   
b_fc2 = bias_variable([4096])  
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2) 
#droput
"""
为了减少过拟合，在输出层之前加入dropout。 
用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。 

Dropout是指在模型训练时随机让网络某些隐含层节点的权重不工作， 
不工作的那些节点可以暂时认为不是网络结构的一部分， 
但是它的权重得保留下来（只是暂时不更新而已）， 
因为下次样本输入时它可能又得工作了 
"""
keep_prob = tf.placeholder("float")  
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)  

#输出层
# [1,2304]-->[1,CLUSTER_NUM]
W_out = weight_variable([4096, CLUSTER_NUM])  
b_out = bias_variable([CLUSTER_NUM])
y_conv=tf.matmul(h_fc2_drop, W_out) + b_out  # tf.nn.softmax

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                      (labels=y_,logits=y_conv)) # 
print('loss:',loss)
#cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv)) 

train_step = tf.train.AdamOptimizer(1e-2).minimize(loss) 

correct_prediction = tf.equal(tf.argmax(y_conv,1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
sess.run(tf.global_variables_initializer())  

saver = tf.train.Saver() # 声明tf.train.Saver类用于保存模型
for i in range(CYCLE_NUM):
    #每次从训练集里随机选取
    #以相同的概率从smple中选取CHOICE_IMAGES个量
    train_set = petdog_set['train_set']
    batch_set = np.random.choice(train_set, CHOICE_IMAGES, replace=False, p=[1/len(train_set)]*len(train_set)) 
    images_batch,labels_batch = input_data.seperate_img_lebel(batch_set)
    imgs_array_batch = input_data.imgs_to_array(images_batch)
    labels_array_batch = input_data.labels_to_array(labels_batch)
    train_accuracy = accuracy.eval(feed_dict={x_image:imgs_array_batch, y_: labels_array_batch, keep_prob: 1.0})   
    train_step.run(feed_dict={x_image: imgs_array_batch, y_: labels_array_batch, keep_prob: 0.8}) 

    if i%50 == 0 and i !=0:
        print ("step %d, training accuracy %g"%(i, train_accuracy))  
        print(used_time())
        write_log(i,train_accuracy)

    if i%500 == 0 and i !=0 :
        saver_path = saver.save(sess, "save/model_"+str(i)+".ckpt")  # 将模型保存

    if i%1000 == 0 and i !=0 :
        test_set = petdog_set['test_set']
        images_test,labels_test = input_data.seperate_img_lebel(test_set)
        imgs_array_test = input_data.imgs_to_array(images_test)
        labels_array_test = input_data.labels_to_array(labels_test)
        test_accuracy = accuracy.eval(feed_dict={x_image: imgs_array_test, y_: labels_array_test, keep_prob: 1.0}) 
        write_log(i,test_accuracy,5,hint='test')

#用训练后的模型来测试 测试集 
test_set = petdog_set['test_set']
images_test,labels_test = input_data.seperate_img_lebel(test_set)
imgs_array_test = input_data.imgs_to_array(images_test)
labels_array_test = input_data.labels_to_array(labels_test)
test_accuracy = accuracy.eval(feed_dict={x_image: imgs_array_test, y_: labels_array_test, keep_prob: 1.0}) 

saver_path = saver.save(sess, "save/model_final.ckpt")  # 保存最终模型
# 记录最后测试集测试的结果
print ("test accuracy %g"%test_accuracy)
print(used_time())
write_log(CYCLE_NUM,test_accuracy,5,hint='test')










    
