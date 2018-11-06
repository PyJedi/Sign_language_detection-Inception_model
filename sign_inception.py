# MNIST sign language dataset training using tensorflow implementation of inception model
# Written for Google Cloud ML 
# Change the bucket in the path accordingly 

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from pandas.compat import StringIO

def read_data(gcs_path):
   print('downloading csv file from', gcs_path)     
   file_stream = file_io.FileIO(gcs_path, mode='r')
   data = pd.read_csv(StringIO(file_stream.read()))
   # print(data.head())
   return data

train_set = read_data('gs://cmb/sign_train.csv')
train_set = train_set.iloc[:, :].values
test_set = read_data('gs://cmb/sign_test.csv')
test_set = test_set.iloc[:, :].values

#get labels in own array
train_lb=np.array(train_set[0])
test_lb=np.array(test_set[0])

#one hot encode the labels
train_lb=(np.arange(26) == train_lb[:,None]).astype(np.float32)
test_lb=(np.arange(26) == test_lb[:,None]).astype(np.float32)
 
#drop the labels column from training dataframe
trainX=train_set[:, 1:]
testX=test_set[:, 1:]
 
#put in correct float32 array format
trainX=np.array(trainX).astype(np.float32)
testX=np.array(testX).astype(np.float32)

#reformat the data so it's not flat
trainX=trainX.reshape(len(trainX),28,28,1)
testX = testX.reshape(len(testX),28,28,1)

'''
#get a validation set and remove it from the train set
trainX,valX,train_lb,val_lb=trainX[0:(len(trainX)-500),:,:,:],trainX[(len(trainX)-500):len(trainX),:,:,:],\
                            train_lb[0:(len(trainX)-500),:],train_lb[(len(trainX)-500):len(trainX),:]
'''
#need to batch the test data because running low on memory
class test_batchs:
    def __init__(self,data):
        self.data = data
        self.batch_index = 0
    def nextBatch(self,batch_size):
        if (batch_size+self.batch_index) > self.data.shape[0]:
            print "batch sized is messed up"
        batch = self.data[self.batch_index:(self.batch_index+batch_size),:,:,:]
        self.batch_index= self.batch_index+batch_size
        return batch
 
#set the test batchsize
test_batch_size = 100

#returns accuracy of model
def accuracy(target,predictions):
    return(100.0*np.sum(np.argmax(target,1) == np.argmax(predictions,1))/target.shape[0])

batch_size = 50
map1 = 32
map2 = 64
num_fc1 = 700 #1028
num_fc2 = 26
reduce1x1 = 16
dropout=0.5

graph = tf.Graph()
with graph.as_default():
    #train data and labels
    X = tf.placeholder(tf.float32,shape=(batch_size,28,28,1))
    y_ = tf.placeholder(tf.float32,shape=(batch_size,26))
 
    #validation data
    # tf_valX = tf.placeholder(tf.float32,shape=(len(valX),28,28,1))
 
    #test data
    tf_testX=tf.placeholder(tf.float32,shape=(test_batch_size,28,28,1))
 
    def createWeight(size,Name):
        return tf.Variable(tf.truncated_normal(size, stddev=0.1),
                          name=Name)
 
    def createBias(size,Name):
        return tf.Variable(tf.constant(0.1,shape=size),
                          name=Name)
 
    def conv2d_s1(x,W):
        return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
 
    def max_pool_3x3_s1(x):
        return tf.nn.max_pool(x,ksize=[1,3,3,1],
                             strides=[1,1,1,1],padding='SAME')
 
    #Inception Module1
    #
    #follows input
    W_conv1_1x1_1 = createWeight([1,1,1,map1],'W_conv1_1x1_1')
    b_conv1_1x1_1 = createWeight([map1],'b_conv1_1x1_1')
 
    #follows input
    W_conv1_1x1_2 = createWeight([1,1,1,reduce1x1],'W_conv1_1x1_2')
    b_conv1_1x1_2 = createWeight([reduce1x1],'b_conv1_1x1_2')
 
    #follows input
    W_conv1_1x1_3 = createWeight([1,1,1,reduce1x1],'W_conv1_1x1_3')
    b_conv1_1x1_3 = createWeight([reduce1x1],'b_conv1_1x1_3')
 
    #follows 1x1_2
    W_conv1_3x3 = createWeight([3,3,reduce1x1,map1],'W_conv1_3x3')
    b_conv1_3x3 = createWeight([map1],'b_conv1_3x3')
 
    #follows 1x1_3
    W_conv1_5x5 = createWeight([5,5,reduce1x1,map1],'W_conv1_5x5')
    b_conv1_5x5 = createBias([map1],'b_conv1_5x5')
 
    #follows max pooling
    W_conv1_1x1_4= createWeight([1,1,1,map1],'W_conv1_1x1_4')
    b_conv1_1x1_4= createWeight([map1],'b_conv1_1x1_4')
 
    #Inception Module2
    #
    #follows inception1
    W_conv2_1x1_1 = createWeight([1,1,4*map1,map2],'W_conv2_1x1_1')
    b_conv2_1x1_1 = createWeight([map2],'b_conv2_1x1_1')
 
    #follows inception1
    W_conv2_1x1_2 = createWeight([1,1,4*map1,reduce1x1],'W_conv2_1x1_2')
    b_conv2_1x1_2 = createWeight([reduce1x1],'b_conv2_1x1_2')
 
    #follows inception1
    W_conv2_1x1_3 = createWeight([1,1,4*map1,reduce1x1],'W_conv2_1x1_3')
    b_conv2_1x1_3 = createWeight([reduce1x1],'b_conv2_1x1_3')
 
    #follows 1x1_2
    W_conv2_3x3 = createWeight([3,3,reduce1x1,map2],'W_conv2_3x3')
    b_conv2_3x3 = createWeight([map2],'b_conv2_3x3')
 
    #follows 1x1_3
    W_conv2_5x5 = createWeight([5,5,reduce1x1,map2],'W_conv2_5x5')
    b_conv2_5x5 = createBias([map2],'b_conv2_5x5')
 
    #follows max pooling
    W_conv2_1x1_4= createWeight([1,1,4*map1,map2],'W_conv2_1x1_4')
    b_conv2_1x1_4= createWeight([map2],'b_conv2_1x1_4')
 
    #Fully connected layers
    #since padding is same, the feature map with there will be 4 28*28*map2
    W_fc1 = createWeight([28*28*(4*map2),num_fc1],'W_fc1')
    b_fc1 = createBias([num_fc1],'b_fc1')
 
    W_fc2 = createWeight([num_fc1,num_fc2],'W_fc2')
    b_fc2 = createBias([num_fc2],'b_fc2')
 
    def model(x,train=True):
        #Inception Module 1
        conv1_1x1_1 = conv2d_s1(x,W_conv1_1x1_1)+b_conv1_1x1_1
        conv1_1x1_2 = tf.nn.relu(conv2d_s1(x,W_conv1_1x1_2)+b_conv1_1x1_2)
        conv1_1x1_3 = tf.nn.relu(conv2d_s1(x,W_conv1_1x1_3)+b_conv1_1x1_3)
        conv1_3x3 = conv2d_s1(conv1_1x1_2,W_conv1_3x3)+b_conv1_3x3
        conv1_5x5 = conv2d_s1(conv1_1x1_3,W_conv1_5x5)+b_conv1_5x5
        maxpool1 = max_pool_3x3_s1(x)
        conv1_1x1_4 = conv2d_s1(maxpool1,W_conv1_1x1_4)+b_conv1_1x1_4
 
        #concatenate all the feature maps and hit them with a relu
        inception1 = tf.nn.relu(tf.concat([conv1_1x1_1,conv1_3x3,conv1_5x5,conv1_1x1_4],3))
 
        #Inception Module 2
        conv2_1x1_1 = conv2d_s1(inception1,W_conv2_1x1_1)+b_conv2_1x1_1
        conv2_1x1_2 = tf.nn.relu(conv2d_s1(inception1,W_conv2_1x1_2)+b_conv2_1x1_2)
        conv2_1x1_3 = tf.nn.relu(conv2d_s1(inception1,W_conv2_1x1_3)+b_conv2_1x1_3)
        conv2_3x3 = conv2d_s1(conv2_1x1_2,W_conv2_3x3)+b_conv2_3x3
        conv2_5x5 = conv2d_s1(conv2_1x1_3,W_conv2_5x5)+b_conv2_5x5
        maxpool2 = max_pool_3x3_s1(inception1)
        conv2_1x1_4 = conv2d_s1(maxpool2,W_conv2_1x1_4)+b_conv2_1x1_4
 
        #concatenate all the feature maps and hit them with a relu
        inception2 = tf.nn.relu(tf.concat([conv2_1x1_1,conv2_3x3,conv2_5x5,conv2_1x1_4],3))
 
        #flatten features for fully connected layer
        inception2_flat = tf.reshape(inception2,[-1,28*28*4*map2])
 
        #Fully connected layers
        if train:
            h_fc1 =tf.nn.dropout(tf.nn.relu(tf.matmul(inception2_flat,W_fc1)+b_fc1),dropout)
        else:
            h_fc1 = tf.nn.relu(tf.matmul(inception2_flat,W_fc1)+b_fc1)
 
        return tf.matmul(h_fc1,W_fc2)+b_fc2
 
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits = model(X),labels = y_))
    opt = tf.train.AdamOptimizer(1e-4).minimize(loss)
 
    # predictions_val = tf.nn.softmax(model(tf_valX,train=False))
    predictions_test = tf.nn.softmax(model(tf_testX,train=False))
 
    #initialize variable
    init = tf.initialize_all_variables()
 
    #use to save variables so we can pick up later
    saver = tf.train.Saver()

num_steps = 20000
sess = tf.Session(graph=graph)
 
#initialize variables
sess.run(init)
print("Model initialized.")
 
#set use_previous=1 to use file_path model
#set use_previous=0 to start model from scratch
use_previous = 0
 
#use the previous model or don't and initialize variables
if use_previous:
    saver.restore(sess,'gs://cmb/mnist/sign.ckpt')
    print("Model restored.")
 
def next_batch(batch_size, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[: batch_size]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

#training
for s in range(num_steps):
    '''
    offset = (s*batch_size) % (len(trainX)-batch_size)
    batch_x,batch_y = trainX[offset:(offset+batch_size),:],train_lb[offset:(offset+batch_size),:]
    '''
    batch_x, batch_y = next_batch(batch_size, trainX, train_lb)
    feed_dict={X : batch_x, y_ : batch_y}
    _,loss_value = sess.run([opt,loss],feed_dict=feed_dict)
    '''
    if s%100 == 0:
        feed_dict = {tf_valX : valX}
        preds=sess.run(predictions_val,feed_dict=feed_dict)
 
        print "step: "+str(s)
        print "validation accuracy: "+str(accuracy(val_lb,preds))
        print " "
    '''
    #get test accuracy and save model
    if s == (num_steps-1):
        #create an array to store the outputs for the test
        result = np.array([]).reshape(0,26)
 
        #use the batches class
        batch_testX=test_batchs(testX)
 
        for i in range(len(testX)/test_batch_size):
            feed_dict = {tf_testX : batch_testX.nextBatch(test_batch_size)}
            preds=sess.run(predictions_test, feed_dict=feed_dict)
            result=np.concatenate((result,preds),axis=0)
 
        print "test accuracy: "+str(accuracy(test_lb,result))
        
        save_path = saver.save(sess,'gs://cmb/mnist/sign.ckpt')
        print("Model saved.")
        