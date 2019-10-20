import os
import string
import numpy as np
from itertools import islice
import random
import csv
import tensorflow as tf
import tensorflow.contrib.slim as slim
from scipy.misc import imread
from time import time
import json
import pandas as pd 
import math
from PIL import Image
from scipy.fftpack import dct


#
#
#
#

def get_feature(files, blocksize, blockdim, fealen):
    im = Image.open(files)
    #print("processing files %s"%files)
    imdata=np.asarray(im.convert('L'))
    tempfeature=feature(imdata, blocksize, blockdim, fealen)
    return np.rollaxis(tempfeature, 0, 3)

def quantization(size, val=1):
    return np.empty(size*size, dtype=int).reshape(size,size).fill(val)

def rescale(img):
    return (img/255)

#calculate 2D DCT of a matrix
def dct2(img):
    return dct(dct(img.T, norm = 'ortho').T, norm = 'ortho')

def subfeature(imgraw, fealen):

    if fealen > len(imgraw)*len(imgraw[:,0]):
        print ('ERROR: Feature vector length exceeds block size.')
        print ('Abort.')
        quit()

    img =dct2(imgraw)
    size=fealen
    idx =0
    scaled=img
    feature=np.zeros(fealen, dtype=np.int)
    for i in range(0, size):
        if idx>=size:
            break
        elif i==0:
            feature[0]=scaled[0,0]
            idx=idx+1
        elif i%2==1:
            for j in range(0, i+1):
                if idx<size:
                    feature[idx]=scaled[j, i-j]
                    idx=idx+1
                else:
                    break
        elif i%2==0:
            for j in range(0, i+1):
                if idx<size:
                    feature[idx]=scaled[i-j, j]
                    idx=idx+1
                else:
                    break

    return feature


def cutblock(img, block_size, block_dim):
    blockarray=[]
    for i in range(0, block_dim):
        blockarray.append([])
        for j in range(0, block_dim):
            blockarray[i].append(img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size])

    return np.asarray(blockarray)


def feature(img, block_size, block_dim, fealen):
    img=rescale(img)
    feaarray = np.empty(fealen*block_dim*block_dim).reshape(fealen, block_dim, block_dim)
    blocked = cutblock(img, block_size, block_dim)
    for i in range(0, block_dim):
        for j in range(0, block_dim):
            featemp=subfeature(blocked[i,j], fealen)
            feaarray[:,i,j]=featemp
    return feaarray






'''
    readcsv: Read feature tensors from csv data packet
    args:
        target: the directory that stores the csv files
        fealen: the length of feature tensor, related to to discarded DCT coefficients
    returns: (1) numpy array of feature tensors with shape: N x H x W x C
             (2) numpy array of labels with shape: N x 1 
'''
def readcsv_(target, fealen=32):
    #read label
    path  = target + '/label.csv'
    label = np.genfromtxt(path, delimiter=',')
    #read feature
    feature = []
    for dirname, dirnames, filenames in os.walk(target):
        for i in range(0, len(filenames)-1):
            if i==0:
                file = '/dc.csv'
                path = target + file
                featemp = np.genfromtxt(path, delimiter=',')
                feature.append(featemp)
            else:
                file = '/ac'+str(i)+'.csv'
                path = target + file
                featemp = np.genfromtxt(path, delimiter=',')
                feature.append(featemp)          
    return np.rollaxis(np.asarray(feature), 0, 3)[:,:,0:fealen], label
def readcsv(target, fealen=32):
    #read label
    path  = target + '/label.csv'
    label = np.genfromtxt(path, delimiter=',')
    #read feature
    feature = []
    for dirname, dirnames, filenames in os.walk(target):
        for i in range(0, len(filenames)-1):
            if i==0:
                file = '/dc.csv'
                path = target + file
                featemp = pd.read_csv(path, header=None).as_matrix()
                feature.append(featemp)
            else:
                file = '/ac'+str(i)+'.csv'
                path = target + file
                featemp = pd.read_csv(path, header=None).as_matrix()
                feature.append(featemp)          
    return np.rollaxis(np.asarray(feature), 0, 3)[:,:,0:fealen], label
'''
    processlabel: adjust ground truth for biased learning
    args:
        label: numpy array contains labels
        cato : number of classes in the task
        delta1: bias for class 1
        delta2: bias for class 2
    return: softmax label with bias
'''
def processlabel(label, cato=2, delta1 = 0, delta2=0):
    softmaxlabel=np.zeros(len(label)*cato, dtype=np.float32).reshape(len(label), cato)
    for i in range(0, len(label)):
        if int(label[i])==0:
            softmaxlabel[i,0]=1-delta1
            softmaxlabel[i,1]=delta1
        if int(label[i])==1:
            softmaxlabel[i,0]=delta2
            softmaxlabel[i,1]=1-delta2
    return softmaxlabel
'''
    loss_to_bias: calculate the bias term for batch biased learning
    args:
        loss: the average loss of current batch with respect to the label without bias
        threshold: start biased learning when loss is below the threshold
    return: the bias value to calculate the gradient
'''
def loss_to_bias(loss,  alpha, threshold=0.3):
    if loss >= threshold:
        bias = 0
    else:
        bias = 1.0/(1+np.exp(alpha*loss))
    return bias

'''
    forward: define the neural network architecute
    args:
        input: feature tensor batch with size B x H x W x C
        is_training: whether the forward process is training, affect dropout layer
        reuse: undetermined
        scope: undetermined
    return: prediction socre(s) of input batch
'''
def get_one(test_file_list, id):
    img = imread(test_file_list[id].split()[0])
    label = int(test_file_list[id].split()[1])

    return np.expand_dims(np.expand_dims(img, axis = 0), axis = -1), label

def get_data(train_file_list):
    datalen = len(train_file_list)
    datalist = []
    labellist = []

    for i in range(datalen):
        datalist.append(train_file_list[i].split()[0])
        labellist.append(int(train_file_list[i].split()[1]))
    
    return np.array(datalist), np.array(labellist)


def get_batch(data_list, label_list, batch_size):

    hs_idx = np.where(label_list==1)[0]
    nhs_idx = np.where(label_list==0)[0]


    half_bs = batch_size//2

    hs_batch = random.sample(range(len(hs_idx)), half_bs)
    nhs_batch = random.sample(range(len(nhs_idx)), half_bs)

    datalist = np.concatenate((data_list[hs_idx[hs_batch]], data_list[nhs_idx[nhs_batch]]))
    batch_label = np.zeros(batch_size)
    batch_label[:half_bs]=1
    for i in range(batch_size):
        tmp = imread(datalist[i])
        tmp = np.expand_dims(tmp, axis = 0)
        if i == 0:
            batch_data = tmp
        else:
            batch_data = np.concatenate((batch_data, tmp), axis = 0)

    batch_data = batch_data/255



    batch_data_nhs = batch_data[half_bs:]
    batch_data_label_nhs = batch_label[half_bs:]

    return np.expand_dims(batch_data, axis=-1), batch_label, np.expand_dims(batch_data_nhs, axis=-1), batch_data_label_nhs

def get_dct_kernel(block_size, fealen):
    kernel = np.zeros((block_size, block_size, fealen), dtype=float)
    c=0
    for i in range(block_size):
            for j in range(i+1):
                if c<fealen:
                    #print(c)
                    for x in range(block_size):
                        for y in range(block_size):
                            kernel[x,y,c]=math.cos(math.pi/block_size*(x+0.5)*i)*math.cos(math.pi/block_size*(y+0.5)*j)
                    c+=1

    
    print("dct_kernel_size is ", kernel.shape)
    return np.expand_dims(kernel, axis = 2)

def forward_dct(input, dct_kernel, is_training=True, reuse=tf.AUTO_REUSE, scope='model'):

    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, stride=1, padding='SAME',
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                            biases_initializer=tf.constant_initializer(0.0)):

            net = tf.nn.conv2d(input, dct_kernel, strides = [1, dct_kernel.shape[0], dct_kernel.shape[1], 1], padding = 'VALID')
            net = slim.conv2d(net, 16, [3, 3], scope='conv1_1')
            net = slim.conv2d(net, 16, [3, 3], scope='conv1_2')
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool1')
            net = slim.conv2d(net, 32, [3, 3], scope='conv2_1')
            net = slim.conv2d(net, 32, [3, 3], scope='conv2_2')
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool2')
            net = slim.flatten(net)
            w_init = tf.contrib.layers.xavier_initializer(uniform=False)
            net = slim.fully_connected(net, 250, activation_fn=tf.nn.relu, scope='fc1')
            net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout')
            predict = slim.fully_connected(net, 2, activation_fn=None, scope='fc2')
    return predict



def forward_spie(input, is_training=True, reuse=tf.AUTO_REUSE, scope='model'):

    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, stride=1, padding='SAME',
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                            biases_initializer=tf.constant_initializer(0.0)):
            net = slim.conv2d(input, 4, kernel_size=[3,3], stride=2, padding='SAME', scope='conv0_1')
            net = slim.conv2d(net, 4, kernel_size=[3,3], stride=2, padding='SAME', scope='conv0_2')
            #net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool0_1')
            for cs in range(3):
                for ci in range(3):
                    net = slim.conv2d(net, 8*(2**cs), [3, 3], scope='conv'+str(cs+1)+'_'+str(ci+1))
                    #net = slim.batch_norm(net, center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training, updates_collections=None, reuse=reuse, scope='bn'+str(cs+1)+'_'+str(ci+1))
                net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool'+str(cs+1))
            for ci in range(3):
                net = slim.conv2d(net, 32, [3, 3], scope='conv5_'+str(ci+1))
            net   = slim.max_pool2d(net, [2, 2], stride=2, scope='pool6')
            # TODO: complete the baseline model structure


            # Flatten the feature map of pool6 into a feature vector
            net = slim.flatten(net)
            # Please check them carefully when you modify
            w_init = tf.contrib.layers.xavier_initializer(uniform=False)
            net = slim.fully_connected(net, 1024, scope='fc7', activation_fn=tf.nn.relu)
            #fc7 = slim.batch_norm(fc7, center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training, updates_collections=None, reuse=reuse, scope='bnfc7')
            net = slim.dropout(net, keep_prob=0.5, is_training=is_training, scope='drop7')
            net = slim.fully_connected(net, 256, scope='fc8', activation_fn=tf.nn.relu)
            #fc8 = slim.batch_norm(fc8, center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training, updates_collections=None, reuse=reuse, scope='bnfc8')
            net = slim.dropout(net, keep_prob=0.5, is_training=is_training, scope='drop8')
            predict = slim.fully_connected(net, 2, activation_fn=None, scope='predict')
    return predict



def forward(input, is_training=True, reuse=tf.AUTO_REUSE, scope='model'):

    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, stride=1, padding='SAME',
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                            biases_initializer=tf.constant_initializer(0.0)):

            #net = tf.nn.conv2d(input, dct_kernel, strides = [1, dct_kernel.shape[0], dct_kernel.shape[1], 1], padding = 'SAME')
            net = slim.conv2d(input, 16, [3, 3], scope='conv1_1')
            net = slim.conv2d(net, 16, [3, 3], scope='conv1_2')
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool1')
            net = slim.conv2d(net, 32, [3, 3], scope='conv2_1')
            net = slim.conv2d(net, 32, [3, 3], scope='conv2_2')
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool2')
            net = slim.flatten(net)
            w_init = tf.contrib.layers.xavier_initializer(uniform=False)
            net = slim.fully_connected(net, 250, activation_fn=tf.nn.relu, scope='fc1')
            net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout')
            predict = slim.fully_connected(net, 2, activation_fn=None, scope='fc2')
    return predict

'''
    data: a class to handle the training and testing data, implement minibatch fetch
    args: 
        fea: feature tensor of whole data set
        lab: labels of whole data set
        ptr: a pointer for the current location of minibatch
        maxlen: length of entire dataset
        preload: in current version, to reduce the indexing overhead of SGD, we load all the data into memeory at initialization.
    methods:
        nextinstance():  returns a single instance and its label from the training set, used for SGD
        nextbatch(): returns a batch of instances and their labels from the training set, used for MGD
            args: 
                batch: minibatch number
                channel: the channel length of feature tersor, lenth > channel will be discarded
                delta1, delta2: see process_label
        sgd_batch(): returns a batch of instances and their labels from the trainin set randomly, number of hs and nhs are equal.
            args:
                batch: minibatch number
                channel: the channel length of feature tersor, lenth > channel will be discarded
                delta1, delta2: see process_label

'''
class data:
    def __init__(self, fea, lab, preload=False):
        self.ptr_n=0
        self.ptr_h=0
        self.ptr=0
        self.dat=fea
        self.label=lab
        with open(lab) as f:
            self.maxlen=sum(1 for _ in f)
        if preload:
            print("loading data into the main memory...")
            self.ft_buffer, self.label_buffer=readcsv(self.dat)

    def nextinstance(self):
        temp_fea=[]
        label=None
        idx=random.randint(0,self.maxlen)
        for dirname, dirnames, filenames in os.walk(self.dat):
            for i in range(0, len(filenames)-1):
                    if i==0:
                        file='/dc.csv'
                        path=self.dat+file
                        with open(path) as f:
                            r=csv.reader(f)
                            fea=[[int(s) for s in row] for j,row in enumerate(r) if j==idx]
                            temp_fea.append(np.asarray(fea))
                    else:
                        file='/ac'+str(i)+'.csv'
                        path=self.dat+file
                        with open(path) as f:
                            r=csv.reader(f)
                            fea=[[int(s) for s in row] for j,row in enumerate(r) if j==idx]
                            temp_fea.append(np.asarray(fea))        
        with open(self.label) as l:
            temp_label=np.asarray(list(l)[idx]).astype(int)
            if temp_label==0:
                label=[1,0]
            else:
                label=[0,1]
        return np.rollaxis(np.array(temp_fea),0,3),np.array([label])

    def sgd(self, channel=None, delta1=0, delta2=0):
        with open(self.label) as l:
            labelist=np.asarray(list(l)).astype(int)
        length=labelist.size
        idx=random.randint(0, length-1)
        temp_label=labelist[idx]
        if temp_label==0:
            label=[1,0]
        else:
            label=[0,1]
        ft= self.ft_buffer[idx]

        return ft, np.array(label)
    def sgd_batch_2(self, batch, channel=None, delta1=0, delta2=0):
        with open(self.label) as l:
            labelist=np.asarray(list(l)).astype(int)
            labexn = np.where(labelist==0)[0]
            labexh = np.where(labelist==1)[0]
        n_length = labexn.size
        h_length = labexh.size
        if not batch % 2 == 0:
            print('ERROR:Batch size must be even')
            print('Abort.')
            quit()
        else:
            num = batch // 2
        idxn = labexn[(np.random.rand(num)*n_length).astype(int)]
        idxh = labexh[(np.random.rand(num)*h_length).astype(int)]
        label = np.concatenate((np.zeros(num), np.ones(num)))
        label = processlabel(label,2, 0,0 )
        ft_batch = np.concatenate((self.ft_buffer[idxn], self.ft_buffer[idxh]))
        ft_batch_nhs = self.ft_buffer[idxn]
        label_nhs = np.zeros(num)
        return ft_batch, label


    def sgd_batch(self, batch, channel=None, delta1=0, delta2=0):
        with open(self.label) as l:
            labelist=np.asarray(list(l)).astype(int)
            labexn = np.where(labelist==0)[0]
            labexh = np.where(labelist==1)[0]
        n_length = labexn.size
        h_length = labexh.size
        if not batch % 2 == 0:
            print('ERROR:Batch size must be even')
            print('Abort.')
            quit()
        else:
            num = batch // 2
        idxn = labexn[(np.random.rand(num)*n_length).astype(int)]
        idxh = labexh[(np.random.rand(num)*h_length).astype(int)]
        label = np.concatenate((np.zeros(num), np.ones(num)))
        #label = processlabel(label,2, delta1, delta2)
        ft_batch = np.concatenate((self.ft_buffer[idxn], self.ft_buffer[idxh]))
        ft_batch_nhs = self.ft_buffer[idxn]
        label_nhs = np.zeros(num)
        return ft_batch, label, ft_batch_nhs, label_nhs
    '''
    nextbatch_beta: returns the balalced batch, used for training only
    '''
    def nextbatch_beta(self, batch, channel=None, delta1=0, delta2=0):
        def update_ptr(ptr, batch, length):
            if ptr+batch<length:
                ptr+=batch
            if ptr+batch>=length:
                ptr=ptr+batch-length
            return ptr
        with open(self.label) as l:
            labelist=np.asarray(list(l)).astype(int)
            labexn = np.where(labelist==0)[0]
            labexh = np.where(labelist==1)[0]
        n_length = labexn.size
        h_length = labexh.size

        if not batch % 2 == 0:
            print('ERROR:Batch size must be even')
            print('Abort.')
            quit()
        else:
            num = batch//2
            if num>=n_length or num>=h_length:
                print('ERROR:Batch size exceeds data size')
                print('Abort.')
                quit()
            else:
                if self.ptr_n+num <n_length:
                    idxn = labexn[self.ptr_n:self.ptr_n+num]
                elif self.ptr_n+num >=n_length:
                    idxn = np.concatenate((labexn[self.ptr_n:n_length], labexn[0:self.ptr_n+num-n_length]))
                self.ptr_n = update_ptr(self.ptr_n, num, n_length)
                if self.ptr_h+num <h_length:
                    idxh = labexh[self.ptr_h:self.ptr_h+num]
                elif self.ptr_h+num >=h_length:
                    idxh = np.concatenate((labexh[self.ptr_h:h_length], labexh[0:self.ptr_h+num-h_length]))
                self.ptr_h = update_ptr(self.ptr_h, num, h_length)
                #print self.ptr_n, self.ptr_h
                label = np.concatenate((np.zeros(num), np.ones(num)))
                #label = processlabel(label,2, delta1, delta2)
                ft_batch = np.concatenate((self.ft_buffer[idxn], self.ft_buffer[idxh]))
                ft_batch_nhs = self.ft_buffer[idxn]
                label_nhs = np.zeros(num)
        return ft_batch, label, ft_batch_nhs, label_nhs
    '''
    nextbatch_without_balance: returns the normal batch. Suggest to use for training and validation
    '''
    def nextbatch_without_balance_alpha(self, batch, channel=None, delta1=0, delta2=0):
        def update_ptr(ptr, batch, length):
            if ptr+batch<length:
                ptr+=batch
            if ptr+batch>=length:
                ptr=ptr+batch-length
            return ptr
        if self.ptr + batch < self.maxlen:
            label = self.label_buffer[self.ptr:self.ptr+batch]
            ft_batch = self.ft_buffer[self.ptr:self.ptr+batch]
        else:
            label = np.concatenate((self.label_buffer[self.ptr:self.maxlen], self.label_buffer[0:self.ptr+batch-self.maxlen]))
            ft_batch = np.concatenate((self.ft_buffer[self.ptr:self.maxlen], self.ft_buffer[0:self.ptr+batch-self.maxlen]))
        self.ptr = update_ptr(self.ptr, batch, self.maxlen)
        return ft_batch, label
    def nextbatch(self, batch, channel=None, delta1=0, delta2=0):
        #print('recommed to use nextbatch_beta() instead')
        databat=None
        temp_fea=[]
        label=None
        if batch>self.maxlen:
            print('ERROR:Batch size exceeds data size')
            print('Abort.')
            quit()
        if self.ptr+batch < self.maxlen:
            #processing labels
            with open(self.label) as l:
                temp_label=np.asarray(list(l)[self.ptr:self.ptr+batch])
                label=processlabel(temp_label, 2, delta1, delta2)
            for dirname, dirnames, filenames in os.walk(self.dat):
                for i in range(0, len(filenames)-1):
                    if i==0:
                        file='/dc.csv'
                        path=self.dat+file
                        with open(path) as f:
                            temp_fea.append(np.genfromtxt(islice(f, self.ptr, self.ptr+batch),delimiter=','))
                    else:
                        file='/ac'+str(i)+'.csv'
                        path=self.dat+file
                        with open(path) as f:
                            temp_fea.append(np.genfromtxt(islice(f, self.ptr, self.ptr+batch),delimiter=','))
            self.ptr=self.ptr+batch
        elif (self.ptr+batch) >= self.maxlen:
            
            #processing labels
            with open(self.label) as l:
                a=np.genfromtxt(islice(l, self.ptr, self.maxlen),delimiter=',')
            with open(self.label) as l:
                b=np.genfromtxt(islice(l, 0, self.ptr+batch-self.maxlen),delimiter=',')
            #processing data
            if self.ptr==self.maxlen-1 or self.ptr==self.maxlen:
                temp_label=b
            elif self.ptr+batch-self.maxlen==1 or self.ptr+batch-self.maxlen==0:
                temp_label=a
            else:
                temp_label=np.concatenate((a,b))
            label=processlabel(temp_label,2, delta1, delta2)
            #print label.shape
            for dirname, dirnames, filenames in os.walk(self.dat):
                for i in range(0, len(filenames)-1):
                    if i==0:
                        file='/dc.csv'
                        path=self.dat+file
                        with open(path) as f:
                            a=np.genfromtxt(islice(f, self.ptr, self.maxlen),delimiter=',')
                        with open(path) as f:
                            b=np.genfromtxt(islice(f, None, self.ptr+batch-self.maxlen),delimiter=',')
                        if self.ptr==self.maxlen-1 or self.ptr==self.maxlen:
                            temp_fea.append(b)
                        elif self.ptr+batch-self.maxlen==1 or self.ptr+batch-self.maxlen==0:
                            temp_fea.append(a)
                        else:
                            try:
                                temp_fea.append(np.concatenate((a,b)))
                            except:
                                print (a.shape, b.shape, self.ptr)
                    else:
                        file='/ac'+str(i)+'.csv'
                        path=self.dat+file
                        with open(path) as f:
                            a=np.genfromtxt(islice(f, self.ptr, self.maxlen),delimiter=',')
                        with open(path) as f:
                            b=np.genfromtxt(islice(f, 0, self.ptr+batch-self.maxlen),delimiter=',')
                        if self.ptr==self.maxlen-1 or self.ptr==self.maxlen:
                            temp_fea.append(b)
                        elif self.ptr+batch-self.maxlen==1 or self.ptr+batch-self.maxlen==0:
                            temp_fea.append(a)
                        else:
                            try:
                                temp_fea.append(np.concatenate((a,b)))
                            except:
                                print (a.shape, b.shape, self.ptr)
            self.ptr=self.ptr+batch-self.maxlen
        #print np.asarray(temp_fea).shape
        return np.rollaxis(np.asarray(temp_fea), 0, 3)[:,:,0:channel], label
