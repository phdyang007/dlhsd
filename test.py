from model import *
import configparser as cp
import sys
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[2])
from progress.bar import Bar

'''
Initialize Path and Global Params
'''
infile = cp.SafeConfigParser()
infile.read(sys.argv[1])

test_path   = infile.get('dir','test_path')
test_list = open(test_path).readlines()
max_len = len(test_list)
model_path = infile.get('dir','model_path')
fealen     = int(infile.get('feature','ft_length'))
blockdim   = int(infile.get('feature','block_dim'))
imgdim   = int(infile.get('feature','img_dim'))


'''
Prepare the Input
'''


dct_kernel = get_dct_kernel(imgdim//blockdim, fealen)
x_data = tf.placeholder(tf.float32, shape=[None, imgdim, imgdim, 1])              #input FT
y_gt   = tf.placeholder(tf.float32, shape=[None, 2])                                      #ground truth label
                                     #ground truth label without bias
                            #reshap to NHWC
predict= forward_spie(x_data, is_training=False)    
y_gt   = tf.placeholder(tf.float32, shape=[None, 2])                                      #ground truth label

y      = tf.cast(tf.argmax(predict, 1), tf.int32)                                         
accu   = tf.equal(y, tf.cast(tf.argmax(y_gt, 1), tf.int32))                               #calc batch accu
accu   = tf.reduce_mean(tf.cast(accu, tf.float32))
'''
Start testing
'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4


ckpt = tf.train.get_checkpoint_state(model_path)
if ckpt and ckpt.model_checkpoint_path:
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    print(ckpt_name)


with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    saver    = tf.train.Saver()
    saver.restore(sess, os.path.join(model_path, ckpt_name))
    chs = 0   #correctly predicted hs
    cnhs= 0   #correctly predicted nhs
    ahs = 0   #actual hs
    anhs= 0   #actual hs
    start   = time.time()
    bar = Bar('Detecting', max=max_len)
    for titr in range(0, max_len):
        #if not titr == test_data.maxlen//1000:


        t = get_one(test_list, titr)
        tdata = t[0]
        tlabel= t[1]
        tmp_y    = y.eval(feed_dict={x_data: tdata})
        #tmp_label= np.argmax(tlabel, axis=1)
        tmp      = tlabel+tmp_y
        chs += (tmp==2)
        cnhs+= (int(tmp)==0)
        ahs += tlabel
        anhs+= tlabel==0
        bar.next()
    bar.finish()
    print (chs, ahs, cnhs, anhs)
    if not ahs ==0:
        hs_accu = 1.0*chs/ahs
    else:
        hs_accu = 0
    fs      = anhs - cnhs
    end       = time.time()
print (ahs, anhs)
print('Hotspot Detection Accuracy is %f'%hs_accu)
print('False Alarm is %f'%fs)
print('Test Runtime is %f seconds'%(end-start))
    



