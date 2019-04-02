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


model_path = infile.get('dir','model_path')
fealen     = int(infile.get('feature','ft_length'))
blockdim   = int(infile.get('feature','block_dim'))
aug=int(infile.get('feature','AUG'))

'''
Prepare the Input
'''
test_data = data(test_path, test_path+'/label.csv')
x_data = tf.placeholder(tf.float32, shape=[None, blockdim*blockdim, fealen])              #input FT
y_gt   = tf.placeholder(tf.float32, shape=[None, 2])                                      #ground truth label
x      = tf.reshape(x_data, [-1, blockdim, blockdim, fealen])                             #reshap to NHWC
x_ud   = tf.map_fn(lambda img: tf.image.flip_up_down(img), x)                   #up down flipped
x_lr   = tf.map_fn(lambda img: tf.image.flip_left_right(img), x)                #left right flipped
x_lu   = tf.map_fn(lambda img: tf.image.flip_up_down(img), x_lr)                #both flipped
predict_or = forward(x, is_training=False)                                      #do forward
predict_ud = forward(x_ud, is_training=False, reuse=True)  
predict_lr = forward(x_lr, is_training=False, reuse=True)
predict_lu = forward(x_lu, is_training=False, reuse=True)
if aug==1:
    predict = (predict_or + predict_lr + predict_lu + predict_ud)/4.0
else:
    predict = predict_or
#predict = predict_or
y      = tf.cast(tf.argmax(predict, 1), tf.int32)                                         
accu   = tf.equal(y, tf.cast(tf.argmax(y_gt, 1), tf.int32))                               #calc batch accu
accu   = tf.reduce_mean(tf.cast(accu, tf.float32))
'''
Start testing
'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    saver    = tf.train.Saver()
    saver.restore(sess, model_path)
    chs = 0   #correctly predicted hs
    cnhs= 0   #correctly predicted nhs
    ahs = 0   #actual hs
    anhs= 0   #actual hs
    start   = time.time()
    bar = Bar('Detecting', max=test_data.maxlen//1000+1)
    for titr in range(0, test_data.maxlen//1000+1):
        if not titr == test_data.maxlen//1000:
            tbatch = test_data.nextbatch(1000, fealen)
        else:
            tbatch = test_data.nextbatch(test_data.maxlen-titr*1000, fealen)
        tdata = tbatch[0]
        tlabel= tbatch[1]
        tmp_y    = y.eval(feed_dict={x_data: tdata, y_gt: tlabel})
        tmp_label= np.argmax(tlabel, axis=1)
        tmp      = tmp_label+tmp_y
        chs += sum(tmp==2)
        cnhs+= sum(tmp==0)
        ahs += sum(tmp_label)
        anhs+= sum(tmp_label==0)
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
    



