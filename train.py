from model import *
import ConfigParser as cp
import sys
import os
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[2])
'''
Initialize Path and Global Params
'''
infile = cp.SafeConfigParser()
infile.read(sys.argv[1])
train_path = infile.get('dir','train_path')

save_path = infile.get('dir','save_path')
fealen     = int(infile.get('feature','ft_length'))
blockdim   = int(infile.get('feature','block_dim'))
aug        = int(infile.get('feature','aug'))
'''
Prepare the Optimizer
'''
train_data = data(train_path, train_path+'/label.csv', preload=True)

x_data = tf.placeholder(tf.float32, shape=[None, blockdim*blockdim, fealen])              #input FT
y_gt   = tf.placeholder(tf.float32, shape=[None, 2])                                      #ground truth label
                                     #ground truth label without bias
x      = tf.reshape(x_data, [-1, blockdim, blockdim, fealen])                             #reshap to NHWC
if aug==1:
    predict= forward(x, flip=True)   
else:
    predict= forward(x)                                                        #do forward
loss   = tf.nn.softmax_cross_entropy_with_logits(labels=y_gt, logits=predict) 
loss   = tf.reduce_mean(loss)                                                             #calc batch loss
                                                          #calc batch loss without bias
y      = tf.cast(tf.argmax(predict, 1), tf.int32)                                         
accu   = tf.equal(y, tf.cast(tf.argmax(y_gt, 1), tf.int32))                                                    #calc batch accu
accu   = tf.reduce_mean(tf.cast(accu, tf.float32))
gs     = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)       #define global step
#lr     = tf.train.exponential_decay(0.001, gs, decay_steps=20000, decay_rate = 0.65, staircase = True) #initial learning rate and lr decay
lr     = tf.constant(0.0003)
dr     = 0.65 #learning rate decay rate
opt    = tf.train.AdamOptimizer(lr, beta1=0.9)
opt    = opt.minimize(loss, gs)
maxitr = 10000
bs     = 32   #training batch size

l_step = 5    #display step
c_step = 500 #check point step
d_step = 20000 #lr decay step
ckpt   = True

'''
Start the training
'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.44
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    saver    = tf.train.Saver(max_to_keep=400)

    for step in xrange(maxitr):
        batch = train_data.sgd_batch(bs, fealen)
        batch_data = batch[0]
        batch_label= batch[1]
        batch_nhs  = batch[2]
        batch_label_all_without_bias = processlabel(batch_label)
        batch_label_nhs_without_bias = processlabel(batch[3])
        nhs_loss = loss.eval(feed_dict={x_data: batch_nhs, y_gt: batch_label_nhs_without_bias})
        delta1 = loss_to_bias(nhs_loss, 6,)
        batch_label_all_with_bias = processlabel(batch_label, delta1 = delta1)
        training_loss, learning_rate, training_acc = \
            loss.eval(feed_dict={x_data: batch_data, y_gt: batch_label_all_without_bias}), \
            sess.run(lr), accu.eval(feed_dict={x_data:batch_data, y_gt:batch_label_all_without_bias})
        opt.run(feed_dict={x_data: batch_data, y_gt: batch_label_all_with_bias})
        if step % l_step == 0:
            format_str = ('%s: step %d, loss = %.2f, learning_rate = %f, training_accu = %f, nhs_loss = %.2f, bias = %.2f')
            print (format_str % (datetime.now(), step, training_loss, learning_rate, training_acc, nhs_loss, delta1))
        if step % c_step == 0 and ckpt and step > 0:
            path = save_path + 'model-'+str(step)+'-'+'.ckpt'
            saver.save(sess, path)
        if step % d_step == 0 and step >0:
            lr = lr * dr
